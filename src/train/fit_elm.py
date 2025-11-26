import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

from ..config import load_config
from ..dataio.window import cache_dcenn_windows
from ..models.dcenn import TinyDCENN

class RandomFeatureELM(nn.Module):
    # ... [Same Class Definition as before] ...
    def __init__(self, in_dim, out_dim, hidden=1024, act="tanh", ridge_lambda=1e-2, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.device = torch.device(device)
        self.dtype = torch.get_default_dtype()
        self.W = nn.Parameter(torch.randn(in_dim, hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden, generator=g, device=self.device, dtype=self.dtype) * 0.5, requires_grad=False)
        self.act = act
        self.ridge = float(ridge_lambda)
        self.beta = None
        self.out_dim = out_dim
    def _phi(self, XH):
        if self.act == "tanh": return torch.tanh(XH)
        if self.act == "relu": return torch.relu(XH)
        return XH
    def fit(self, X, Y):
        X = X.to(device=self.device, dtype=self.W.dtype)
        Y = Y.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        HtH = H.T @ H
        n = int(HtH.size(0))
        lamI = self.ridge * torch.eye(n, device=H.device, dtype=H.dtype)
        self.beta = torch.linalg.solve(HtH + lamI, H.T @ Y)
    def predict(self, X):
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        X = X.to(device=self.device, dtype=self.W.dtype)
        H = self._phi(X @ self.W + self.b)
        return H @ self.beta

def extract_latents(encoder, X_np, device="cpu", batch_size=256):
    encoder.eval()
    device = torch.device(device)
    X = torch.from_numpy(X_np).to(device)
    N = X.shape[0]
    outs = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i : i + batch_size]
            z = encoder(xb)
            outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)

def run(cfg_path: str):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    Xtr, Ytr, Xva, Yva, _, _ = cache_dcenn_windows(cfg)
    
    # 2. Scale Targets (Critical Step!)
    N_tr, H, C = Ytr.shape
    Ytr_flat = Ytr.reshape(-1, C)
    target_scaler = StandardScaler()
    Ytr_scaled = target_scaler.fit_transform(Ytr_flat).reshape(N_tr, H, C).astype(np.float32)
    
    # DEBUG: Print Scaler Stats
    print(f"[fit_elm] Target Means: {target_scaler.mean_}")
    print(f"[fit_elm] Target Scales: {target_scaler.scale_}")
    
    y_cf_wind_tr  = Ytr_scaled[:, :, 0]
    y_cf_solar_tr = Ytr_scaled[:, :, 1]
    y_load_tr     = Ytr_scaled[:, :, 2]
    y_price_tr    = Ytr_scaled[:, :, 3]

    # 3. Load Encoder & Extract Latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = Xtr.shape[2]
    enc = TinyDCENN(
        in_channels=in_channels,
        hidden_channels=cfg["training"]["encoder"]["latent_channels"],
    ).to(device)
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    
    print("[fit_elm] Extracting latents...")
    Ztr = extract_latents(enc, Xtr, device=device)
    
    # Standardize Latents
    mu = Ztr.mean(axis=0, keepdims=True)
    sigma = Ztr.std(axis=0, keepdims=True) + 1e-6
    Ztr_n = (Ztr - mu) / sigma

# 4. Prepare ELM Inputs with Skip Connections (Lags)
    # feature_cols indices from window.py:
    # 0-5: Calendar, 6: Holiday, 7-8: Caps, 
    # 9: wind_mw, 10: solar_mw, 11: load_mw, 12: price_eur_mwh
    
    # Extract last known values (t-1) and 24h ago (t-24)
    # Shapes: [N]
    
    # Wind Lags
    wind_lag1  = Xtr[:, -1, 9, 0, 0]
    wind_lag24 = Xtr[:, -24, 9, 0, 0]
    extra_wind = np.stack([wind_lag1, wind_lag24], axis=1) # [N, 2]

    # Solar Lags
    solar_lag1  = Xtr[:, -1, 10, 0, 0]
    solar_lag24 = Xtr[:, -24, 10, 0, 0]
    extra_solar = np.stack([solar_lag1, solar_lag24], axis=1)

    # Load Lags
    load_lag1  = Xtr[:, -1, 11, 0, 0]
    load_lag24 = Xtr[:, -24, 11, 0, 0]
    extra_load = np.stack([load_lag1, load_lag24], axis=1)

    # Price Lags (already had this, but ensuring consistency)
    price_lag1  = Xtr[:, -1, 12, 0, 0]
    price_lag24 = Xtr[:, -24, 12, 0, 0]
    extra_price = np.stack([price_lag1, price_lag24], axis=1)

    # Concatenate Latents + Lags
    # Note: Ztr_n is [N, 256]. Extra is [N, 2]. Result [N, 258].
    X_wind_tr  = np.concatenate([Ztr_n, extra_wind], axis=1)
    X_solar_tr = np.concatenate([Ztr_n, extra_solar], axis=1)
    X_load_tr  = np.concatenate([Ztr_n, extra_load], axis=1)
    X_price_tr = np.concatenate([Ztr_n, extra_price], axis=1)

    # 5. Fit ELMs
    elm_cfg = cfg["training"]["elm"]
    hidden = elm_cfg["hidden"]
    ridge = elm_cfg["ridge_lambda"]
    ensemble_size = elm_cfg.get("ensemble", 1)
    out_dim = H

    models_wind, models_solar, models_load, models_price = [], [], [], []

    # Helper for ensemble fitting
    def fit_ensemble(X, Y, model_list, name):
        print(f"Fitting {name} ensemble...")
        X_t = torch.from_numpy(X.astype(np.float32))
        Y_t = torch.from_numpy(Y.astype(np.float32))
        for k in range(ensemble_size):
            elm = RandomFeatureELM(
                in_dim=X.shape[1], out_dim=out_dim, hidden=hidden,
                act="tanh", ridge_lambda=ridge, seed=k, device="cpu"
            )
            elm.fit(X_t, Y_t)
            model_list.append(elm)
            
        # DEBUG: Check Training MAE on first model
        with torch.no_grad():
            pred = model_list[0].predict(X_t).cpu().numpy()
            mae = np.mean(np.abs(pred - Y.astype(np.float32)))
            print(f"  > {name} (Model 0) Train MAE (Scaled): {mae:.4f}")

    fit_ensemble(X_wind_tr,  y_cf_wind_tr,  models_wind,  "Wind")
    fit_ensemble(X_solar_tr, y_cf_solar_tr, models_solar, "Solar")
    fit_ensemble(X_load_tr,  y_load_tr,     models_load,  "Load")
    fit_ensemble(X_price_tr, y_price_tr,    models_price, "Price")

    # 6. Save
    out_dir = Path(cfg["paths"]["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "elms.joblib"
    
    save_obj = {
        "mu": mu,
        "sigma": sigma,
        "elm_wind": models_wind,
        "elm_solar": models_solar,
        "elm_load": models_load,
        "elm_price": models_price,
        "target_scaler": target_scaler # <--- Save the scaler
    }
    joblib.dump(save_obj, save_path)
    print(f"[fit_elm] Saved ELMs and target scaler to {save_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); run(args.config)
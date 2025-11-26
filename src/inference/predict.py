import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from ..config import load_config
from ..dataio.window import cache_dcenn_windows
from ..models.dcenn import TinyDCENN
from ..train.fit_elm import extract_latents, RandomFeatureELM

def _ensemble_predict(models, X_np):
    preds = []
    for m in models:
        with torch.no_grad():
            y = m.predict(torch.from_numpy(X_np.astype(np.float32)))
            preds.append(y.cpu().numpy())
    return np.mean(preds, axis=0)

def run(cfg_path: str):
    cfg = load_config(cfg_path)
    
    # 1. Load Data
    _, _, _, _, Xte, Yte = cache_dcenn_windows(cfg)
    
    # Reconstruct index (Basic approximation for saving)
    from ..dataio.preprocess import build_master
    _, _, test_df = build_master(cfg)
    ctx = cfg["features"]["context_hours"]
    hz = cfg["features"]["horizon_hours"]
    valid_starts = range(ctx, len(test_df) - hz)
    te_index = test_df.index[list(valid_starts)]
    
    # Length Check
    if len(te_index) != Xte.shape[0]:
        min_len = min(len(te_index), Xte.shape[0])
        te_index = te_index[:min_len]
        Xte = Xte[:min_len]

    # 2. Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = Xte.shape[2]
    enc = TinyDCENN(
        in_channels=in_channels,
        hidden_channels=cfg["training"]["encoder"]["latent_channels"],
    ).to(device)
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / "encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    
    Zte = extract_latents(enc, Xte, device=device)

    # 3. ELMs & Stats
    elms_path = Path(cfg["paths"]["checkpoints_dir"]) / "elms.joblib"
    saved = joblib.load(elms_path)
    mu = saved["mu"]
    sigma = saved["sigma"]
    target_scaler = saved["target_scaler"]
    
    Zte_n = (Zte - mu) / sigma
    
# 4. Extra Features (Skip Connections)
    # Indices: 9=Wind, 10=Solar, 11=Load, 12=Price
    
    # Wind
    w_lag1  = Xte[:, -1, 9, 0, 0]
    w_lag24 = Xte[:, -24, 9, 0, 0]
    extra_wind = np.stack([w_lag1, w_lag24], axis=1)

    # Solar
    s_lag1  = Xte[:, -1, 10, 0, 0]
    s_lag24 = Xte[:, -24, 10, 0, 0]
    extra_solar = np.stack([s_lag1, s_lag24], axis=1)

    # Load
    l_lag1  = Xte[:, -1, 11, 0, 0]
    l_lag24 = Xte[:, -24, 11, 0, 0]
    extra_load = np.stack([l_lag1, l_lag24], axis=1)

    # Price
    p_lag1  = Xte[:, -1, 12, 0, 0]
    p_lag24 = Xte[:, -24, 12, 0, 0]
    extra_price = np.stack([p_lag1, p_lag24], axis=1)

    # 5. Predict
    # Concatenate Latents + Lags
    X_wind_te  = np.concatenate([Zte_n, extra_wind], axis=1)
    X_solar_te = np.concatenate([Zte_n, extra_solar], axis=1)
    X_load_te  = np.concatenate([Zte_n, extra_load], axis=1)
    X_price_te = np.concatenate([Zte_n, extra_price], axis=1)
    
    cf_wind_te  = _ensemble_predict(saved["elm_wind"],  X_wind_te)
    cf_solar_te = _ensemble_predict(saved["elm_solar"], X_solar_te)
    load_te     = _ensemble_predict(saved["elm_load"],  X_load_te)
    price_te    = _ensemble_predict(saved["elm_price"], X_price_te)
    
    # 6. Inverse Scale
    N, H = cf_wind_te.shape
    preds_scaled = np.stack([cf_wind_te, cf_solar_te, load_te, price_te], axis=2)
    preds_flat = preds_scaled.reshape(-1, 4)
    preds_orig = target_scaler.inverse_transform(preds_flat).reshape(N, H, 4)
    
    cf_wind_final  = preds_orig[:, :, 0]
    cf_solar_final = preds_orig[:, :, 1]
    load_final     = preds_orig[:, :, 2]
    price_final    = preds_orig[:, :, 3]

# ---- NEW: PHYSICS & RULE ENFORCEMENT ----
    
    # 1. Solar Night Masking
    # We need the 'hour' of the prediction to know if it's night.
    # The 'te_index' holds the timestamps for the START of the window.
    # We need to broadcast this to the horizon.
    
    # Create an array of hours [N, H]
    # te_index is UTC. Convert to roughly local hour if needed, or just use sun elevation.
    # Simple heuristic: Night is approx hours 20,21,22,23,0,1,2,3,4,5
    # (Adjust based on your specific location/season, but this is a safe baseline)
    
    start_hours = te_index.hour.values  # [N]
    # We need to add +1, +2, ... +H for the horizon steps
    # hours_2d: [N, H]
    hours_2d = (start_hours[:, None] + np.arange(1, hz + 1)[None, :]) % 24
    
    # Define Night Hours (e.g., 9 PM to 5 AM)
    night_mask = (hours_2d >= 21) | (hours_2d <= 4)
    
    # Force Solar to 0 at night
    cf_solar_final[night_mask] = 0.0
    
    # 2. General Clamping (Physics)
    cf_wind_final = np.maximum(cf_wind_final, 0.0)
    cf_solar_final = np.maximum(cf_solar_final, 0.0)
    load_final = np.maximum(load_final, 0.0)
    # -----------------------------------------

    # 7. Convert CF to MW
    cap_w = test_df.loc[te_index, "cap_wind_mw"].values
    cap_s = test_df.loc[te_index, "cap_solar_mw"].values
    
    cap_w_2d = np.repeat(cap_w[:, None], H, axis=1)
    cap_s_2d = np.repeat(cap_s[:, None], H, axis=1)
    
    wind_mw_final  = cf_wind_final * cap_w_2d
    solar_mw_final = cf_solar_final * cap_s_2d

    # 8. Save
    cols = []
    blocks = []
    for h in range(hz):
        cols.extend([f"wind_mw+h{h+1}", f"solar_mw+h{h+1}", f"load_mw+h{h+1}", f"price+h{h+1}"])
        blocks.append(np.column_stack([
            wind_mw_final[:, h], solar_mw_final[:, h], load_final[:, h], price_final[:, h]
        ]))
        
    arr = np.concatenate(blocks, axis=1)
    pred_df = pd.DataFrame(arr, index=te_index, columns=cols)
    out_path = Path(cfg["paths"]["outputs_dir"]) / "predictions" / "test_predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path)
    print(f"[predict] Saved test predictions to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); run(args.config)
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from ..config import load_config
from ..utils.log import get_logger
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows
from ..dataio.dataset import WindowDataset
from ..models.dcenn import TinyDCENN
import torch.nn as nn
from tqdm import tqdm

def run(cfg_path):
    cfg = load_config(cfg_path); log = get_logger("train-encoder")
    # build or load interim data
    train_df, val_df, test_df = build_master(cfg)

    # choose features & targets (no weather grid yet → use calendar + caps + lagged)
    feature_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
                    "holiday","cap_wind_mw","cap_solar_mw",
                    "wind_mw","solar_mw","load_mw","price_eur_mwh"]  # include lags implicitly via context
    target_cols  = ["cf_wind","cf_solar","load_mw","price_eur_mwh"]

    ctx = cfg["features"]["context_hours"]; hz = cfg["features"]["horizon_hours"]
    Xtr, Ytr, _ = make_windows(train_df, feature_cols, target_cols, ctx, hz)
    Xva, Yva, _ = make_windows(val_df,   feature_cols, target_cols, ctx, hz)

    train_ds = WindowDataset(Xtr, Ytr); val_ds = WindowDataset(Xva, Yva)
    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["encoder"]["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["training"]["encoder"]["batch_size"], shuffle=False)

    device = torch.device("cpu")
    model = TinyDCENN(in_channels=Xtr.shape[2], hidden_channels=cfg["training"]["encoder"]["latent_channels"]).to(device)

    # tiny linear heads for joint pretrain
    heads = nn.ModuleDict({
        "cf_wind":  nn.Linear(cfg["training"]["encoder"]["latent_channels"], hz),
        "cf_solar": nn.Linear(cfg["training"]["encoder"]["latent_channels"], hz),
        "load":     nn.Linear(cfg["training"]["encoder"]["latent_channels"], hz),
        "price":    nn.Linear(cfg["training"]["encoder"]["latent_channels"], hz),
    }).to(device)

    optim = torch.optim.Adam(list(model.parameters())+list(heads.parameters()), lr=cfg["training"]["encoder"]["lr"])
    mae = nn.L1Loss()
    price_w = cfg["training"]["encoder"]["price_loss_weight"]

    best_val = 1e9
    for epoch in range(cfg["training"]["encoder"]["epochs"]):
        model.train(); heads.train(); total=0
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch+1}"):
            xb = xb.to(device); yb = yb.to(device)  # yb: [B, H, 4]
            z = model(xb)         # [B, latent]
            pred = torch.stack([
                heads["cf_wind"](z),
                heads["cf_solar"](z),
                heads["load"](z),
                heads["price"](z),
            ], dim=-1)            # [B, H, 4]
            loss = mae(pred[:,:,0], yb[:,:,0]) + mae(pred[:,:,1], yb[:,:,1]) + mae(pred[:,:,2], yb[:,:,2]) + price_w*mae(pred[:,:,3], yb[:,:,3])
            optim.zero_grad(); loss.backward(); optim.step(); total += loss.item()
        # val
        model.eval(); heads.eval()
        with torch.no_grad():
            vloss=0
            for xb, yb in val_dl:
                xb=xb.to(device); yb=yb.to(device)
                z = model(xb)
                pred = torch.stack([
                    heads["cf_wind"](z),
                    heads["cf_solar"](z),
                    heads["load"](z),
                    heads["price"](z),
                ], dim=-1)
                vloss += (mae(pred[:,:,0], yb[:,:,0]) + mae(pred[:,:,1], yb[:,:,1]) + mae(pred[:,:,2], yb[:,:,2]) + price_w*mae(pred[:,:,3], yb[:,:,3])).item()
        log.info(f"epoch {epoch+1} train_loss={total/len(train_dl):.4f} val_loss={vloss/len(val_dl):.4f}")
        if vloss < best_val:
            best_val = vloss
            Path(cfg["paths"]["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save({"encoder": model.state_dict()}, Path(cfg["paths"]["checkpoints_dir"])/"encoder.pt")
    log.info("Saved best encoder.")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    run(args.config)


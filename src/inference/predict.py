import torch, numpy as np, pandas as pd, joblib
from pathlib import Path
from ..config import load_config
from ..dataio.preprocess import build_master
from ..dataio.window import make_windows
from ..models.dcenn import TinyDCENN
from ..train.fit_elm import extract_latents

def run(cfg_path):
    cfg = load_config(cfg_path)
    train_df, val_df, test_df = build_master(cfg)
    feature_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
                    "holiday","cap_wind_mw","cap_solar_mw",
                    "wind_mw","solar_mw","load_mw","price_eur_mwh"]
    target_cols  = ["cf_wind","cf_solar","load_mw","price_eur_mwh"]
    ctx = cfg["features"]["context_hours"]; hz = cfg["features"]["horizon_hours"]
    Xte, Yte, te_index = make_windows(test_df, feature_cols, target_cols, ctx, hz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TinyDCENN(in_channels=Xte.shape[2], hidden_channels=cfg["training"]["encoder"]["latent_channels"]).to(device)
    ckpt = torch.load(Path(cfg["paths"]["checkpoints_dir"])/"encoder.pt", map_location=device)
    enc.load_state_dict(ckpt["encoder"])
    Zte = extract_latents(enc, Xte, device=device)

    elms = joblib.load(Path(cfg["paths"]["checkpoints_dir"])/"elms.joblib")
    cf_wind_te  = elms["elm_wind"].predict(torch.from_numpy(Zte)).numpy()
    cf_solar_te = elms["elm_solar"].predict(torch.from_numpy(Zte)).numpy()
    load_te     = elms["elm_load"].predict(torch.from_numpy(Zte)).numpy()

    # capacities aligned to horizons (yearly constant within year)
    caps_te_w = test_df["cap_wind_mw"].values[ctx: ctx + cf_wind_te.shape[0]]
    caps_te_s = test_df["cap_solar_mw"].values[ctx: ctx + cf_solar_te.shape[0]]
    caps_te_w = np.repeat(caps_te_w[:,None], cf_wind_te.shape[1], axis=1)
    caps_te_s = np.repeat(caps_te_s[:,None], cf_solar_te.shape[1], axis=1)

    wind_mw_te  = cf_wind_te  * caps_te_w
    solar_mw_te = cf_solar_te * caps_te_s
    residual_te = load_te - (wind_mw_te + solar_mw_te)

    Xte_price = np.concatenate([Zte, residual_te], axis=1)
    price_te  = elms["elm_price"].predict(torch.from_numpy(Xte_price)).numpy()

    # save predictions flat
    cols = []
    for h in range(hz):
        cols += [f"wind_mw+h{h+1}", f"solar_mw+h{h+1}", f"load_mw+h{h+1}", f"price+h{h+1}"]
    arr = np.concatenate([wind_mw_te, solar_mw_te, load_te, price_te], axis=1).reshape(Zte.shape[0], -1)
    pred_df = pd.DataFrame(arr, index=te_index, columns=cols)
    out = Path(cfg["paths"]["outputs_dir"])/"predictions"/"test_predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out)
    print(f"Saved {out}")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    run(args.config)


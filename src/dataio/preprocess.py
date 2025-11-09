import pandas as pd, numpy as np
from pathlib import Path
from .loader import load_engineered

def build_master(cfg):
    df = load_engineered(cfg)

    # add cyclical time features if missing
    if not {"hour_sin","hour_cos"}.issubset(df.columns):
        idx = df.index.tz_convert("UTC")
        df["hour_sin"]  = np.sin(2*np.pi*idx.hour/24);  df["hour_cos"]  = np.cos(2*np.pi*idx.hour/24)
        df["dow_sin"]   = np.sin(2*np.pi*idx.dayofweek/7); df["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
        df["month_sin"] = np.sin(2*np.pi*(idx.month-1)/12); df["month_cos"] = np.cos(2*np.pi*(idx.month-1)/12)

    split = cfg["time"]["split"]
    train_end = pd.Timestamp(split["train_until"], tz="UTC")
    val_end   = pd.Timestamp(split["val_until"],   tz="UTC")
    test_end  = pd.Timestamp(split["test_until"],  tz="UTC")

    df = df.loc[:test_end]
    train_df = df.loc[:train_end]
    val_df   = df.loc[train_end + pd.Timedelta(hours=1): val_end]
    test_df  = df.loc[val_end + pd.Timedelta(hours=1):]

    outdir = Path(cfg["paths"]["interim_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(outdir/"train.parquet")
    val_df.to_parquet(outdir/"val.parquet")
    test_df.to_parquet(outdir/"test.parquet")
    return train_df, val_df, test_df


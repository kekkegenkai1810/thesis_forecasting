import pandas as pd
from pathlib import Path
from src.config import load_config
from src.utils.asp_rules import apply_asp_bounds

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg = load_config(args.config)
    preds_p = Path(cfg["paths"]["outputs_dir"])/"predictions"/"test_predictions.parquet"
    df = pd.read_parquet(preds_p)

    # Example: apply ASP across final horizon only (you can extend to all horizons)
    wind_cols  = [c for c in df.columns if c.startswith("wind_mw+h")]
    solar_cols = [c for c in df.columns if c.startswith("solar_mw+h")]
    load_cols  = [c for c in df.columns if c.startswith("load_mw+h")]
    price_cols = [c for c in df.columns if c.startswith("price+h")]

    # add capacity columns by merging with processed test caps at aligned index
    # (for brevity assume you can merge from interim/test.parquet)
    test_df = pd.read_parquet(Path(cfg["paths"]["interim_dir"])/"test.parquet")
    caps = test_df[["cap_wind_mw","cap_solar_mw"]].iloc[len(test_df)-len(df):]  # crude align; refine as needed
    df = df.join(caps)

    cleaned, flags = apply_asp_bounds(
        df.copy(),
        wind_cols=wind_cols, solar_cols=solar_cols, load_cols=load_cols, price_cols=price_cols,
        pv_night_hours=tuple(cfg["asp"]["pv_night_hours"]),
        ramp_limits=None
    )
    out_p = Path(cfg["paths"]["outputs_dir"])/"predictions"/"test_predictions_asp.parquet"
    cleaned.to_parquet(out_p)
    print(f"ASP applied; saved {out_p}. Adjusted rows: {flags['adjusted'].sum()}")


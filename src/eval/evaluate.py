import pandas as pd, numpy as np
from pathlib import Path
from ..config import load_config
from ..utils.metrics import mae, smape

def run(cfg_path):
    cfg = load_config(cfg_path)
    pred_p = Path(cfg["paths"]["outputs_dir"])/"predictions"/"test_predictions.parquet"
    pred = pd.read_parquet(pred_p)

    # You can also load Yte to compute true MAE; for brevity, evaluate only on final horizon if you supply ground truth
    # In practice, load processed test_df and reconstruct Yte to compare.

    print("Evaluation stub: add loading of ground truth Yte and compute MAE/SMAPE per horizon.")

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    run(args.config)


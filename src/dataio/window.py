import numpy as np
import pandas as pd

def make_windows(df, feature_cols, target_cols, context=168, horizon=12):
    X, Y = [], []
    idx = df.index
    values = df[feature_cols + target_cols].values
    n = len(df)
    for t in range(context, n - horizon):
        feat = values[t-context:t, :len(feature_cols)]
        targ = values[t+1:t+1+horizon, len(feature_cols):]
        X.append(feat)
        Y.append(targ)
    X = np.array(X, dtype=np.float32)           # [N, T, F]
    Y = np.array(Y, dtype=np.float32)           # [N, H, C_targets]
    # reshape for dCeNN expecting [B,T,C,H,W] with H=W=1
    X_dcenn = np.transpose(X, (0,1,2))[:, :, :, None, None]  # [N,T,F,1,1]
    return X_dcenn, Y, idx[context: n - horizon]


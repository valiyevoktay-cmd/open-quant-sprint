"""
Track 10: the absorption ratio function code
"""

import numpy as np
import pandas as pd


def absorption_ratio(df, window= 168, k = 1):

# hours x symbols matrix of hourly log returns on a full hourly grid
    close_df = df.pivot(index="timestamp", columns="symbol", values="close")
    close_df = close_df.reindex(pd.date_range(close_df.index[0], close_df.index[-1], freq="h"))
    log_returns = np.log(close_df).diff().to_numpy()

# NaN rule
    arr = np.full(len(log_returns), np.nan)

    for t in range(window, len(log_returns)):
        window_rets = log_returns[t - window + 1 : t + 1]
        if np.isnan(window_rets).any():
            continue
        # Covariance of the last window returns up to and including t, and its eigenvalues
        eig = np.linalg.eigvalsh(np.cov(log_returns[t - window + 1 : t + 1], rowvar=False))
        
        # Absorption ratio: k largest eigenvalues over the sum of all
        total = eig.sum()
        arr[t] = min(eig[-k:].sum() / total, 1.0)

# 5. Same value on every row of that hour
    by_hour = pd.Series(arr, index=close_df.index)
    return df["timestamp"].map(by_hour)

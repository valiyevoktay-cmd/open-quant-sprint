from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """Weekly residual momentum with a lagged funding rank adjustment."""
    if df.empty:
        return pd.Series(index=df.index, dtype=float)

    days = pd.to_datetime(df["timestamp"]).dt.floor("D")
    daily = df.assign(_day=days).groupby(["_day", "symbol"])

    close = daily["close"].last().unstack("symbol").sort_index().asfreq("D")
    close = close.where(close > 0)

    # Average each day's observed funding, then use only completed days.
    funding = (
        daily["funding_rate"]
        .mean()
        .unstack("symbol")
        .reindex(index=close.index, columns=close.columns)
    )
    lagged_funding = funding.rolling(7, min_periods=3).mean().shift(1)

    daily_return = np.log(close).diff()
    market_return = daily_return.mean(axis=1)

    # Monday's signal uses returns through Sunday. Betas use trailing history
    # only and are estimated before the rebalance day.
    market_7d = market_return.rolling(7, min_periods=7).sum().shift(1)
    asset_7d = daily_return.rolling(7, min_periods=7).sum().shift(1)
    market_variance = market_return.rolling(28, min_periods=20).var()
    beta = (
        daily_return.rolling(28, min_periods=20)
        .cov(market_return)
        .div(market_variance.where(market_variance > 1e-12), axis=0)
        .shift(1)
        .clip(-3.0, 3.0)
    )
    residual = asset_7d - beta.mul(market_7d, axis=0)

    # Residual momentum receives more risk when returns are dispersed and
    # losses are localized. Broad, synchronized selloffs receive less risk.
    dispersion = asset_7d.std(axis=1)
    trailing_dispersion = dispersion.rolling(84, min_periods=28).median().shift(1)
    observed = asset_7d.notna().sum(axis=1)
    negative_breadth = asset_7d.lt(0).sum(axis=1).div(observed.where(observed > 0))

    valid = residual.notna() & beta.notna()
    valid_count = valid.sum(axis=1)
    score = residual.where(valid).rank(axis=1, method="average")
    score = score.sub(score.mean(axis=1), axis=0)

    # Missing funding is neutral within each day's cross-section.
    funding_rank = lagged_funding.where(valid).rank(axis=1, method="average")
    funding_centered = funding_rank.sub(funding_rank.mean(axis=1), axis=0).fillna(0.0)
    score = score - 0.25 * funding_centered

    # Remove the component correlated with trailing market beta.
    centered_beta = beta.where(valid).sub(beta.where(valid).mean(axis=1), axis=0)
    beta_size = centered_beta.pow(2).sum(axis=1)
    beta_component = (score * centered_beta).sum(axis=1).div(
        beta_size.where(beta_size > 1e-12)
    ).fillna(0.0)
    score = score - centered_beta.mul(beta_component, axis=0)
    score = score.sub(score.mean(axis=1), axis=0)

    mixed = negative_breadth.between(0.25, 0.75)
    high_dispersion = dispersion.gt(trailing_dispersion)
    synchronized_selloff = negative_breadth.ge(0.80) & market_7d.lt(0)
    exposure = pd.Series(np.where(high_dispersion & mixed, 1.0, 0.40),
                         index=close.index)
    exposure = exposure.where(~synchronized_selloff, 0.15)
    gross = score.abs().sum(axis=1)
    target = score.div(gross.where(gross > 1e-12), axis=0).mul(exposure, axis=0)
    target = target.where(valid_count.ge(3), 0.0).fillna(0.0)

    # Rebalance on Monday and carry the target until the next Monday.
    monday = close.index.dayofweek == 0
    daily_weights = target.loc[monday].reindex(close.index).ffill().fillna(0.0)
    daily_betas = beta.loc[monday].fillna(1.0).reindex(close.index).ffill().fillna(1.0)

    lookup = pd.MultiIndex.from_arrays([days, df["symbol"]])
    weights = pd.Series(
        daily_weights.stack().reindex(lookup).fillna(0.0).to_numpy(),
        index=df.index,
    )
    betas = pd.Series(
        daily_betas.stack().reindex(lookup).fillna(1.0).to_numpy(),
        index=df.index,
    )

    # Reproject for the assets present at each timestamp to remove dollar
    # and estimated market exposure, then cap gross exposure at one.
    groups = df["timestamp"]
    weights -= weights.groupby(groups).transform("mean")
    centered_beta = betas - betas.groupby(groups).transform("mean")
    beta_size = centered_beta.pow(2).groupby(groups).transform("sum")
    beta_component = (weights * centered_beta).groupby(groups).transform("sum")
    weights -= centered_beta * (
        beta_component / beta_size.where(beta_size > 1e-12)
    ).fillna(0.0)
    weights -= weights.groupby(groups).transform("mean")

    gross = weights.abs().groupby(groups).transform("sum").clip(lower=1.0)
    return weights.div(gross).astype(float)

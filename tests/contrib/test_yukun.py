from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.contrib.yukun import daily_sigma_volume, impact_bps, run_impact_cost, turnover, weight_change

N_HOURS, N_SYM = 2000, 5
AUM = 1e6


def make_panel(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """5 symbols x 2000 hours, sorted by (timestamp, symbol), RangeIndex: row = hour * N_SYM + k.

    Random-walk closes with a different volatility per symbol, log-normal quote_volume, and new
    random weights every 6 hours, large enough that the gross cap of 1 binds on most hours.
    """
    rng = np.random.default_rng(seed)
    vol = np.array([0.002, 0.004, 0.006, 0.008, 0.010])
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, vol, (N_HOURS, N_SYM)), axis=0))
    qv = rng.lognormal(16.0, 0.7, (N_HOURS, N_SYM))
    w = np.repeat(rng.normal(0.0, 0.5, (N_HOURS // 6 + 1, N_SYM)), 6, axis=0)[:N_HOURS]
    ts = pd.date_range("2025-01-01", periods=N_HOURS, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, N_SYM),
            "symbol": np.tile([f"S{k}" for k in range(N_SYM)], N_HOURS),
            "close": close.ravel(),
            "quote_volume": qv.ravel(),
            "funding_rate": 0.0,
        }
    )
    return df, w.ravel()


def bar_pos(df: pd.DataFrame) -> np.ndarray:
    """Position of each row within its symbol: 0 for the first bar."""
    return df.groupby("symbol").cumcount().to_numpy()


@pytest.mark.parametrize("lag_bars", [1, 2])
def test_turnover_matches_run_backtest(lag_bars: int) -> None:
    df, w = make_panel()
    gross = pd.Series(np.abs(w)).groupby(df["timestamp"]).sum()
    assert (gross > 1.0).mean() > 0.9  # the gross cap binds on most hours

    to = turnover(df, w, lag_bars)
    rows = np.random.default_rng(1).choice(np.flatnonzero(to > 0), size=5, replace=False)
    for i in rows:
        fee = np.zeros(len(df))
        fee[i] = 1e4  # fee rate 1 on row i: fee_cost_cum is the turnover of row i
        m = run_backtest(df, w, fee_bps=fee, lag_bars=lag_bars)
        assert abs(m.fee_cost_cum - to[i]) < 1e-4  # the metric is rounded to 4 decimals


def test_zero_on_rows_without_trade() -> None:
    df, w = make_panel()
    to = turnover(df, w)
    imp = impact_bps(df, w, AUM)
    assert (to == 0).sum() > 0
    assert (imp[to == 0] == 0.0).all()


def test_square_root_scaling_in_aum_and_volume() -> None:
    df, w = make_panel()
    imp = impact_bps(df, w, AUM)
    ok = np.isfinite(imp) & (imp > 0)
    assert ok.sum() > 1000

    np.testing.assert_allclose(impact_bps(df, w, 4 * AUM)[ok], 2.0 * imp[ok], rtol=1e-12)
    more_volume = df.assign(quote_volume=4 * df["quote_volume"])
    np.testing.assert_allclose(impact_bps(more_volume, w, AUM)[ok], 0.5 * imp[ok], rtol=1e-12)


def test_hand_computed_row() -> None:
    df, w = make_panel()
    close = df["close"].to_numpy().reshape(N_HOURS, N_SYM)
    qv = df["quote_volume"].to_numpy().reshape(N_HOURS, N_SYM)
    w_grid = w.reshape(N_HOURS, N_SYM)

    t, k = 1003, 2  # weights change at hour 1002, so the trade is on row t (lag 1); t > 30 days
    row = t * N_SYM + k

    scaled = w_grid / np.maximum(np.abs(w_grid).sum(axis=1, keepdims=True), 1.0)
    trade = abs(scaled[t - 1, k] - scaled[t - 2, k])
    returns = np.diff(np.log(close[t - 721 : t, k]))  # closes of bars t-721 .. t-1
    volumes = qv[t - 720 : t, k]  # bars t-720 .. t-1
    assert len(returns) == 720 and len(volumes) == 720 and trade > 0

    sigma_d = np.std(returns, ddof=1) * np.sqrt(24)
    v_d = np.mean(volumes) * 24
    expected = (2 / 3) * 1.0 * sigma_d * np.sqrt(AUM * trade / v_d) * 1e4

    assert turnover(df, w)[row] == pytest.approx(trade, rel=1e-12)
    assert impact_bps(df, w, AUM)[row] == pytest.approx(expected, rel=1e-10)


def test_values_do_not_change_when_later_rows_are_dropped() -> None:
    df, w = make_panel()
    def outputs(d: pd.DataFrame, wt: np.ndarray) -> tuple[np.ndarray, ...]:
        return (turnover(d, wt), *daily_sigma_volume(d), impact_bps(d, wt, AUM))

    full = outputs(df, w)
    for hours in (30, 500, 1201):
        n = hours * N_SYM
        for a, b in zip(full, outputs(df.iloc[:n], w[:n])):
            np.testing.assert_array_equal(a[:n], b)


def test_current_bar_does_not_affect_its_own_value() -> None:
    df, w = make_panel()
    t = 1003
    changed = df.copy()
    rows = changed["timestamp"] == changed["timestamp"].iloc[t * N_SYM]
    changed.loc[rows, "close"] *= 1.5
    changed.loc[rows, "quote_volume"] *= 10.0

    before, after = impact_bps(df, w, AUM), impact_bps(changed, w, AUM)
    np.testing.assert_array_equal(after[: (t + 1) * N_SYM], before[: (t + 1) * N_SYM])
    assert not np.allclose(after[(t + 1) * N_SYM :], before[(t + 1) * N_SYM :])  # it does enter later


def test_no_contamination_across_symbols() -> None:
    df, w = make_panel()
    changed = df.copy()
    s0 = (changed["symbol"] == "S0").to_numpy()
    changed.loc[s0, "close"] = changed.loc[s0, "close"] * np.exp(np.random.default_rng(2).normal(0, 0.05, s0.sum()))
    changed.loc[s0, "quote_volume"] *= 7.0

    before, after = impact_bps(df, w, AUM), impact_bps(changed, w, AUM)
    np.testing.assert_array_equal(after[~s0], before[~s0])
    assert not np.allclose(after[s0], before[s0], equal_nan=True)


def test_nan_in_warm_up_only() -> None:
    df, w = make_panel()
    w = w.copy()
    w[23 * N_SYM : 24 * N_SYM] *= -1.0  # extra trades on bars 24 and 25, around the boundary
    pos, to = bar_pos(df), turnover(df, w)
    imp = impact_bps(df, w, AUM)
    sigma, volume = daily_sigma_volume(df)

    # sigma_d needs 24 returns before t (bars 0..24 NaN), V_d needs 24 volumes (bars 0..23 NaN)
    assert np.isnan(sigma[pos <= 24]).all() and np.isfinite(sigma[pos >= 25]).all()
    assert np.isnan(volume[pos <= 23]).all() and np.isfinite(volume[pos >= 24]).all()

    warm = pos <= 24
    assert (to[pos == 24] > 0).all() and (to[pos == 25] > 0).all()
    assert np.isnan(imp[warm & (to > 0)]).all()
    assert (imp[warm & (to == 0)] == 0.0).all()
    assert np.isfinite(imp[~warm]).all()


def test_aligned_with_non_default_index() -> None:
    df, w = make_panel()
    shifted = df.set_axis(np.arange(len(df)) * 3 + 10)
    np.testing.assert_array_equal(
        impact_bps(shifted, pd.Series(w, index=shifted.index), AUM),
        impact_bps(df, w, AUM),
    )


def test_zero_volume_gives_nan_not_inf() -> None:
    df, w = make_panel()
    s0 = (df["symbol"] == "S0").to_numpy()
    df.loc[s0, "quote_volume"] = 0.0
    imp = impact_bps(df, w, AUM)
    to = turnover(df, w)
    assert not np.isinf(imp).any()
    assert np.isnan(imp[s0 & (to > 0)]).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"aum": 0.0}, {"aum": -1.0}, {"aum": np.nan}, {"y": -1.0}, {"days": 0}, {"days": 1.5}, {"lag_bars": 0}],
)
def test_invalid_arguments(kwargs: dict) -> None:
    df, w = make_panel()
    args = {"aum": AUM, **kwargs}
    with pytest.raises(ValueError):
        impact_bps(df, w, **args)


def run_panel(noise: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    """The test panel with weights only on S2: buys at hours 1000-1002 and 1004, a sell at 1005.

    With lag 1 the trades are on bars 1001, 1002, 1003, 1005 (buys) and 1006 (sell). `noise` is
    added to the weight of hour 1003, which makes a tiny buy on bar 1004.
    """
    df, _ = make_panel()
    step = np.zeros(N_HOURS)
    step[[1000, 1001, 1002, 1004]] = 0.1
    step[1005] = -0.2
    w = np.zeros((N_HOURS, N_SYM))
    w[:, 2] = np.cumsum(step)
    w[1003, 2] += noise
    return df, w.ravel()


def expected_run_cost(df: pd.DataFrame, bar: int, size: float) -> float:
    sigma, volume = daily_sigma_volume(df)
    r = bar * N_SYM + 2
    return size * (2 / 3) * sigma[r] * np.sqrt(AUM * size / volume[r])


def test_weight_change_sign() -> None:
    df, w = run_panel()
    change = weight_change(df, w).reshape(N_HOURS, N_SYM)[:, 2]
    assert np.flatnonzero(change > 1e-12).tolist() == [1001, 1002, 1003, 1005]
    assert np.flatnonzero(change < -1e-12).tolist() == [1006]
    np.testing.assert_array_equal(np.abs(weight_change(*make_panel())), turnover(*make_panel()))


@pytest.mark.parametrize(
    ("gap", "runs"),
    [(0, [(1001, 0.3), (1005, 0.1), (1006, 0.2)]), (1, [(1001, 0.4), (1006, 0.2)])],
)
def test_hand_computed_runs(gap: int, runs: list[tuple[int, float]]) -> None:
    df, w = run_panel()
    cost = run_impact_cost(df, w, AUM, gap=gap)
    assert np.flatnonzero(cost).tolist() == [bar * N_SYM + 2 for bar, _ in runs]
    for bar, size in runs:
        assert cost[bar * N_SYM + 2] == pytest.approx(expected_run_cost(df, bar, size), rel=1e-10)


def test_noise_below_tol_does_not_join_runs() -> None:
    df, w = run_panel(noise=1e-15)
    np.testing.assert_allclose(run_impact_cost(df, w, AUM), run_impact_cost(*run_panel(), AUM), rtol=1e-9)
    joined = run_impact_cost(df, w, AUM, tol=0.0)  # the noise trade on bar 1004 bridges 1003 and 1005
    assert np.count_nonzero(joined) == 2
    assert joined[1001 * N_SYM + 2] == pytest.approx(expected_run_cost(df, 1001, 0.4), rel=1e-9)


def test_runs_of_one_bar_cost_the_same_as_hourly() -> None:
    df, _ = make_panel()
    sign = np.where(np.arange(N_HOURS) % 2 == 0, 1.0, -1.0)
    w = (sign[:, None] * np.array([0.05, 0.1, 0.15, 0.2, 0.25])).ravel()  # direction flips every bar
    hourly = turnover(df, w) * impact_bps(df, w, AUM) * 1e-4
    np.testing.assert_allclose(run_impact_cost(df, w, AUM), hourly, rtol=1e-12)


def test_merged_cost_scales_with_sqrt_aum() -> None:
    df, w = make_panel()
    cost = run_impact_cost(df, w, AUM, gap=3)
    ok = np.isfinite(cost) & (cost > 0)
    assert ok.sum() > 100
    np.testing.assert_allclose(run_impact_cost(df, w, 4 * AUM, gap=3)[ok], 2.0 * cost[ok], rtol=1e-12)


@pytest.mark.parametrize("kwargs", [{"aum": 0.0}, {"y": -1.0}, {"gap": -1}, {"gap": 1.5}])
def test_run_impact_cost_invalid_arguments(kwargs: dict) -> None:
    df, w = make_panel()
    with pytest.raises(ValueError):
        run_impact_cost(df, w, **{"aum": AUM, **kwargs})

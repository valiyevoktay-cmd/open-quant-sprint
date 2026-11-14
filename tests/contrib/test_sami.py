from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contrib.sami import abnormal_volume


@pytest.fixture(scope="module")
def synthetic_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_symbols = 20
    n_hours = 3000
    hours = np.arange(n_hours)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    log_volume = (
        10.0
        + 0.1 * np.arange(n_symbols)[None, :]
        + 0.6 * np.sin(2.0 * np.pi * hours[:, None] / 24.0)
        + rng.normal(0.0, 0.3, size=(n_hours, n_symbols))
    )
    timestamps = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, n_symbols),
            "symbol": np.tile(symbols, n_hours),
            "quote_volume": np.exp(log_volume.ravel()),
        }
    )


def test_warmup_valid_values_and_index_alignment(synthetic_panel: pd.DataFrame) -> None:
    result = abnormal_volume(synthetic_panel)
    values = result.to_numpy().reshape(3000, 20)

    assert result.index.equals(synthetic_panel.index)
    assert result.name == "abnormal_volume"
    assert np.isnan(values[: 24 * 14]).all()
    assert np.isfinite(values[24 * 14 :]).all()


def test_daily_pattern_is_removed_and_theoretical_scale_matches(
    synthetic_panel: pd.DataFrame,
) -> None:
    result = abnormal_volume(synthetic_panel).dropna().to_numpy()
    expected_std = np.sqrt((1.0 + 1.0 / 14.0) * (14.0 - 1.0) / (14.0 - 3.0))

    assert abs(float(result.mean())) < 0.02
    assert abs(float(result.std()) - expected_std) < 0.03


def test_volume_shock_is_detected(synthetic_panel: pd.DataFrame) -> None:
    df = synthetic_panel.copy()
    row = 1000 * 20 + 3
    df.loc[row, "quote_volume"] *= np.exp(3.0)

    result = abnormal_volume(df)

    assert result.loc[row] > 5.0


def test_later_rows_do_not_change_earlier_values(synthetic_panel: pd.DataFrame) -> None:
    full = abnormal_volume(synthetic_panel)
    cutoff = synthetic_panel["timestamp"].drop_duplicates().iloc[2000]
    prefix = synthetic_panel.loc[synthetic_panel["timestamp"] <= cutoff]
    truncated = abnormal_volume(prefix)

    np.testing.assert_allclose(
        truncated.to_numpy(),
        full.loc[prefix.index].to_numpy(),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_missing_hour_invalidates_next_same_hour_history_only(
    synthetic_panel: pd.DataFrame,
) -> None:
    df = synthetic_panel.copy()
    baseline = abnormal_volume(df)
    target_row = 1000 * 20 + 3
    target_time = df.loc[target_row, "timestamp"]
    target_symbol = df.loc[target_row, "symbol"]
    missing = df.drop(index=target_row)
    actual = abnormal_volume(missing)

    affected_times = [target_time + pd.Timedelta(hours=24 * lag) for lag in range(1, 15)]
    affected = missing["symbol"].eq(target_symbol) & missing["timestamp"].isin(affected_times)
    assert affected.sum() == 14
    assert actual.loc[missing.index[affected]].isna().all()
    np.testing.assert_allclose(
        actual.loc[missing.index[~affected]].to_numpy(),
        baseline.loc[missing.index[~affected]].to_numpy(),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_zero_volume_invalidates_current_and_next_same_hour_history_only(
    synthetic_panel: pd.DataFrame,
) -> None:
    df = synthetic_panel.copy()
    baseline = abnormal_volume(df)
    target_row = 1000 * 20 + 3
    target_time = df.loc[target_row, "timestamp"]
    target_symbol = df.loc[target_row, "symbol"]
    df.loc[target_row, "quote_volume"] = 0.0
    actual = abnormal_volume(df)

    affected_times = [target_time + pd.Timedelta(hours=24 * lag) for lag in range(15)]
    affected = df["symbol"].eq(target_symbol) & df["timestamp"].isin(affected_times)
    assert affected.sum() == 15
    assert actual.loc[df.index[affected]].isna().all()
    np.testing.assert_allclose(
        actual.loc[df.index[~affected]].to_numpy(),
        baseline.loc[df.index[~affected]].to_numpy(),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )


def test_constant_history_returns_nan() -> None:
    n_hours = 24 * 16
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n_hours, freq="h", tz="UTC"),
            "symbol": ["S00"] * n_hours,
            "quote_volume": [100.0] * n_hours,
        }
    )

    assert abnormal_volume(df).isna().all()


@pytest.mark.parametrize("days", [0, 1, -1, 1.5, True])
def test_days_must_be_an_integer_of_at_least_two(days: int | float) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "symbol": ["S00", "S00"],
            "quote_volume": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="days must be an integer greater than or equal to 2"):
        abnormal_volume(df, days=days)


def test_input_order_and_non_default_index_are_preserved(
    synthetic_panel: pd.DataFrame,
) -> None:
    df = synthetic_panel.sample(frac=1.0, random_state=17).copy()
    df.index = pd.Index(np.arange(len(df)) + 10_000)
    actual = abnormal_volume(df)

    assert actual.index.equals(df.index)

    baseline_df = synthetic_panel
    baseline = abnormal_volume(baseline_df)
    baseline_by_key = pd.Series(
        baseline.to_numpy(),
        index=pd.MultiIndex.from_frame(baseline_df[["timestamp", "symbol"]]),
    )
    shuffled_keys = pd.MultiIndex.from_frame(df[["timestamp", "symbol"]])
    np.testing.assert_allclose(
        actual.to_numpy(),
        baseline_by_key.reindex(shuffled_keys).to_numpy(),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )

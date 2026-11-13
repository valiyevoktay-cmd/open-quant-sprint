from itertools import combinations

import numpy as np
import pandas as pd

from .models import engle_granger, kalman_hedge_ratio


EG_CRITICAL_5 = -3.34  # Approximate 5% critical value, 2 variables + intercept.


def make_pairs(columns):
    return list(combinations(columns, 2))


def test_all_pairs(window: pd.DataFrame, pairs=None, critical_value=EG_CRITICAL_5):
    """Test all requested pairs; skip pairs with missing values in the window."""
    if pairs is None:
        pairs = make_pairs(window.columns)
    values = window.to_numpy(dtype=float)
    col_index = {name: i for i, name in enumerate(window.columns)}
    rows = []

    for symbol_y, symbol_x in pairs:
        y = values[:, col_index[symbol_y]]
        x = values[:, col_index[symbol_x]]
        if not (np.isfinite(y).all() and np.isfinite(x).all()):
            continue
        try:
            statistic, beta = engle_granger(y, x)
        except ValueError:
            continue
        rows.append({
            "symbol_y": symbol_y,
            "symbol_x": symbol_x,
            "t_stat": statistic,
            "beta": beta,
            "cointegrated": statistic < critical_value,
        })
    return pd.DataFrame(rows, columns=[
        "symbol_y", "symbol_x", "t_stat", "beta", "cointegrated"
    ])


def rolling_cointegration(
    log_prices, pairs=None, formation_days=30, testing_days=30,
    step_days=7, critical_value=EG_CRITICAL_5,
):
    """Evaluate formation/test windows; windows advance by step_days."""
    if pairs is None:
        pairs = make_pairs(log_prices.columns)
    formation_n = formation_days * 24
    testing_n = testing_days * 24
    step_n = step_days * 24
    rows = []

    for start in range(0, len(log_prices) - formation_n - testing_n + 1, step_n):
        split = start + formation_n
        end = split + testing_n
        formation = test_all_pairs(
            log_prices.iloc[start:split], pairs, critical_value
        )
        testing = test_all_pairs(
            log_prices.iloc[split:end], pairs, critical_value
        )
        if formation.empty or testing.empty:
            continue

        merged = formation.merge(
            testing, on=["symbol_y", "symbol_x"],
            suffixes=("_formation", "_testing")
        )
        form_pass = merged["cointegrated_formation"]
        test_pass = merged["cointegrated_testing"]
        n_form = int(form_pass.sum())
        n_test = int(test_pass.sum())
        n_both = int((form_pass & test_pass).sum())
        rows.append({
            "formation_start": log_prices.index[start],
            "testing_start": log_prices.index[split],
            "formation_passes": n_form,
            "testing_passes": n_test,
            "persistent": n_both,
            "persistence_rate": n_both / n_form if n_form else np.nan,
        })
    return pd.DataFrame(rows)


def hedge_ratio_stability(formation_results, testing_results):
    """Compare testing betas for every pair selected in formation."""
    selected = formation_results.loc[
        formation_results["cointegrated"],
        ["symbol_y", "symbol_x", "beta", "t_stat"]
    ].merge(
        testing_results[["symbol_y", "symbol_x", "beta", "t_stat", "cointegrated"]],
        on=["symbol_y", "symbol_x"], suffixes=("_formation", "_testing")
    )
    selected = selected.rename(columns={
        "beta_formation": "formation_beta",
        "beta_testing": "testing_beta",
        "t_stat_formation": "formation_t_stat",
        "t_stat_testing": "testing_t_stat",
    })
    selected["absolute_change"] = (
        selected["testing_beta"] - selected["formation_beta"]
    ).abs()
    denominator = selected["formation_beta"].abs()
    selected["relative_change"] = selected["absolute_change"].div(
        denominator.where(denominator > 1e-10)
    )
    return selected


def kalman_pair_analysis(
    log_prices, formation_results, formation_hours=720,
    testing_hours=720, warmup_hours=168, delta=1e-5, obs_var=1e-3,
):
    """Run one continuous Kalman filter over both periods for selected pairs."""
    selected = formation_results.loc[formation_results["cointegrated"]]
    end = formation_hours + testing_hours
    rows = []

    for row in selected.itertuples(index=False):
        y = log_prices[row.symbol_y].iloc[:end].to_numpy()
        x = log_prices[row.symbol_x].iloc[:end].to_numpy()
        beta = kalman_hedge_ratio(y, x, delta=delta, obs_var=obs_var)
        f = beta[warmup_hours:formation_hours]
        t = beta[formation_hours:end]
        b_f, b_t = beta[formation_hours - 1], beta[end - 1]
        rows.append({
            "symbol_y": row.symbol_y,
            "symbol_x": row.symbol_x,
            "formation_beta": b_f,
            "testing_beta": b_t,
            "absolute_change": abs(b_t - b_f),
            "relative_change": (
                abs(b_t - b_f) / abs(b_f) if abs(b_f) > 1e-10 else np.nan
            ),
            "mean_formation_beta": float(np.mean(f)),
            "mean_testing_beta": float(np.mean(t)),
            "std_formation_beta": float(np.std(f)),
            "std_testing_beta": float(np.std(t)),
            "formation_variation": float(np.mean(np.abs(np.diff(f)))),
            "testing_variation": float(np.mean(np.abs(np.diff(t)))),
        })
    return pd.DataFrame(rows)


def rolling_pair_results(
    log_prices,
    pairs=None,
    formation_days=30,
    testing_days=30,
    step_days=7,
    critical_value=EG_CRITICAL_5,
):
    """Return detailed pair-level results for every rolling window."""

    if pairs is None:
        pairs = make_pairs(log_prices.columns)

    formation_n = formation_days * 24
    testing_n = testing_days * 24
    step_n = step_days * 24

    results = []

    for start in range(
        0,
        len(log_prices) - formation_n - testing_n + 1,
        step_n,
    ):
        split = start + formation_n
        end = split + testing_n

        formation = test_all_pairs(
            log_prices.iloc[start:split],
            pairs,
            critical_value,
        )

        testing = test_all_pairs(
            log_prices.iloc[split:end],
            pairs,
            critical_value,
        )

        if formation.empty or testing.empty:
            continue

        merged = formation.merge(
            testing,
            on=["symbol_y", "symbol_x"],
            suffixes=("_formation", "_testing"),
        )

        merged["formation_start"] = log_prices.index[start]
        merged["formation_end"] = log_prices.index[split - 1]
        merged["testing_start"] = log_prices.index[split]
        merged["testing_end"] = log_prices.index[end - 1]

        merged["persistent"] = (
            merged["cointegrated_formation"]
            & merged["cointegrated_testing"]
        )

        results.append(merged)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


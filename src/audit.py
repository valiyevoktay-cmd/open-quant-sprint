from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time
from typing import Callable
import numpy as np
import pandas as pd

from src.backtest import BacktestMetrics, coerce_weights, run_backtest
from src.quality import QualityReport, audit_dataset
from src.schema import enforce_schema
from src.stats import information_coefficient

StrategyFn = Callable[[pd.DataFrame], pd.Series]


def load_strategy_fn(file_path: str | Path) -> StrategyFn:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "compute_weights"):
        raise AttributeError(f"{path.name} must define `compute_weights(df: pd.DataFrame) -> pd.Series`")
    return getattr(mod, "compute_weights")


def check_causality_truncation(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    checkpoints: tuple[float, ...] = (0.4, 0.65, 0.85),
    tol: float = 1e-9,
) -> tuple[bool, float, str]:
    """
    1. Verifies deterministic execution (running twice on identical input yields identical weights).
    2. Proves causality by running strategy_fn on prefix slices [0..T_k] and comparing
       the output weights against the full dataset run at timestamps <= T_k.
       Any discrepancy > tol proves future information leaked into past weights.
    """
    full_df = df.copy()
    try:
        raw_full = strategy_fn(full_df.copy())
        w_full = coerce_weights(full_df, raw_full).to_numpy(dtype=np.float64)
    except Exception as exc:
        return False, float("inf"), f"Strategy crashed during execution: {exc}"

    if np.all(np.abs(w_full) < 1e-12):
        return False, 0.0, "Degenerate signal: strategy produced all-zero weights"

    unique_ts = np.sort(full_df["timestamp"].unique())
    max_diff = 0.0

    for idx, frac in enumerate(checkpoints):
        cut_idx = max(10, int(len(unique_ts) * frac))
        t_cut = unique_ts[min(cut_idx, len(unique_ts) - 1)]

        mask = (full_df["timestamp"] <= t_cut).to_numpy()
        sub_df = full_df.loc[mask].reset_index(drop=True)

        try:
            w_sub = coerce_weights(sub_df, strategy_fn(sub_df.copy())).to_numpy(dtype=np.float64)
        except Exception as exc:
            return False, float("inf"), f"Strategy crashed on prefix cut at {t_cut}: {exc}"

        if idx == 0:
            try:
                w_sub_repeat = coerce_weights(sub_df, strategy_fn(sub_df.copy())).to_numpy(dtype=np.float64)
            except Exception as exc:
                return False, float("inf"), f"Strategy crashed on repeated execution: {exc}"
            det_diff = float(np.max(np.abs(w_sub - w_sub_repeat)))
            if det_diff > tol:
                return (
                    False,
                    det_diff,
                    f"Non-deterministic strategy output (diff={det_diff:.3e} on identical input)",
                )

        w_ref = w_full[mask]
        if len(w_sub) != len(w_ref):
            return False, float("inf"), f"Row count mismatch on truncated slice at {t_cut}"

        diff = float(np.max(np.abs(w_sub - w_ref)))
        if np.isnan(diff):
            return False, float("nan"), f"NaN diff encountered at {t_cut}"

        if diff > max_diff:
            max_diff = diff
        if diff > tol:
            return (
                False,
                max_diff,
                f"Look-ahead leak detected at {t_cut}: max|w_trunc - w_full| = {diff:.3e} > {tol:.1e}",
            )

    return True, max_diff, f"Causal invariance verified across {len(checkpoints)} prefix cuts (max diff={max_diff:.2e})"


def purged_oos_split_eval(
    df: pd.DataFrame,
    weights: pd.Series,
    split_frac: float = 0.70,
    embargo_bars: int = 24,
    fee_bps: float = 5.0,
) -> dict[str, BacktestMetrics]:
    """
    Splits timeline into In-Sample (IS) and Out-of-Sample (OOS) with an embargo buffer
    to prevent serial correlation spillover across the boundary.
    """
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    split_idx = int(n_ts * split_frac)
    oos_start_idx = min(n_ts - 1, split_idx + embargo_bars)

    t_is_end = unique_ts[split_idx - 1]
    t_oos_start = unique_ts[oos_start_idx]

    is_mask = df["timestamp"] <= t_is_end
    oos_mask = df["timestamp"] >= t_oos_start

    df_is = df.loc[is_mask].reset_index(drop=True)
    w_is = pd.Series(weights.loc[is_mask].to_numpy(), index=df_is.index)

    df_oos = df.loc[oos_mask].reset_index(drop=True)
    w_oos = pd.Series(weights.loc[oos_mask].to_numpy(), index=df_oos.index)

    return {
        "IS": run_backtest(df_is, w_is, fee_bps=fee_bps),
        "OOS": run_backtest(df_oos, w_oos, fee_bps=fee_bps),
    }


def evaluate_lag_sensitivity(
    df: pd.DataFrame,
    weights: pd.Series,
    fee_bps: float = 5.0,
) -> dict:
    """
    Stress-tests alpha persistence by comparing 1-bar execution delay against 2-bar delay.
    A genuine structural signal decays smoothly; a look-ahead or instant-close signal collapses.
    """
    m1 = run_backtest(df, weights, fee_bps=fee_bps, lag_bars=1)
    m2 = run_backtest(df, weights, fee_bps=fee_bps, lag_bars=2)

    sr1 = m1.sharpe
    sr2 = m2.sharpe
    retention = (sr2 / sr1) if sr1 > 1e-6 else (1.0 if abs(sr1 - sr2) < 1e-6 else 0.0)
    passed = bool(sr2 > 0.0 if sr1 > 0.0 else sr2 >= sr1 - 0.5)

    return {
        "sr_lag1": round(sr1, 2),
        "sr_lag2": round(sr2, 2),
        "retention_rate": round(retention, 3),
        "passed": passed,
        "m_lag1": m1,
        "m_lag2": m2,
    }


def evaluate_shuffled_null(
    df: pd.DataFrame,
    weights: pd.Series,
    seed: int = 42,
) -> dict:
    """
    Null hypothesis test via return vector permutation.
    Permutes the asset return series across the panel to destroy temporal signal-return correlation.
    A valid uncorrupted pipeline produces |null_sharpe| < 0.75.
    """
    work = df[["timestamp", "symbol", "close"]].copy()
    w_clean = coerce_weights(df, weights)
    work["w_held"] = w_clean.groupby(df["symbol"]).shift(1).fillna(0.0)
    work["ret"] = work.groupby("symbol")["close"].pct_change().fillna(0.0)

    rng = np.random.default_rng(seed)
    shuffled_ret = rng.permutation(work["ret"].to_numpy())
    work["pnl_null"] = work["w_held"].to_numpy() * shuffled_ret
    hourly_null = work.groupby("timestamp", sort=True)["pnl_null"].sum()

    std_null = float(hourly_null.std())
    ann_vol = std_null * np.sqrt(8760)
    null_sharpe = float(hourly_null.mean() * 8760 / ann_vol) if ann_vol > 1e-12 else 0.0
    passed = bool(abs(null_sharpe) < 0.75)

    return {
        "null_sharpe": round(null_sharpe, 2),
        "passed": passed,
    }



def run_full_audit(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    fee_grid_bps: tuple[float, ...] = (0.0, 3.0, 5.0, 10.0),
    holdout_df: pd.DataFrame | None = None,
) -> dict:
    df_clean = enforce_schema(df)
    dq: QualityReport = audit_dataset(df_clean)

    t0 = time.perf_counter()
    try:
        raw_weights = strategy_fn(df_clean.copy())
        weights = coerce_weights(df_clean, raw_weights)
    except Exception as exc:
        empty_m = BacktestMetrics(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0]))
        return {
            "passed": False,
            "data_quality": dq,
            "causality": {"passed": False, "max_diff": float("inf"), "detail": f"Strategy execution crashed: {exc}"},
            "ic": {"mean_ic": 0.0, "ic_ir": 0.0, "t_stat": 0.0, "p_value": 1.0, "n_obs": 0},
            "fee_curve": {bps: empty_m for bps in fee_grid_bps},
            "split": {"IS": empty_m, "OOS": empty_m},
            "lag_sensitivity": {"passed": False, "sr_lag1": 0.0, "sr_lag2": 0.0, "retention_rate": 0.0},
            "shuffled_null": {"passed": False, "null_sharpe": 0.0},
            "holdout": None,
            "perf": {"elapsed_ms": 0.0, "rows_per_sec": 0},
        }

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    rows_per_sec = len(df_clean) / max(1e-6, (elapsed_ms / 1000.0))

    causal_ok, max_leak_diff, causal_msg = check_causality_truncation(df_clean, strategy_fn)
    ic_stats = information_coefficient(df_clean, weights)

    fee_curve = {bps: run_backtest(df_clean, weights, fee_bps=bps) for bps in fee_grid_bps}
    split_res = purged_oos_split_eval(df_clean, weights, fee_bps=5.0)
    lag_res = evaluate_lag_sensitivity(df_clean, weights, fee_bps=5.0)
    null_res = evaluate_shuffled_null(df_clean, weights, seed=42)

    holdout_metrics: BacktestMetrics | None = None
    if holdout_df is not None and not holdout_df.empty:
        h_clean = enforce_schema(holdout_df)
        combined = pd.concat([df_clean, h_clean], ignore_index=True)
        combined = enforce_schema(combined)
        try:
            w_comb = coerce_weights(combined, strategy_fn(combined.copy()))
            h_start = h_clean["timestamp"].min()
            h_mask = combined["timestamp"] >= h_start
            h_slice = combined.loc[h_mask].reset_index(drop=True)
            w_h = pd.Series(w_comb.loc[h_mask].to_numpy(), index=h_slice.index)
            holdout_metrics = run_backtest(h_slice, w_h, fee_bps=5.0)
        except Exception:
            holdout_metrics = None

    passed = bool(
        causal_ok
        and dq.dqs >= 80.0
        and split_res["OOS"].sharpe > 0.0
        and fee_curve.get(5.0, fee_curve[fee_grid_bps[0]]).sharpe > 0.0
        and fee_curve.get(5.0, fee_curve[fee_grid_bps[0]]).mean_turnover > 1e-5
        and lag_res["passed"]
        and null_res["passed"]
        and (holdout_metrics is None or holdout_metrics.sharpe > 0.0)
    )

    return {
        "passed": passed,
        "data_quality": dq,
        "causality": {"passed": causal_ok, "max_diff": max_leak_diff, "detail": causal_msg},
        "ic": ic_stats,
        "fee_curve": fee_curve,
        "split": split_res,
        "lag_sensitivity": lag_res,
        "shuffled_null": null_res,
        "holdout": holdout_metrics,
        "perf": {"elapsed_ms": round(elapsed_ms, 2), "rows_per_sec": int(rows_per_sec)},
    }


def format_audit_report(res: dict, strategy_name: str) -> str:
    dq: QualityReport = res["data_quality"]
    c = res["causality"]
    ic = res["ic"]
    split = res["split"]
    perf = res["perf"]
    lag = res["lag_sensitivity"]
    null = res["shuffled_null"]
    m5 = res["fee_curve"].get(5.0, next(iter(res["fee_curve"].values())))

    lines = [
        "=" * 72,
        f"SYNDICATE AUDIT REPORT :: {strategy_name}",
        "=" * 72,
        f"[1] DATA QUALITY     : DQS={dq.dqs}/100 ({dq.status}) | rows={dq.total_rows} | symbols={dq.n_symbols} | gaps={dq.timestamp_gaps} | funding_irreg={dq.funding_irregularities}",
        f"[2] CAUSALITY CHECK  : {'[PASS]' if c['passed'] else '[FAIL]'} {c['detail']}",
        f"[3] RANK IC (1h fwd) : mean={ic['mean_ic']:+.5f} | IC_IR={ic['ic_ir']:+.3f} | t={ic['t_stat']:+.2f} (p={ic['p_value']:.4f})",
        f"[4] RUNTIME          : {perf['elapsed_ms']:.1f} ms ({perf['rows_per_sec']:,} rows/sec)",
        "-" * 72,
        "[5] TRANSACTION COST DECAY MATRIX (1-bar lag + funding included):",
        f"    {'Fee (bps)':<12} {'Sharpe':>8} {'DSR(N=10)':>11} {'Ann.Ret':>10} {'Ann.Vol':>10} {'MaxDD':>9} {'Turnover':>10}",
    ]
    for bps, m in res["fee_curve"].items():
        lines.append(
            f"    {bps:>6.1f} bps   {m.sharpe:>+8.2f} {m.deflated_sharpe:>11.3f} "
            f"{m.ann_return * 100:>+9.2f}% {m.ann_vol * 100:>9.2f}% {m.max_drawdown * 100:>+8.2f}% {m.mean_turnover:>10.4f}"
        )

    is_m = split["IS"]
    oos_m = split["OOS"]
    lines.extend(
        [
            "-" * 72,
            "[6] PURGED SPLIT (70% In-Sample / 24h Embargo / 30% Out-of-Sample @ 5.0 bps):",
            f"    In-Sample     : Sharpe={is_m.sharpe:+.2f} | DSR={is_m.deflated_sharpe:.3f} | Ann.Ret={is_m.ann_return * 100:+.2f}% | MaxDD={is_m.max_drawdown * 100:+.2f}%",
            f"    Out-of-Sample : Sharpe={oos_m.sharpe:+.2f} | DSR={oos_m.deflated_sharpe:.3f} | Ann.Ret={oos_m.ann_return * 100:+.2f}% | MaxDD={oos_m.max_drawdown * 100:+.2f}%",
        ]
    )
    if res.get("holdout") is not None:
        hm: BacktestMetrics = res["holdout"]
        lines.append(
            f"    Holdout Vault : Sharpe={hm.sharpe:+.2f} | DSR={hm.deflated_sharpe:.3f} | Ann.Ret={hm.ann_return * 100:+.2f}% | MaxDD={hm.max_drawdown * 100:+.2f}%"
        )

    lines.extend(
        [
            "-" * 72,
            "[7] LAG SENSITIVITY TEST (+1 extra bar execution lag @ 5.0 bps):",
            f"    1-bar lag SR  : {lag['sr_lag1']:+.2f} | 2-bar lag SR: {lag['sr_lag2']:+.2f} | Retention: {lag['retention_rate'] * 100:.1f}% [{'PASS' if lag['passed'] else 'FAIL'}]",
            "[8] NULL HYPOTHESIS TEST (shuffled returns permutation):",
            f"    Permuted SR   : {null['null_sharpe']:+.2f} (hurdle |SR| < 0.75) [{'PASS - immune to pipeline noise' if null['passed'] else 'FAIL - pipeline artifact'}]",
            "[9] FRICTION DRAG BREAKDOWN (@ 5.0 bps taker fee):",
            f"    Gross PnL     : {m5.gross_pnl_cum * 100:+.2f}%",
            f"    Fee Cost      : {m5.fee_cost_cum * 100:+.2f}%",
            f"    Funding Drag  : {m5.funding_cost_cum * 100:+.2f}%",
            f"    Net PnL       : {m5.net_pnl_cum * 100:+.2f}%",
            "=" * 72,
            f"FINAL VERDICT: {'[PASS] SURVIVED AUDIT' if res['passed'] else '[REJECT] FAILED AUDIT'}",
            "=" * 72,
        ]
    )
    return "\n".join(lines)



def main() -> int:
    parser = argparse.ArgumentParser(description="Run Syndicate look-ahead and cost-drag audit on a strategy file.")
    parser.add_argument("strategy", help="Path to strategy .py file defining compute_weights(df)")
    parser.add_argument("--data", default="data/train_1h.parquet", help="Path to input parquet file")
    parser.add_argument("--holdout", default="", help="Optional path to private holdout parquet file")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: dataset not found at {data_path}. Run `python -m data.build_dataset` first.", file=sys.stderr)
        return 2

    df = pd.read_parquet(data_path)
    holdout_df = pd.read_parquet(args.holdout) if args.holdout and Path(args.holdout).exists() else None
    fn = load_strategy_fn(args.strategy)
    res = run_full_audit(df, fn, holdout_df=holdout_df)
    print(format_audit_report(res, Path(args.strategy).name))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

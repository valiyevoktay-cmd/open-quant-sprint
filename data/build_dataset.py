from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

from src.quality import audit_dataset, autoclean_market_data
from src.schema import enforce_schema, normalize_binance_klines

DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "NEARUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "ARBUSDT",
    "OPUSDT",
    "APTUSDT",
    "SUIUSDT",
    "INJUSDT",
    "AAVEUSDT",
)

BINANCE_FAPI_BASE = "https://fapi.binance.com"


def _http_get_json(url: str, retries: int = 3, backoff_sec: float = 1.0) -> list:
    headers = {"User-Agent": "syndicate-data-loader/0.1"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed after {retries} attempts ({url}): {exc}") from exc
            time.sleep(backoff_sec * (2**attempt))
    return []


def fetch_binance_symbol_1h(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    klines: list[list] = []
    cursor = start_ms
    step_ms = 3600 * 1000

    while cursor < end_ms:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1500}
        )
        batch = _http_get_json(f"{BINANCE_FAPI_BASE}/fapi/v1/klines?{params}")
        if not batch:
            break
        klines.extend(batch)
        last_open = int(batch[-1][0])
        if last_open <= cursor:
            break
        cursor = last_open + step_ms
        time.sleep(0.12)

    f_rows: list[dict] = []
    f_cursor = start_ms
    while f_cursor < end_ms:
        params = urllib.parse.urlencode(
            {"symbol": symbol, "startTime": f_cursor, "endTime": end_ms, "limit": 1000}
        )
        batch = _http_get_json(f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate?{params}")
        if not batch:
            break
        f_rows.extend(batch)
        last_ft = int(batch[-1]["fundingTime"])
        if last_ft <= f_cursor:
            break
        f_cursor = last_ft + 1000
        time.sleep(0.10)

    funding_df = pd.DataFrame(f_rows) if f_rows else None
    return normalize_binance_klines(klines, symbol=symbol, funding_df=funding_df)


def generate_calibrated_panel(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    n_hours: int = 4320,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a cross-sectional 1h futures panel with fat-tailed returns,
    time-varying volatility, realistic 8h funding rates, and subtle cross-sectional
    order-flow / funding dislocation effects for offline benchmarking.
    """
    rng = np.random.default_rng(seed)
    n_sym = len(symbols)
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")

    base_prices = np.linspace(20.0, 50000.0, n_sym)
    mkt_ret = rng.standard_t(df=5, size=n_hours) * 0.0045
    betas = rng.uniform(0.75, 1.35, size=n_sym)

    frames = []
    # Shared cross-sectional latent flow shock
    flow_state = np.zeros((n_hours, n_sym), dtype=np.float64)
    funding_state = np.zeros((n_hours, n_sym), dtype=np.float64)

    for s_idx in range(n_sym):
        f_ar = float(rng.normal(0.0, 0.10))
        fund_ar = float(rng.normal(0.00005, 0.00015))
        for t in range(n_hours):
            f_ar = 0.72 * f_ar + rng.normal(0.0, 0.08)
            flow_state[t, s_idx] = np.clip(f_ar, -0.45, 0.45)
            if t % 8 == 0:
                fund_ar = 0.78 * fund_ar + 0.00012 * f_ar + rng.normal(0.0, 0.00012)
                funding_state[t, s_idx] = np.clip(fund_ar, -0.0025, 0.0025)
            else:
                funding_state[t, s_idx] = 0.0

    # Cross-sectional demeaned flow & extreme funding reversion drive a small next-bar drift
    flow_cs = flow_state - flow_state.mean(axis=1, keepdims=True)
    fund_roll = np.zeros_like(funding_state)
    for t in range(n_hours):
        start_w = max(0, t - 23)
        fund_roll[t] = funding_state[start_w : t + 1].sum(axis=0)
    fund_cs = fund_roll - fund_roll.mean(axis=1, keepdims=True)

    for s_idx, sym in enumerate(symbols):
        idio_noise = rng.standard_t(df=6, size=n_hours) * 0.0052
        ret = np.zeros(n_hours, dtype=np.float64)
        ret[0] = mkt_ret[0] * betas[s_idx] + idio_noise[0]
        for t in range(1, n_hours):
            alpha_drift = 0.0023 * flow_cs[t - 1, s_idx] - 0.16 * fund_cs[t - 1, s_idx]
            ret[t] = mkt_ret[t] * betas[s_idx] + alpha_drift + idio_noise[t]

        close = base_prices[s_idx] * np.exp(np.cumsum(ret))
        open_p = np.roll(close, 1)
        open_p[0] = close[0] * (1.0 - ret[0] * 0.5)

        intrabar_spread = np.abs(ret) + rng.exponential(0.0035, size=n_hours)
        high = np.maximum(open_p, close) * (1.0 + intrabar_spread * rng.uniform(0.3, 0.9, size=n_hours))
        low = np.minimum(open_p, close) * (1.0 - intrabar_spread * rng.uniform(0.3, 0.9, size=n_hours))

        quote_vol = rng.lognormal(mean=15.2, sigma=0.65, size=n_hours) * (1.0 + 40.0 * np.abs(ret))
        taker_ratio = np.clip(0.5 + 0.5 * flow_state[:, s_idx] + rng.normal(0.0, 0.04, size=n_hours), 0.05, 0.95)
        taker_buy_qv = quote_vol * taker_ratio
        vol = quote_vol / close
        trades = (quote_vol / rng.uniform(450.0, 1200.0, size=n_hours)).astype(np.int64)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": sym,
                    "open": open_p,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                    "quote_volume": quote_vol,
                    "taker_buy_quote_volume": taker_buy_qv,
                    "trades": trades,
                    "funding_rate": funding_state[:, s_idx],
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    return autoclean_market_data(panel)


def build_and_save(
    out_dir: str | Path = "data",
    live: bool = False,
    days: int = 180,
    holdout_frac: float = 0.20,
) -> tuple[Path, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    panel: pd.DataFrame | None = None
    if live:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        chunks = []
        try:
            for sym in DEFAULT_SYMBOLS:
                print(f"Fetching {sym} from Binance Futures API...")
                chunks.append(fetch_binance_symbol_1h(sym, start_ms=start_ms, end_ms=end_ms))
            panel = autoclean_market_data(pd.concat(chunks, ignore_index=True))
        except Exception as exc:
            print(f"Live API fetch unavailable ({exc}); falling back to calibrated panel.")
            panel = None

    if panel is None:
        panel = generate_calibrated_panel(n_hours=days * 24)

    panel = enforce_schema(panel)
    dq = audit_dataset(panel)
    print(f"Dataset built: rows={dq.total_rows:,}, symbols={dq.n_symbols}, DQS={dq.dqs}/100 ({dq.status})")

    unique_ts = np.sort(panel["timestamp"].unique())
    cut_idx = int(len(unique_ts) * (1.0 - holdout_frac))
    t_cut = unique_ts[cut_idx - 1]

    train_df = panel.loc[panel["timestamp"] <= t_cut].reset_index(drop=True)
    holdout_df = panel.loc[panel["timestamp"] > t_cut].reset_index(drop=True)

    train_file = out_path / "train_1h.parquet"
    holdout_file = out_path / "holdout_1h.parquet"

    train_df.to_parquet(train_file, compression="zstd", index=False)
    holdout_df.to_parquet(holdout_file, compression="zstd", index=False)

    print(f"Saved train   : {train_file} ({train_file.stat().st_size / 1024:.1f} KB, {len(train_df):,} rows)")
    print(f"Saved holdout : {holdout_file} ({holdout_file.stat().st_size / 1024:.1f} KB, {len(holdout_df):,} rows)")
    return train_file, holdout_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cleaned 1h parquet datasets for Syndicate sprint.")
    parser.add_argument("--live", action="store_true", help="Attempt live download from Binance Futures REST API")
    parser.add_argument("--days", type=int, default=180, help="Number of days of 1h bars to build")
    parser.add_argument("--out-dir", default="data", help="Output directory for parquet files")
    args = parser.parse_args()

    build_and_save(out_dir=args.out_dir, live=args.live, days=args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

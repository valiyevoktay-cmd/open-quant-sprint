"""Track 03: fixed-rule paper strategy and two research benchmarks.

compute_weights is the STRONG-FLOW catch-up specification, fixed in advance of
the economic experiment. Research comparisons do not automatically change it.
All decisions use closed bars only. No IO, future labels or persistent state.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contrib.janice import leader_flow_impulse
from src.contrib.janice_gaps import on_full_grid
from src.features import cross_sectional_zscore


@dataclass(frozen=True)
class TradingRules:
    gap_window: int = 4
    beta_window: int = 168
    impulse_span: int = 4
    quantile_window: int = 720
    strong_quantile: float = 0.90
    rebalance_hours: int = 8
    deadband: float = 0.015
    clip: float = 3.0


def prepare_signals(df, rules=TradingRules()):
    """Causal signals on a balanced hourly panel; retain original row positions."""
    if not set(['timestamp','symbol','close','quote_volume','taker_buy_quote_volume']) <= set(df):
        raise ValueError('Missing required signal columns')
    d=df.copy()
    d['_input_position']=np.arange(len(d))
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True)
    if d[['timestamp','symbol']].isna().any().any() or d.duplicated(['timestamp','symbol']).any():
        raise ValueError('Missing or duplicate timestamp/symbol key')
    d=d.sort_values(['timestamp','symbol']).reset_index(drop=True)
    close=d.pivot(index='timestamp',columns='symbol',values='close').astype(float)
    leaders=['BTCUSDT','ETHUSDT']
    if close.shape[1]!=20 or not set(leaders)<=set(close.columns) or close.isna().any().any():
        raise ValueError('Expected complete BTC/ETH plus 18-alt cross-sections')
    if not np.isfinite(close.to_numpy()).all() or not close.gt(0).all().all():
        raise ValueError('Prices must be finite and positive')
    # DatetimeIndex.equals also compares timestamp resolution (s/ms/us/ns).
    # The audit may preserve a different resolution than a notebook; compare
    # elapsed time in a common unit, not representation-dependent equality.
    hours_ns=close.index.as_unit('ns').asi8
    if not np.all(np.diff(hours_ns)==3_600_000_000_000):
        raise ValueError('Panel must have contiguous hourly timestamps')
    if not (close.index==close.index.floor('h')).all():
        raise ValueError('Expected UTC hour-aligned timestamps')
    q=d.pivot(index='timestamp',columns='symbol',values='quote_volume').astype(float)
    b=d.pivot(index='timestamp',columns='symbol',values='taker_buy_quote_volume').astype(float)
    if not (np.isfinite(q.to_numpy()).all() and np.isfinite(b.to_numpy()).all()
            and q.ge(0).all().all() and b.ge(0).all().all() and b.le(q).all().all()):
        raise ValueError('Invalid volume bounds')
    returns=np.log(close).diff()
    lead=(returns[leaders]*q[leaders]).sum(axis=1,min_count=2).div(
        q[leaders].sum(axis=1).where(lambda v:v>0))
    alts=close.columns.difference(leaders)
    covariance=returns[alts].rolling(rules.beta_window,min_periods=rules.beta_window).cov(lead).shift(1)
    variance=lead.rolling(rules.beta_window,min_periods=rules.beta_window).var().shift(1)
    beta=covariance.div(variance.where(variance>1e-12),axis=0)
    alt_move=returns[alts].rolling(rules.gap_window).sum()
    leader_move=lead.rolling(rules.gap_window).sum()
    catchup=-(alt_move-beta.mul(leader_move,axis=0))
    reversal=-alt_move
    impulse=leader_flow_impulse(d,span=rules.impulse_span).groupby(d['timestamp']).first().reindex(close.index)
    cutoff=impulse.abs().rolling(rules.quantile_window,min_periods=rules.quantile_window).quantile(
        rules.strong_quantile).shift(1)
    ready=(np.isfinite(catchup).all(axis=1)&np.isfinite(reversal).all(axis=1)
           &impulse.notna()&cutoff.notna())
    strong=(impulse.abs()>cutoff)&ready
    return dict(data=d,close=close,alts=alts,catchup=catchup,reversal=reversal,
                ready=ready,strong=strong,impulse=impulse,cutoff=cutoff)


def threshold_step(previous,target,buffer):
    """Per-symbol no-trade band, matching rebalance_deadband's threshold rule."""
    return np.where(np.abs(target-previous)>buffer,target,previous)


def neutral_unit_gross(values):
    """Recenter and normalize an active book; preserve the zero book."""
    centered=np.asarray(values,dtype=float)-np.mean(values)
    gross=np.abs(centered).sum()
    return centered/gross if gross>1e-14 else np.zeros_like(centered)


def scheduled_book(target,ready,strong,index,rules,gate):
    """Update only every 8 UTC epoch-hours; hold weights between decisions.

    A gated book flattens at a scheduled decision if flow is weak/unknown.
    Flattening bypasses the band; weak flow between decisions does not exit.
    The band is applied against the last ISSUED, neutralized book. Projection
    can move additional names, so the buffer is not a strict final-trade minimum.
    """
    out=np.zeros_like(target,dtype=float)
    previous=np.zeros(target.shape[1],dtype=float)
    epoch_hours=index.as_unit('ns').asi8//3_600_000_000_000
    for t in range(len(index)):
        if epoch_hours[t]%rules.rebalance_hours==0:
            if not ready[t] or (gate and not strong[t]) or np.abs(target[t]).sum()<1e-14:
                previous=np.zeros_like(previous)
            else:
                previous=neutral_unit_gross(threshold_step(previous,target[t],rules.deadband))
        out[t]=previous
    return out


def build_portfolios(df,rules=TradingRules(),variants=('reversal','catchup','catchup_strong')):
    """Return aligned weight Series for fixed research comparisons."""
    if df.empty:
        return {name:pd.Series(index=df.index,dtype=float,name=name) for name in variants}
    prepared=prepare_signals(df,rules)
    d,close,alts=prepared['data'],prepared['close'],prepared['alts']
    alt_rows=d.loc[d['symbol'].isin(alts),['timestamp','symbol']]
    time_pos=close.index.get_indexer(d['timestamp'])
    name_pos=close.columns.get_indexer(d['symbol'])
    portfolios={}
    for name in variants:
        if name not in ('reversal','catchup','catchup_strong'):
            raise ValueError('Unknown portfolio variant')
        signal=prepared['reversal'] if name=='reversal' else prepared['catchup']
        # Stack with explicit array lookup avoids pandas-version-specific stack defaults.
        signal_values=signal.to_numpy()[signal.index.get_indexer(alt_rows['timestamp']),
                                         signal.columns.get_indexer(alt_rows['symbol'])]
        values=cross_sectional_zscore(alt_rows,pd.Series(signal_values,index=alt_rows.index),clip=rules.clip)
        targets=alt_rows.assign(weight=values).pivot(index='timestamp',columns='symbol',values='weight').reindex(
            index=close.index,columns=alts).fillna(0).to_numpy()
        book=scheduled_book(targets,prepared['ready'].to_numpy(),prepared['strong'].to_numpy(),
                            close.index,rules,gate=name=='catchup_strong')
        wide=pd.DataFrame(0.,index=close.index,columns=close.columns)
        wide.loc[:,alts]=book
        sorted_values=wide.to_numpy()[time_pos,name_pos]
        original_values=np.empty(len(d),dtype=float)
        original_values[d['_input_position'].to_numpy()]=sorted_values
        portfolios[name]=pd.Series(original_values,index=df.index,name=name)
    return portfolios


def compute_weights(df:pd.DataFrame)->pd.Series:
    """Audit entry point: predeclared strong-flow beta-adjusted catch-up book."""
    # Maintainer change before the freeze: missing hours are filled as the data builder fills a short gap.
    return on_full_grid(lambda d:build_portfolios(d,variants=('catchup_strong',))['catchup_strong'],df)

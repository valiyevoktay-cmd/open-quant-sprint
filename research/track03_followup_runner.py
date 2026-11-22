"""Fixed two-candidate research runner. No parameter grid or automatic selection."""
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from src.backtest import run_backtest,tiered_fee_bps
from src.contrib.janice_alternatives import (TradingRules,validated_panel,prepare_signals,
    reference_catchup_weights,reference_reversal_weights,gradual_catchup_weights,slow_momentum_weights)

def synthetic_economic_panel(n=1400,seed=41):
    rng=np.random.default_rng(seed)
    times=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC')
    names=['BTCUSDT','ETHUSDT']+[f'ALT{i:02d}USDT' for i in range(18)]
    leader=rng.normal(0,.003,n)
    r=leader[:,None]*np.linspace(.8,1.8,20)+rng.normal(0,.002,(n,20))
    close=100*np.exp(np.cumsum(r,axis=0));op=np.vstack([close[:1],close[:-1]])
    quote=rng.uniform(100,10000,(n,20))
    d=pd.MultiIndex.from_product([times,names],names=['timestamp','symbol']).to_frame(index=False)
    d['open']=op.ravel();d['close']=close.ravel()
    d['high']=np.maximum(op,close).ravel()*1.001
    d['low']=np.minimum(op,close).ravel()*.999
    d['quote_volume']=quote.ravel();d['volume']=(quote/close).ravel()
    d['taker_buy_quote_volume']=(quote*rng.uniform(.03,.97,(n,20))).ravel()
    d['trades']=100
    funding=rng.normal(0,.00005,(n,20))
    funding[np.arange(n)%8!=0]=0
    d['funding_rate']=funding.ravel()
    return d

def execution_exposure(df,w,lag_bars):
    held=w.groupby(df['symbol']).shift(lag_bars).fillna(0)
    gross=held.abs().groupby(df['timestamp']).sum()
    changes=held-held.groupby(df['symbol']).shift(1).fillna(0)
    turn=changes.abs().groupby(df['timestamp']).sum()
    return dict(active_hour_fraction=float(gross.gt(1e-12).mean()),
                average_gross=float(gross.mean()),turnover_total=float(turn.sum()),
                turnover_per_day=float(turn.mean()*24))

def economic_grid(df,weights,fees=(0.,3.,5.,10.),tiered_fee_rows=None):
    rows=[];series={};all_metrics={}
    tiered=tiered_fee_bps(df) if tiered_fee_rows is None else tiered_fee_rows
    for name,w in weights.items():
        for lag in (1,2):
            exposure=execution_exposure(df,w,lag)
            for fee_label,fee in [(f'{f:g}bps',f) for f in fees]+[('tiered',tiered)]:
                m=run_backtest(df,w,fee_bps=fee,include_funding=True,lag_bars=lag)
                key=(name,lag,fee_label)
                all_metrics[key]=m;series[key]=m.hourly_returns
                rows.append(dict(portfolio=name,lag_bars=lag,cost=fee_label,
                    sharpe=m.sharpe,sharpe_ci_low=m.sharpe_ci_low,sharpe_ci_high=m.sharpe_ci_high,
                    annualized_mean=m.ann_return,max_drawdown_notional=m.max_drawdown,
                    cumulative_net_pnl=m.net_pnl_cum,gross_price_pnl=m.gross_pnl_cum,
                    fee_cost=m.fee_cost_cum,funding_cost=m.funding_cost_cum,
                    **exposure))
    summary=pd.DataFrame(rows).set_index(['portfolio','lag_bars','cost'])
    hours=pd.DatetimeIndex(sorted(df['timestamp'].unique()))
    returns=pd.DataFrame(series,index=hours)
    returns.columns=pd.MultiIndex.from_tuples(returns.columns,names=['portfolio','lag_bars','cost'])
    return summary,returns,all_metrics

def paired_mean_interval(diff,block=168,reps=1000,seed=20261116):
    """Interval for mean hourly NET PnL difference, not difference in Sharpe."""
    y=np.asarray(diff,dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Paired PnL comparisons require identical finite hourly samples')
    n=len(y)
    if n<2*block:
        return np.nan,np.nan
    rng=np.random.default_rng(seed);full,rem=divmod(n,block)
    a=np.r_[y,y[:block]];cs=np.r_[0.,np.cumsum(a)]
    starts=rng.integers(0,n,size=(reps,full))
    sums=(cs[starts+block]-cs[starts]).sum(axis=1)
    if rem:
        s=rng.integers(0,n,size=reps);sums+=cs[s+rem]-cs[s]
    return tuple(np.quantile(sums/n,[.025,.975]))

def breakeven_fees(returns):
    rows=[]
    for name in returns.columns.get_level_values(0).unique():
        for lag in (1,2):
            r0=returns[(name,lag,'0bps')]
            r5=returns[(name,lag,'5bps')]
            total_turn=(r0-r5).sum()/.0005
            # At constant weights and linear fees, break-even SUM/MEAN PnL:
            # gross price PnL - funding - fee*turnover = 0.
            raw_bps=r0.sum()/total_turn*10000 if total_turn>1e-12 else np.nan
            rows.append(dict(portfolio=name,lag_bars=lag,
                breakeven_fee_bps=raw_bps if raw_bps>=0 else np.nan,
                note=('No nonnegative fee breaks even' if raw_bps<0 else
                      'No turnover / undefined' if not np.isfinite(raw_bps) else
                      'Linear fee-only estimate; funding included; no spread/impact')))
    return pd.DataFrame(rows)

def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def specification(candidate):
    if candidate not in ('A','B'):
        raise ValueError('candidate must be A or B')
    return dict(candidate=candidate,trial_budget='one fixed new specification per candidate; no retuning',
        hypothesis=('Gradual adjustment reduces turnover enough to improve net economics despite added lag.'
                    if candidate=='A' else 'A slower momentum effect may persist through execution delay and daily trading costs.'),
        reference='ungated original catchup' if candidate=='A' else 'original fast reversal (context, not a single-factor control)',
        universe='18 alts; BTC/ETH zero',burnin_hours=720,
        rules=(dict(gap_hours=4,beta_hours=168,flow_span_for_shared_burnin=4,
                    historical_flow_quantile=.9,quantile_hours=720,flow_gate=False,
                    decision_hours=8,target_deadband=.015,clip=3,
                    target='original ungated issued catchup book, including its existing band/projection',
                    update='0.75*previous+0.25*reference_target',renormalize_after_blend=False)
               if candidate=='A' else dict(return_hours=168,volatility_hours=168,
                    volatility='std(ddof=1) of hourly log returns, through current closed bar',
                    score='168h log return / 168h hourly log-return volatility',
                    decision_hours=24,decision_phase='UTC epoch-hour modulo24=0',
                    deadband=.015,clip=3,beta=False,flow_gate=False,
                    weighting='alt-only cross-sectional zscore; centered and unit-gross when active')),
        common=dict(fees_bps=[0,3,5,10,'repository tiered'],funding=True,
                    lag_bars=[1,2],primary_cost_bps=5,primary_lag_bars=2,
                    bootstrap_hours=168,bootstrap_reps=1000,seed=20261116,
                    accounting='repository additive fixed-notional PnL; no terminal forced close',
                    sample='same dates for candidate/reference, beginning after720 prior hours',
                    scope='exploratory training reanalysis; earlier full-panel results already seen',
                    limitations='no spread, impact, margin liquidation, fill uncertainty or sealed validation'))


def plan_text(candidate):
    return ("FIXED EXPLORATORY FOLLOW-UP PLAN — CANDIDATE "+candidate+"\n\n"
        "Record this in your PR BEFORE the new real-data performance run. Previous IC,\n"
        "beta diagnostics and economic results have already been examined. A local\n"
        "plan file is not evidence that it was posted. Keep the original audited\n"
        "strategy unchanged. Run each candidate once as planned, disclose any bug-fix\n"
        "reruns, and report every cost/delay outcome. No search over windows, signs,\n"
        "fractions, clock phases or cutoffs. A post-hoc historical split is not fresh\n"
        "validation. Do not tune on the sprint holdout or forward periods.\n\n"
        "Success is not merely losing less through lower exposure. Inspect gross/net\n"
        "PnL, fees, turnover per unit exposure, delay sensitivity and uncertainty.\n"
        "Candidate A additionally has a descriptive, EX-POST constant-scaled reference\n"
        "matching its average gross. This uses realized future exposures, is NOT a\n"
        "prospective strategy, has no inferential CI and is never audited/submitted.\n"
        "It is not volatility/risk matching. Cash (zero modeled PnL) is also a reference.\n"
        "Paired intervals are pointwise and exploratory, not multiple-testing-adjusted.\n\n"
        +json.dumps(specification(candidate),indent=2,ensure_ascii=False)+"\n")


def make_books(df,candidate):
    if candidate=='A':
        return {'reference':reference_catchup_weights(df),'candidate':gradual_catchup_weights(df)}
    if candidate=='B':
        return {'reference':reference_reversal_weights(df),'candidate':slow_momentum_weights(df)}
    raise ValueError('Unknown candidate')


def check_books(df,candidate):
    books=make_books(df,candidate)
    for name,w in books.items():
        if not w.index.equals(df.index) or not np.isfinite(w).all():
            raise AssertionError('Bad weight index or non-finite weights')
        if not w.loc[df['symbol'].isin(['BTCUSDT','ETHUSDT'])].eq(0).all():
            raise AssertionError('Leaders must have zero weights')
        if w.groupby(df['timestamp']).sum().abs().max()>1e-12:
            raise AssertionError('Non-neutral book')
        if w.abs().groupby(df['timestamp']).sum().max()>1+1e-12:
            raise AssertionError('Gross exposure exceeds1')
    hours=pd.DatetimeIndex(sorted(pd.to_datetime(df['timestamp'],utc=True).unique()))
    for k in sorted(set([len(hours)//2,int(.85*len(hours)),len(hours)-2])):
        mask=pd.to_datetime(df['timestamp'],utc=True)<=hours[k]
        short=make_books(df.loc[mask],candidate)
        for name in books:
            assert_series_equal(short[name],books[name].loc[mask],atol=1e-10,rtol=0)
    changed=df.copy();future=pd.to_datetime(df['timestamp'],utc=True)>hours[len(hours)//2]
    changed.loc[future,'close']*=1.4
    changed.loc[future,'taker_buy_quote_volume']=changed.loc[future,'quote_volume']
    altered=make_books(changed,candidate)
    repeated=make_books(df,candidate)
    for name in books:
        assert_series_equal(altered[name].loc[~future],books[name].loc[~future],atol=1e-10,rtol=0)
        assert_series_equal(repeated[name],books[name])
    return books


def prepare_evaluation(df,books):
    hours=pd.DatetimeIndex(sorted(pd.to_datetime(df['timestamp'],utc=True).unique()))
    if len(hours)<=720:
        raise ValueError('Need more than720 hours for the common reporting period')
    start=hours[720]
    mask=pd.to_datetime(df['timestamp'],utc=True)>=start
    for w in books.values():
        if w.loc[~mask].abs().max()>1e-12:
            raise AssertionError('Nonzero book before common operational burn-in')
    d=df.loc[mask].reset_index(drop=True)
    trimmed={name:pd.Series(w.loc[mask].to_numpy(),index=d.index,name=name) for name,w in books.items()}
    full_tiered=pd.Series(tiered_fee_bps(df),index=df.index)
    return d,trimmed,start,full_tiered.loc[mask].to_numpy()


def paired_tables(returns):
    rows=[]
    for lag in (1,2):
        a=returns[('candidate',lag,'5bps')]
        for j,(label,b) in enumerate([('candidate minus reference',returns[('reference',lag,'5bps')]),
                                      ('candidate minus cash',a*0)]):
            diff=a-b
            low,high=paired_mean_interval(diff,seed=20261116+100*lag+j)
            rows.append(dict(lag_bars=lag,comparison=label,mean_hourly_difference=diff.mean(),
                annualized_mean_difference=diff.mean()*8760,
                annualized_ci_low=low*8760,annualized_ci_high=high*8760))
    return pd.DataFrame(rows)


def exposure_control(summary,returns):
    """DESCRIPTIVE ex-post mean-gross matching; not a causal trading rule."""
    rows=[]
    for lag in (1,2):
        ref=summary.loc[('reference',lag,'5bps')]
        new=summary.loc[('candidate',lag,'5bps')]
        factor=new['average_gross']/ref['average_gross'] if ref['average_gross']>1e-12 else np.nan
        r=returns[('reference',lag,'5bps')]*factor
        rows.append(dict(lag_bars=lag,expost_scaling_factor=factor,
            candidate_average_gross=new['average_gross'],
            candidate_net_pnl=new['cumulative_net_pnl'],
            descriptive_scaled_reference_net_pnl=r.sum(),
            warning='EX-POST mean-exposure comparison only; not prospective, not risk matching, no CI'))
    return pd.DataFrame(rows)


def run_study(root,candidate,plan_path,data_path=None,synthetic=False,out_parent=None):
    root=Path(root);plan_path=Path(plan_path)
    if not plan_path.is_file() or plan_path.read_text(encoding='utf-8')!=plan_text(candidate):
        raise ValueError('Generate a matching plan first; preserve any prior plans/results.')
    data_path=Path(data_path) if data_path is not None else root/'data'/'train_1h.parquet'
    df=synthetic_economic_panel(1400) if synthetic else pd.read_parquet(data_path)
    # Validate and standardize input, preserving all schema fields for the backtest.
    df=validated_panel(df)[0].drop(columns=['_input_position']).reset_index(drop=True)
    if 'funding_rate' not in df or not np.isfinite(df['funding_rate'].to_numpy(dtype=float)).all():
        raise ValueError('Actual finite funding_rate is required; no invented real-data zeros')
    books=check_books(df,candidate)
    d,trimmed,start,tiered=prepare_evaluation(df,books)
    summary,returns,_=economic_grid(d,trimmed,tiered_fee_rows=tiered)
    summary['annualized_volatility']=[returns[key].std(ddof=1)*np.sqrt(8760) for key in summary.index]
    summary['turnover_per_day_per_average_gross']=summary['turnover_per_day'].div(
        summary['average_gross'].where(summary['average_gross']>1e-12))
    pairs=paired_tables(returns)
    breakeven=breakeven_fees(returns)
    control=exposure_control(summary,returns) if candidate=='A' else None
    q=returns.loc[:,[(name,2,'5bps') for name in ('reference','candidate')]].copy()
    q.columns=['reference','candidate']
    quarters=q.groupby(q.index.tz_localize(None).to_period('Q')).sum()
    # Frozen full-input reference to explain audit vs trimmed-sample differences.
    full_rows=[]
    for name,w in books.items():
        m=run_backtest(df,w,fee_bps=5,lag_bars=1)
        full_rows.append(dict(book=name,sharpe=m.sharpe,net_pnl=m.net_pnl_cum))
    audit_reference=pd.DataFrame(full_rows)
    out_parent=Path(out_parent) if out_parent else root/'results'
    out=out_parent/('track03_candidate_'+candidate.lower()+'_'+('synthetic_' if synthetic else 'real_')+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    summary.to_csv(out/'all_cost_delay_results.csv');returns.to_csv(out/'hourly_net_pnl.csv')
    pairs.to_csv(out/'paired_comparisons.csv',index=False);breakeven.to_csv(out/'breakeven_fees.csv',index=False)
    quarters.to_csv(out/'quarterly_5bps_delayed.csv');audit_reference.to_csv(out/'full_panel_reference.csv',index=False)
    if control is not None:control.to_csv(out/'EXPOST_exposure_control_NOT_strategy.csv',index=False)
    pd.DataFrame(books).to_csv(out/'issued_weights.csv',index_label='input_index')
    (out/'recorded_plan.txt').write_text(plan_text(candidate),encoding='utf-8')
    paths=['src/contrib/janice_alternatives.py','src/contrib/janice.py','src/backtest.py',
           'src/features.py','src/stats.py','research/track03_followup_runner.py',
           'strategies/janice_gradual_catchup.py' if candidate=='A' else 'strategies/janice_slow_momentum.py']
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),specification=specification(candidate),
        synthetic=synthetic,data_sha256=None if synthetic else file_hash(data_path),
        synthetic_seed=41 if synthetic else None,evaluation_start=str(start),
        evaluation_end=str(returns.index[-1]),n_evaluation_hours=len(returns),
        python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        source_sha256={p:file_hash(root/p) for p in paths},
        checks='final-weight constraints,3 prefix cuts,future perturbation,determinism passed',
        plan_sha256=hashlib.sha256(plan_text(candidate).encode()).hexdigest(),
        warning='Exploratory training study. Not a sealed validation, official verdict or guarantee of profit.')
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return dict(summary=summary,returns=returns,pairs=pairs,breakeven=breakeven,
                quarters=quarters,control=control,out=out,metadata=metadata)


def plot_study(result,candidate,synthetic=False):
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    r=result['returns'];s=result['summary']
    fig,axes=plt.subplots(1,2,figsize=(12,3.6),sharey=True)
    for ax,lag in zip(axes,(1,2)):
        for book in ('reference','candidate'):
            ax.plot(r.index,r[(book,lag,'5bps')].cumsum(),label=book,lw=1)
        if candidate=='A':
            c=result['control'].set_index('lag_bars').loc[lag,'expost_scaling_factor']
            ax.plot(r.index,(r[('reference',lag,'5bps')]*c).cumsum(),':',label='EX-POST exposure control',lw=1)
        ax.axhline(0,color='black',lw=.6);ax.set_title(f'Candidate {candidate},5bps,lag_bars={lag}')
        loc=mdates.AutoDateLocator(minticks=3,maxticks=5)
        ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Cumulative net PnL / initial notional (additive)')
    fig.suptitle('SYNTHETIC — not market results' if synthetic else 'Exploratory training results')
    fig.tight_layout();plt.show()
    fig,axes=plt.subplots(1,2,figsize=(11,3.4))
    for ax,lag in zip(axes,(1,2)):
        for book in ('reference','candidate'):
            ax.plot([0,3,5,10],[s.loc[(book,lag,f'{f}bps'),'cumulative_net_pnl'] for f in (0,3,5,10)],'o-',label=book)
        ax.axhline(0,color='black',lw=.6);ax.set_title(f'Fee sensitivity,lag_bars={lag}')
        ax.set_xlabel('Flat fee (bps)');ax.legend(fontsize=8)
    axes[0].set_ylabel('Cumulative net PnL / initial notional')
    fig.tight_layout();plt.show()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--candidate',choices=['A','B'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data/train_1h.parquet'))
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.action=='plan':
        args.plan.parent.mkdir(parents=True,exist_ok=True)
        with args.plan.open('x',encoding='utf-8') as f:f.write(plan_text(args.candidate))
        print('Wrote',args.plan,'— post before new real-data performance. No market data read.')
        return
    result=run_study(root,args.candidate,args.plan,args.data,args.synthetic)
    print(result['summary'].xs((2,'5bps'),level=('lag_bars','cost')).to_string())
    print(result['pairs'].to_string(index=False));print('Saved:',result['out'])


if __name__=='__main__':main()

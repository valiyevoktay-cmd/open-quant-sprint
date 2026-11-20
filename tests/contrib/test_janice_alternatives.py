import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal,assert_frame_equal

from src.contrib.janice_alternatives import (gradual_catchup_weights,slow_momentum_weights,
    reference_catchup_weights,reference_reversal_weights,neutral_unit_gross,threshold_step)
from research.track03_followup_runner import (synthetic_economic_panel,check_books,
    prepare_evaluation,make_books,plan_text,paired_mean_interval)
from src.backtest import run_backtest
from src.features import cross_sectional_zscore


class FixedAlternativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=synthetic_economic_panel(1100)

    def test_a_first_position_is_quarter_target_not_full_gross(self):
        d=self.df
        ref=reference_catchup_weights(d)
        a=gradual_catchup_weights(d)
        gross=ref.abs().groupby(d['timestamp']).sum()
        first=gross.index[gross>1e-12][0]
        mask=d['timestamp'].eq(first)
        np.testing.assert_allclose(a[mask],.25*ref[mask],atol=1e-14)
        self.assertAlmostEqual(a[mask].abs().sum(),.25,places=12)

    def test_a_matches_fixed_convex_rule_at_every_decision(self):
        ref=reference_catchup_weights(self.df).to_numpy().reshape(-1,20)
        actual=gradual_catchup_weights(self.df).to_numpy().reshape(-1,20)
        # Fixture row order is fixed per timestamp; retain that same order.
        hours=pd.DatetimeIndex(self.df['timestamp'].unique())
        clock=hours.as_unit('ns').asi8//3_600_000_000_000
        prior=np.zeros(20)
        for i in range(len(hours)):
            expected=.75*prior+.25*ref[i] if clock[i]%8==0 else prior
            np.testing.assert_allclose(actual[i],expected,atol=1e-12)
            prior=actual[i]

    def test_b_changes_only_at_fixed_daily_decisions(self):
        w=slow_momentum_weights(self.df).to_numpy().reshape(-1,20)
        self.assertGreater(np.abs(w).max(),0)
        hours=pd.DatetimeIndex(self.df['timestamp'].unique())
        clock=hours.as_unit('ns').asi8//3_600_000_000_000
        for i in range(1,len(w)):
            if clock[i]%24!=0:np.testing.assert_array_equal(w[i],w[i-1])

    def test_b_is_independent_of_flow_and_volume_values(self):
        changed=self.df.copy()
        changed['quote_volume']=0.;changed['taker_buy_quote_volume']=0.
        assert_series_equal(slow_momentum_weights(self.df),slow_momentum_weights(changed))

    def test_b_first_decision_matches_independent_window_calculation(self):
        d=self.df
        close=d.pivot(index='timestamp',columns='symbol',values='close')
        alts=close.columns.difference(['BTCUSDT','ETHUSDT'])
        t=720
        log=np.log(close[alts].to_numpy())
        historical=np.diff(log[t-168:t+1],axis=0)
        score=(log[t]-log[t-168])/historical.std(axis=0,ddof=1)
        cross=pd.DataFrame({'symbol':alts,'timestamp':close.index[t]})
        target=cross_sectional_zscore(cross,pd.Series(score,index=cross.index)).to_numpy()
        expected=neutral_unit_gross(threshold_step(np.zeros(18),target,.015))
        weights=slow_momentum_weights(d)
        actual=d[['timestamp','symbol']].assign(weight=weights).pivot(
            index='timestamp',columns='symbol',values='weight').loc[close.index[t],alts]
        np.testing.assert_allclose(actual,expected,atol=1e-11)

    def test_a_full_constraints_causality_and_determinism(self):
        books=check_books(self.df,'A')
        self.assertGreater(books['candidate'].abs().max(),0)

    def test_b_full_constraints_causality_and_determinism(self):
        books=check_books(self.df,'B')
        self.assertGreater(books['candidate'].abs().max(),0)

    def test_timestamp_resolution_and_single_bar_prefix(self):
        for fn in (gradual_catchup_weights,slow_momentum_weights):
            reference=fn(self.df)
            for unit in ('ms','us','ns'):
                d=self.df.copy();d['timestamp']=d['timestamp'].dt.as_unit(unit)
                assert_series_equal(reference,fn(d),atol=1e-12,rtol=0)
                self.assertTrue(fn(d.iloc[:20]).eq(0).all())

    def test_shuffle_index_and_input_unchanged(self):
        before=self.df.copy(deep=True)
        shuffled=self.df.sample(frac=1,random_state=19)
        for fn in (gradual_catchup_weights,slow_momentum_weights):
            assert_series_equal(fn(self.df).loc[shuffled.index],fn(shuffled),atol=1e-12,rtol=0)
        assert_frame_equal(self.df,before)

    def test_real_gaps_rejected_and_empty_allowed(self):
        hours=self.df['timestamp'].unique()
        broken=self.df.loc[self.df['timestamp']!=hours[10]]
        for fn in (gradual_catchup_weights,slow_momentum_weights):
            with self.assertRaisesRegex(ValueError,'contiguous'):fn(broken)
            self.assertTrue(fn(self.df.iloc[:0]).empty)

    def test_reporting_trim_preserves_ledger(self):
        for candidate in ('A','B'):
            books=make_books(self.df,candidate)
            d,trimmed,start,fee=prepare_evaluation(self.df,books)
            self.assertEqual(len(d),len(fee))
            mask=np.array(sorted(self.df['timestamp'].unique()))>=start
            for name in books:
                a=run_backtest(self.df,books[name],fee_bps=5,lag_bars=2)
                b=run_backtest(d,trimmed[name],fee_bps=5,lag_bars=2)
                np.testing.assert_allclose(a.hourly_returns[mask],b.hourly_returns,atol=1e-14)

    def test_constant_exposure_scaling_is_only_linear_ledger_diagnostic(self):
        w=reference_catchup_weights(self.df)
        a=run_backtest(self.df,w,fee_bps=5,lag_bars=2)
        b=run_backtest(self.df,.2*w,fee_bps=5,lag_bars=2)
        np.testing.assert_allclose(.2*a.hourly_returns,b.hourly_returns,atol=1e-14)

    def test_plan_and_bootstrap_guards(self):
        self.assertIn('0.75*previous+0.25*reference_target',plan_text('A'))
        self.assertIn('168h log return',plan_text('B'))
        self.assertEqual(paired_mean_interval(np.zeros(500),block=24,reps=100),(0.,0.))
        with self.assertRaises(ValueError):plan_text('C')


if __name__=='__main__':unittest.main()

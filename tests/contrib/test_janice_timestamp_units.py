"""Regression checks for the strategy's timestamp representation compatibility."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from strategies.janice_leadlag import build_portfolios, compute_weights


def panel(n=900):
    rng=np.random.default_rng(41)
    times=pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC')
    names=['BTCUSDT','ETHUSDT']+[f'ALT{i:02d}USDT' for i in range(18)]
    shared=rng.normal(0,.003,n)
    returns=shared[:,None]*np.linspace(.8,1.8,20)+rng.normal(0,.002,(n,20))
    close=100*np.exp(np.cumsum(returns,axis=0))
    quote=rng.uniform(100,10000,(n,20))
    d=pd.MultiIndex.from_product([times,names],names=['timestamp','symbol']).to_frame(index=False)
    d['close']=close.ravel();d['quote_volume']=quote.ravel()
    d['taker_buy_quote_volume']=(quote*rng.uniform(.03,.97,(n,20))).ravel()
    return d


class TimestampCompatibilityTests(unittest.TestCase):
    def test_same_weights_for_seconds_milliseconds_microseconds_nanoseconds(self):
        d=panel()
        reference=build_portfolios(d)
        self.assertGreater(reference['catchup'].abs().max(),0)
        for unit in ('s','ms','us','ns'):
            other=d.copy()
            other['timestamp']=other['timestamp'].dt.as_unit(unit)
            actual=build_portfolios(other)
            for name in reference:
                with self.subTest(unit=unit,portfolio=name):
                    assert_series_equal(reference[name],actual[name],atol=1e-12,rtol=0)

    def test_real_missing_hour_is_still_rejected(self):
        d=panel(12)
        d['timestamp']=d['timestamp'].dt.as_unit('ms')
        hour=d['timestamp'].iloc[5*20]
        # Maintainer change before the freeze: compute_weights now fills missing hours (src/contrib/janice_gaps.py),
        # so the check that a real missing hour is rejected runs on build_portfolios.
        with self.assertRaisesRegex(ValueError,'contiguous hourly'):
            build_portfolios(d.loc[d['timestamp']!=hour])

    def test_single_bar_prefix_is_valid_for_every_unit(self):
        for unit in ('s','ms','us','ns'):
            d=panel(1)
            d['timestamp']=d['timestamp'].dt.as_unit(unit)
            out=compute_weights(d)
            self.assertTrue(out.eq(0).all())

    def test_prefix_weights_agree_with_millisecond_input(self):
        d=panel()
        d['timestamp']=d['timestamp'].dt.as_unit('ms')
        full=build_portfolios(d)
        prefix=d.iloc[:-20]
        short=build_portfolios(prefix)
        for name in full:
            assert_series_equal(full[name].loc[prefix.index],short[name],atol=1e-12,rtol=0)


if __name__=='__main__':
    unittest.main()

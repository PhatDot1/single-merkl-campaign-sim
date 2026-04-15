"""
Tests for data loading, validation, and transformations.
"""

import numpy as np
import pandas as pd
import pytest

from bayesian_incentives.data.schema import validate_market_data
from bayesian_incentives.data.transforms import (
    compute_first_differences,
    compute_log_returns,
    compute_realized_volatility,
    compute_utilization,
)


@pytest.fixture
def valid_df():
    n = 100
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n),
        "tvl": np.random.default_rng(0).uniform(1e7, 1e8, n),
        "incentive_total": np.random.default_rng(0).uniform(100, 1000, n),
        "price": np.random.default_rng(0).uniform(2000, 4000, n),
    })


class TestSchema:

    def test_valid_data(self, valid_df):
        report = validate_market_data(valid_df)
        assert report.valid

    def test_missing_required(self):
        df = pd.DataFrame({"date": [1], "foo": [2]})
        report = validate_market_data(df)
        assert not report.valid
        assert "tvl" in report.missing_required

    def test_warnings_no_split(self, valid_df):
        report = validate_market_data(valid_df)
        assert any("supply/borrow" in w for w in report.warnings)

    def test_short_data_warning(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10),
            "tvl": range(10),
            "incentive_total": range(10),
        })
        report = validate_market_data(df)
        assert any("observations" in w.lower() for w in report.warnings)


class TestTransforms:

    def test_first_differences(self, valid_df):
        df = valid_df.set_index("date")
        out = compute_first_differences(df, columns=["tvl"])
        assert "d_tvl" in out.columns
        assert np.isnan(out["d_tvl"].iloc[0])
        assert len(out) == len(df)

    def test_log_returns(self, valid_df):
        df = valid_df.set_index("date")
        out = compute_log_returns(df, price_col="price")
        assert "log_return" in out.columns
        assert np.isnan(out["log_return"].iloc[0])

    def test_realized_volatility(self, valid_df):
        df = valid_df.set_index("date")
        df = compute_log_returns(df)
        out = compute_realized_volatility(df, window=7)
        assert "realized_vol" in out.columns
        # First 6 values should be NaN (window=7)
        assert out["realized_vol"].iloc[:6].isna().all()
        assert out["realized_vol"].iloc[7:].notna().all()

    def test_utilization(self):
        df = pd.DataFrame({
            "tvl": [1e8, 2e8],
            "borrow_volume": [5e7, 1.2e8],
        })
        out = compute_utilization(df)
        assert "utilization" in out.columns
        np.testing.assert_allclose(out["utilization"].values, [0.5, 0.6])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

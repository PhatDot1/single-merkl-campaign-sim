"""
Tests for grid search and Bayesian optimizer.
"""

import numpy as np
import pandas as pd
import pytest

from bayesian_incentives.models.elasticity import BayesianPosterior
from bayesian_incentives.models.interest_rate import InterestRateModel
from bayesian_incentives.optimization.grid_search import GridSearchOptimizer
from bayesian_incentives.optimization.bayesian import BayesianOptimizer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_market_df(rng):
    """Create a synthetic market DataFrame for optimizer tests."""
    n = 180
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # AR(1) TVL with rho=0.995
    tvl = np.zeros(n)
    tvl[0] = 1e8
    for t in range(1, n):
        tvl[t] = 5e5 + 0.995 * tvl[t - 1] + rng.normal(0, 2e5)

    inc_supply = 500 + rng.normal(0, 50, n).cumsum().clip(0)
    inc_borrow = 300 + rng.normal(0, 30, n).cumsum().clip(0)

    df = pd.DataFrame(
        {
            "tvl": tvl,
            "incentive_supply": inc_supply,
            "incentive_borrow": inc_borrow,
            "incentive_total": inc_supply + inc_borrow,
            "utilization": 0.6 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n)),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestGridSearchOptimizer:

    def test_basic_grid_search(self):
        """Grid search should return a result with correct shape."""
        opt = GridSearchOptimizer(n_mc=100, horizon=14)

        post_s = BayesianPosterior(mu=0, sigma=100)
        post_b = BayesianPosterior(mu=0, sigma=100)

        result = opt.optimize(
            current_tvl=1e8,
            current_utilization=0.7,
            current_i_supply=500,
            current_i_borrow=300,
            posterior_beta_s=post_s,
            posterior_beta_b=post_b,
            decay_alpha=0.5,
            decay_beta=49.5,
            equilibrium_tvl=1e8,
            grid_n=5,
        )

        assert result.grid_shape == (5, 5)
        assert len(result.grid) == 25
        assert result.optimal is not None

    def test_zero_beta_favors_zero_incentives(self):
        """With beta ~ 0, the optimizer should prefer zero incentives."""
        opt = GridSearchOptimizer(n_mc=500, horizon=30)

        # Very tight posteriors at zero -> incentives have no effect
        post_s = BayesianPosterior(mu=0, sigma=1)
        post_b = BayesianPosterior(mu=0, sigma=1)

        result = opt.optimize(
            current_tvl=1e8,
            current_utilization=0.7,
            current_i_supply=500,
            current_i_borrow=300,
            posterior_beta_s=post_s,
            posterior_beta_b=post_b,
            decay_alpha=1,
            decay_beta=99,  # mode ~ 0, low decay
            equilibrium_tvl=1e8,
            grid_n=10,
            supply_range=(0, 1000),
            borrow_range=(0, 600),
        )

        # Optimal should be near zero
        assert result.optimal.i_supply < 200
        assert result.optimal.i_borrow < 200

    def test_profit_matrix_shape(self):
        opt = GridSearchOptimizer(n_mc=50, horizon=7)
        post = BayesianPosterior(mu=0, sigma=10)

        result = opt.optimize(
            current_tvl=1e8,
            current_utilization=0.7,
            current_i_supply=100,
            current_i_borrow=100,
            posterior_beta_s=post,
            posterior_beta_b=post,
            decay_alpha=1,
            decay_beta=99,
            equilibrium_tvl=1e8,
            grid_n=8,
        )

        s, b, p = result.profit_matrix()
        assert p.shape == (8, 8)


class TestBayesianOptimizer:

    def test_single_iteration(self, synthetic_market_df):
        opt = BayesianOptimizer(n_mc=100, horizon=14, grid_n=5)
        result = opt.run_iteration(synthetic_market_df)

        assert result.iteration == 1
        assert result.ols_supply is not None
        assert result.decay is not None
        assert result.grid_result is not None

    def test_total_only_fallback(self, synthetic_market_df):
        """Should work with only incentive_total (no supply/borrow split)."""
        df = synthetic_market_df.drop(columns=["incentive_supply", "incentive_borrow"])
        opt = BayesianOptimizer(n_mc=100, horizon=14, grid_n=5)
        result = opt.run_iteration(
            df,
            incentive_supply_col=None,
            incentive_borrow_col=None,
        )
        assert result.iteration == 1

    def test_recommendation(self, synthetic_market_df):
        opt = BayesianOptimizer(n_mc=100, horizon=14, grid_n=5)
        opt.run_iteration(synthetic_market_df)
        rec = opt.recommendation()

        assert "assessment" in rec
        assert "optimal_supply" in rec
        assert "optimal_borrow" in rec
        assert rec["iteration"] == 1

    def test_backtest(self, synthetic_market_df):
        opt = BayesianOptimizer(n_mc=50, horizon=7, grid_n=3)
        results = opt.run_backtest(
            synthetic_market_df,
            window_size=60,
            step_size=30,
        )
        assert len(results) >= 3  # 180 days, 60 window, 30 step

    def test_history_accumulates(self, synthetic_market_df):
        opt = BayesianOptimizer(n_mc=50, horizon=7, grid_n=3)
        opt.run_iteration(synthetic_market_df.iloc[:90])
        opt.run_iteration(synthetic_market_df.iloc[:120])
        assert len(opt.history) == 2
        assert opt.history[1].iteration == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

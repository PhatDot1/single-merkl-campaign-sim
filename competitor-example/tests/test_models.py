"""
Tests for core model components.

Uses synthetic data to validate statistical correctness of:
  - OLS elasticity estimation
  - Bayesian posterior computation
  - AR(1) decay estimation
  - Chi-square independence test
  - Regime classification
"""

import numpy as np
import pytest

from bayesian_incentives.models.elasticity import ElasticityModel
from bayesian_incentives.models.decay import DecayModel
from bayesian_incentives.models.independence import IndependenceTest
from bayesian_incentives.models.interest_rate import InterestRateModel, RateParams
from bayesian_incentives.models.regimes import RegimeAnalyzer


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_elastic_data(rng):
    """Generate data where incentives DO affect TVL (beta=500)."""
    n = 300
    true_beta = 500.0
    d_inc = rng.normal(0, 50, n)
    noise = rng.normal(0, 1e6, n)
    d_tvl = 1000 + true_beta * d_inc + noise
    return d_inc, d_tvl, true_beta


@pytest.fixture
def synthetic_inelastic_data(rng):
    """Generate data where incentives have NO effect on TVL."""
    n = 300
    d_inc = rng.normal(0, 50, n)
    d_tvl = rng.normal(0, 1e6, n)  # pure noise
    return d_inc, d_tvl


@pytest.fixture
def synthetic_ar1_series(rng):
    """Generate AR(1) series with known parameters."""
    rho = 0.99
    c = 5e6 * (1 - rho)  # equilibrium = 5M
    n = 1000
    s = np.zeros(n)
    s[0] = 5e6
    for t in range(1, n):
        s[t] = c + rho * s[t - 1] + rng.normal(0, 20000)
    return s, rho, c


# ── Elasticity tests ───────────────────────────────────────────────────

class TestElasticityModel:

    def test_ols_recovers_beta(self, synthetic_elastic_data):
        d_inc, d_tvl, true_beta = synthetic_elastic_data
        model = ElasticityModel()
        result = model.fit_ols(d_inc, d_tvl)

        # Should be within 2 SE of true value
        assert abs(result.beta - true_beta) < 3 * result.se_beta
        assert result.n_obs == 300

    def test_inelastic_low_r2(self, synthetic_inelastic_data):
        d_inc, d_tvl = synthetic_inelastic_data
        model = ElasticityModel()
        result = model.fit_ols(d_inc, d_tvl)

        assert result.r_squared < 0.05  # Should be near zero
        assert result.p_value > 0.05  # Not significant

    def test_posterior_converges_to_ols(self, synthetic_elastic_data):
        d_inc, d_tvl, _ = synthetic_elastic_data
        model = ElasticityModel(prior_mu=0, prior_sigma=1e6)  # diffuse
        ols = model.fit_ols(d_inc, d_tvl)
        post = model.fit_posterior(d_inc, d_tvl)

        # With diffuse prior, posterior mean ~ OLS estimate
        assert abs(post.mu - ols.beta) / abs(ols.beta) < 0.01

    def test_informative_prior_shrinks(self, synthetic_elastic_data):
        d_inc, d_tvl, _ = synthetic_elastic_data
        # Very tight prior at zero
        model = ElasticityModel(prior_mu=0, prior_sigma=10)
        post = model.fit_posterior(d_inc, d_tvl)

        # Posterior should be pulled toward zero relative to OLS
        model2 = ElasticityModel(prior_mu=0, prior_sigma=1e6)
        model2.fit_ols(d_inc, d_tvl)
        assert abs(post.mu) < abs(model2.ols.beta)

    def test_lagged_returns_dict(self, synthetic_inelastic_data):
        d_inc, d_tvl = synthetic_inelastic_data
        model = ElasticityModel()
        # Need level series for lagged
        tvl = np.cumsum(d_tvl) + 1e8
        lagged = model.fit_lagged(d_inc, tvl, lags=[1, 7])
        assert 1 in lagged
        assert 7 in lagged

    def test_minimum_observations(self):
        model = ElasticityModel()
        with pytest.raises(ValueError, match="Need >= 3"):
            model.fit_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# ── Decay tests ─────────────────────────────────────────────────────────

class TestDecayModel:

    def test_recovers_rho(self, synthetic_ar1_series):
        s, true_rho, _ = synthetic_ar1_series
        model = DecayModel()
        result = model.fit(s)

        assert abs(result.rho - true_rho) < 0.02

    def test_half_life_positive(self, synthetic_ar1_series):
        s, _, _ = synthetic_ar1_series
        model = DecayModel()
        result = model.fit(s)

        assert result.half_life is not None
        assert result.half_life > 0

    def test_equilibrium_near_true(self, synthetic_ar1_series):
        s, _, _ = synthetic_ar1_series
        model = DecayModel()
        result = model.fit(s)

        assert result.equilibrium is not None
        assert abs(result.equilibrium - 5e6) / 5e6 < 0.10  # within 10%

    def test_impulse_response_decays(self, synthetic_ar1_series):
        s, _, _ = synthetic_ar1_series
        model = DecayModel()
        result = model.fit(s)
        ir = result.impulse_response(60)

        assert ir[0] == 1.0
        assert ir[-1] < ir[0]
        assert all(np.diff(ir) <= 0)  # monotonically decreasing

    def test_decay_prior_params(self, synthetic_ar1_series):
        s, _, _ = synthetic_ar1_series
        model = DecayModel()
        model.fit(s)
        alpha, beta = model.decay_prior_params()
        assert alpha > 0
        assert beta > 0


# ── Independence tests ──────────────────────────────────────────────────

class TestIndependenceTest:

    def test_independent_data(self, rng):
        """Two independent series should fail to reject H0."""
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        test = IndependenceTest()
        result = test.test(x, y)

        assert result.independent  # p > 0.05
        assert result.df == 1

    def test_dependent_data(self, rng):
        """Perfectly correlated should reject H0."""
        x = rng.normal(0, 1, 500)
        y = x + rng.normal(0, 0.01, 500)  # nearly identical
        test = IndependenceTest()
        result = test.test(x, y)

        assert not result.independent  # p < 0.05

    def test_contingency_sums(self, rng):
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        test = IndependenceTest()
        result = test.test(x, y)

        assert result.contingency.sum() == np.sum(np.isfinite(x) & np.isfinite(y))

    def test_counter_examples(self, rng):
        inc = np.cumsum(rng.normal(0, 10, 100)) + 1000
        tvl = np.cumsum(rng.normal(0, 1e5, 100)) + 1e8
        test = IndependenceTest()
        ce = test.test_counter_examples(inc, tvl, inc_threshold=0.01, tvl_horizon=5)
        assert "n_increases" in ce
        assert "n_counter_examples" in ce


# ── Interest rate model tests ───────────────────────────────────────────

class TestInterestRateModel:

    def test_below_kink(self):
        model = InterestRateModel()
        r_b = model.borrow_rate(0.5)
        # At 50% util with default params: r_base + (0.5/0.8)*0.05
        expected = 0.0 + (0.5 / 0.8) * 0.05
        assert abs(float(r_b) - expected) < 1e-10

    def test_above_kink(self):
        model = InterestRateModel()
        r_b = model.borrow_rate(0.9)
        # Above kink at 90% with u_optimal=0.8
        expected = 0.0 + 0.05 + ((0.9 - 0.8) / (1 - 0.8)) * 3.0
        assert abs(float(r_b) - expected) < 1e-10

    def test_supply_rate(self):
        model = InterestRateModel()
        r_s = model.supply_rate(0.5)
        r_b = model.borrow_rate(0.5)
        expected = float(r_b) * 0.5 * (1 - 0.10)
        assert abs(float(r_s) - expected) < 1e-10

    def test_nii_positive_at_high_util(self):
        model = InterestRateModel()
        nii = model.net_interest_income(tvl=1e8, utilization=0.8)
        assert nii > 0  # Protocol should profit at high util

    def test_vectorized(self):
        model = InterestRateModel()
        u = np.array([0.3, 0.5, 0.8, 0.9, 0.95])
        r_b = model.borrow_rate(u)
        assert r_b.shape == (5,)
        assert all(np.diff(r_b) > 0)  # Rates increase with utilization


# ── Regime analyzer tests ───────────────────────────────────────────────

class TestRegimeAnalyzer:

    def test_volatility_classification(self, rng):
        vol = rng.exponential(0.3, 500)
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_volatility(vol)

        assert regimes["high_vol"].sum() == pytest.approx(125, abs=20)  # ~25%
        assert regimes["extreme_vol"].sum() == pytest.approx(50, abs=15)  # ~10%

    def test_utilization_classification(self):
        u = np.linspace(0, 1, 100)
        analyzer = RegimeAnalyzer()
        regimes = analyzer.classify_utilization(u)

        assert regimes["high_util"].sum() == 15  # u > 0.85: indices 86-100
        assert regimes["extreme_util"].sum() == 5  # u > 0.95: indices 96-100

    def test_conditional_elasticity(self, rng):
        n = 200
        x = rng.normal(0, 50, n)
        y = rng.normal(0, 1e6, n)
        mask = np.zeros(n, dtype=bool)
        mask[:50] = True

        analyzer = RegimeAnalyzer()
        result = analyzer.fit_conditional_elasticity(x, y, mask, "test")
        assert result.n_in == 50
        assert result.n_out == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

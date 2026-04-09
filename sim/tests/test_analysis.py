"""
Tests for sim/campaign/analysis.py and sim/campaign/backtest.py.

Tests cover all major model components:
  - ElasticityModel (OLS, posterior, lagged)
  - DecayModel (AR(1), half-life, equilibrium, prior_params)
  - IndependenceTest (chi-square, counter-examples)
  - RegimeAnalyzer (conditional elasticity, bootstrap CI)
  - ProfitabilityAnalyzer (NII, breakeven, MarketStatus)
  - generate_report() (pipeline integration)
  - run_rolling_backtest() (rolling window)
"""

from __future__ import annotations

import numpy as np
import pytest

from campaign.analysis import (
    AnalysisReport,
    BayesianPosterior,
    DecayModel,
    DecayResult,
    ElasticityModel,
    IndependenceTest,
    MarketStatus,
    OLSResult,
    ProfitabilityAnalyzer,
    RegimeAnalyzer,
    generate_report,
)
from campaign.backtest import BacktestResult, BacktestWindow, run_rolling_backtest
from campaign.historical import PoolHistory, PoolHistoryPoint
from campaign.state import IRMParams


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def elastic_data(rng):
    """Synthetic data where incentives DO affect TVL: beta ≈ 500."""
    n = 300
    true_beta = 500.0
    d_inc = rng.normal(0, 50, n)
    # Noise std (5_000) is small relative to signal std (500*50=25_000)
    # so OLS is well-identified and t-stat >> 2.
    noise = rng.normal(0, 5_000, n)
    d_tvl = 1000.0 + true_beta * d_inc + noise
    return d_inc, d_tvl, true_beta


@pytest.fixture
def inelastic_data(rng):
    """Synthetic data where incentives have NO effect on TVL."""
    n = 300
    d_inc = rng.normal(0, 50, n)
    d_tvl = rng.normal(0, 1e6, n)
    return d_inc, d_tvl


@pytest.fixture
def ar1_series(rng):
    """AR(1) series: rho=0.99, equilibrium=5M."""
    rho = 0.99
    eq = 5_000_000.0
    c = eq * (1 - rho)
    n = 1000
    s = np.zeros(n)
    s[0] = eq
    for t in range(1, n):
        s[t] = c + rho * s[t - 1] + rng.normal(0, 20_000)
    return s, rho, eq


@pytest.fixture
def sample_pool_history(rng):
    """Synthetic PoolHistory with 120 daily data points."""
    n = 120
    rho, eq, inc_mean = 0.995, 50_000_000.0, 5_000.0
    c = eq * (1 - rho)
    tvl = np.zeros(n)
    tvl[0] = eq
    for t in range(1, n):
        tvl[t] = c + rho * tvl[t - 1] + rng.normal(0, eq * 0.004)

    reward_apy = np.maximum(inc_mean / tvl * 365.0, 0.0)
    base_apy = 0.03 * np.ones(n) + rng.normal(0, 0.002, n)

    points = [
        PoolHistoryPoint(
            timestamp=f"2025-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
            tvl_usd=float(tvl[i]),
            apy=float(base_apy[i] + reward_apy[i]),
            apy_base=float(base_apy[i]),
            apy_reward=float(reward_apy[i]),
            il7d=0.0,
        )
        for i in range(n)
    ]

    return PoolHistory(
        pool_id="test-pool",
        project="test-protocol",
        symbol="USDC",
        chain="Ethereum",
        points=points,
    )


# ============================================================================
# ELASTICITY MODEL
# ============================================================================


class TestElasticityModel:
    def test_ols_recovers_beta(self, elastic_data):
        d_inc, d_tvl, true_beta = elastic_data
        m = ElasticityModel()
        r = m.fit_ols(d_inc, d_tvl)

        assert abs(r.beta - true_beta) < 3 * r.se_beta
        assert r.n_obs == 300
        assert isinstance(r, OLSResult)

    def test_ols_significant_elastic(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        r = ElasticityModel().fit_ols(d_inc, d_tvl)
        assert r.significant

    def test_inelastic_low_r2_not_significant(self, inelastic_data):
        d_inc, d_tvl = inelastic_data
        r = ElasticityModel().fit_ols(d_inc, d_tvl)
        assert r.r_squared < 0.05
        assert r.p_value > 0.05, f"Expected p>0.05, got {r.p_value}"

    def test_ols_raises_for_too_few_obs(self):
        m = ElasticityModel()
        with pytest.raises(ValueError, match="3"):
            m.fit_ols(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_posterior_diffuse_converges_to_ols(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        m = ElasticityModel(prior_mu=0.0, prior_sigma=1e6)
        ols = m.fit_ols(d_inc, d_tvl)
        post = m.fit_posterior(d_inc, d_tvl)
        # With extremely diffuse prior, posterior mean ≈ OLS beta
        assert abs(post.mu - ols.beta) / max(abs(ols.beta), 1.0) < 0.01

    def test_posterior_tight_prior_shrinks_toward_zero(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        m_diffuse = ElasticityModel(prior_mu=0.0, prior_sigma=1e6)
        m_tight = ElasticityModel(prior_mu=0.0, prior_sigma=10)
        m_diffuse.fit_ols(d_inc, d_tvl)
        m_tight.fit_ols(d_inc, d_tvl)
        post_diffuse = m_diffuse.fit_posterior(d_inc, d_tvl)
        post_tight = m_tight.fit_posterior(d_inc, d_tvl)
        # Tight prior at zero → posterior mean closer to 0
        assert abs(post_tight.mu) < abs(post_diffuse.mu)

    def test_posterior_ci_95_contains_true(self, elastic_data):
        d_inc, d_tvl, true_beta = elastic_data
        m = ElasticityModel(prior_mu=0.0, prior_sigma=1e6)
        m.fit_ols(d_inc, d_tvl)
        post = m.fit_posterior(d_inc, d_tvl)
        lo, hi = post.ci_95
        assert lo < true_beta < hi

    def test_posterior_sample(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        m = ElasticityModel()
        m.fit_ols(d_inc, d_tvl)
        post = m.fit_posterior(d_inc, d_tvl)
        samples = post.sample(1000)
        assert len(samples) == 1000
        assert abs(np.mean(samples) - post.mu) < 3 * post.sigma / np.sqrt(1000)

    def test_lagged_returns_dict(self, inelastic_data):
        d_inc, d_tvl = inelastic_data
        tvl = np.cumsum(d_tvl) + 1e8
        m = ElasticityModel()
        lagged = m.fit_lagged(d_inc, tvl, lags=[1, 7, 14])
        assert 1 in lagged
        assert 7 in lagged
        assert all(isinstance(v, OLSResult) for v in lagged.values())

    def test_lagged_skips_if_insufficient(self):
        d_inc = np.array([1.0, 2.0, 3.0])
        tvl = np.array([10.0, 11.0, 12.0, 13.0])
        m = ElasticityModel()
        lagged = m.fit_lagged(d_inc, tvl, lags=[50, 100])
        assert len(lagged) == 0  # All lags exceed data length

    def test_ols_summary_string(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        r = ElasticityModel().fit_ols(d_inc, d_tvl)
        s = r.summary()
        assert "beta" in s
        assert "R²" in s


# ============================================================================
# DECAY MODEL
# ============================================================================


class TestDecayModel:
    def test_recovers_rho(self, ar1_series):
        s, true_rho, _ = ar1_series
        dm = DecayModel()
        r = dm.fit(s)
        assert abs(r.rho - true_rho) < 0.02

    def test_half_life_positive(self, ar1_series):
        s, _, _ = ar1_series
        r = DecayModel().fit(s)
        assert r.half_life is not None
        assert r.half_life > 0

    def test_equilibrium_near_true(self, ar1_series):
        s, _, true_eq = ar1_series
        r = DecayModel().fit(s)
        assert r.equilibrium is not None
        assert abs(r.equilibrium - true_eq) / true_eq < 0.10

    def test_decay_rate_equals_one_minus_rho(self, ar1_series):
        s, _, _ = ar1_series
        r = DecayModel().fit(s)
        assert abs(r.decay_rate - (1 - r.rho)) < 1e-10

    def test_impulse_response_decays_monotonically(self, ar1_series):
        s, _, _ = ar1_series
        r = DecayModel().fit(s)
        ir = r.impulse_response(60)
        assert ir[0] == pytest.approx(1.0)
        assert float(ir[-1]) < float(ir[0])
        diffs = np.diff(ir)
        assert all(d <= 0 for d in diffs), "Impulse response should be monotonically decreasing"

    def test_prior_params_shape(self, ar1_series):
        s, _, _ = ar1_series
        dm = DecayModel()
        dm.fit(s)
        alpha, beta = dm.decay_prior_params()
        assert alpha > 0
        assert beta > 0

    def test_fit_raises_for_too_few_obs(self):
        dm = DecayModel()
        with pytest.raises(ValueError, match="4"):
            dm.fit(np.array([1.0, 2.0, 3.0]))

    def test_decay_summary_string(self, ar1_series):
        s, _, _ = ar1_series
        r = DecayModel().fit(s)
        summary = r.summary()
        assert "rho" in summary
        assert "half-life" in summary
        assert "equilibrium" in summary


# ============================================================================
# INDEPENDENCE TEST
# ============================================================================


class TestIndependenceTest:
    def test_independent_data_fails_to_reject(self, rng):
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        result = IndependenceTest().test(x, y)
        assert result.independent
        assert result.df == 1

    def test_perfectly_correlated_rejects(self, rng):
        x = rng.normal(0, 1, 500)
        y = x + rng.normal(0, 0.001, 500)
        result = IndependenceTest().test(x, y)
        assert not result.independent

    def test_contingency_sums_to_n(self, rng):
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = IndependenceTest().test(x, y)
        assert result.contingency.sum() == 200

    def test_contingency_shape(self, rng):
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = IndependenceTest().test(x, y)
        assert result.contingency.shape == (2, 2)

    def test_chi2_non_negative(self, rng):
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        result = IndependenceTest().test(x, y)
        assert result.chi2 >= 0

    def test_counter_examples_keys(self, rng):
        inc = np.cumsum(rng.normal(0, 10, 100)) + 1000
        tvl = np.cumsum(rng.normal(0, 1e5, 100)) + 1e8
        ce = IndependenceTest().counter_examples(inc, tvl, inc_threshold=0.05, tvl_horizon=5)
        assert "n_increases" in ce
        assert "n_counter_examples" in ce
        assert "p_counter" in ce
        assert "p_decline_given_decrease" in ce

    def test_counter_examples_fraction_in_range(self, rng):
        inc = np.cumsum(rng.normal(0, 10, 100)) + 1000
        tvl = np.cumsum(rng.normal(0, 1e5, 100)) + 1e8
        ce = IndependenceTest().counter_examples(inc, tvl)
        if ce["p_counter"] is not None:
            assert 0.0 <= ce["p_counter"] <= 1.0

    def test_summary_string(self, rng):
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        result = IndependenceTest().test(x, y)
        s = result.summary()
        assert "chi²" in s or "chi2" in s.lower() or "Chi" in s


# ============================================================================
# REGIME ANALYZER
# ============================================================================


class TestRegimeAnalyzer:
    def test_classify_volatility_returns_correct_keys(self, ar1_series):
        s, _, _ = ar1_series
        ra = RegimeAnalyzer()
        info = ra.classify_volatility(s)
        assert "high_vol" in info
        assert "extreme_vol" in info
        assert "vol" in info

    def test_classify_volatility_lengths(self, ar1_series):
        s, _, _ = ar1_series
        ra = RegimeAnalyzer()
        info = ra.classify_volatility(s)
        # Should be len(s) - 1 (aligned to first differences)
        assert len(info["high_vol"]) == len(s) - 1
        assert len(info["extreme_vol"]) == len(s) - 1

    def test_high_vol_fraction_near_quantile(self, ar1_series):
        s, _, _ = ar1_series
        ra = RegimeAnalyzer(vol_high_q=0.75)
        info = ra.classify_volatility(s)
        frac = info["high_vol"].mean()
        # Should be roughly 25% in high-vol regime (100% - 75th quantile)
        assert 0.15 < frac < 0.40

    def test_fit_conditional_returns_result(self, elastic_data):
        d_inc, d_tvl, _ = elastic_data
        tvl_level = np.cumsum(d_tvl) + 1e8
        ra = RegimeAnalyzer(bootstrap_n=50)
        info = ra.classify_volatility(tvl_level)
        mask = info["high_vol"][: len(d_inc)]
        result = ra.fit_conditional(d_inc, d_tvl, mask, "Test Regime")
        assert result.regime_name == "Test Regime"
        assert result.n_in + result.n_out <= len(d_inc)

    def test_fit_conditional_insufficient_data(self):
        d_inc = np.ones(5)
        d_tvl = np.ones(5)
        mask = np.array([True, False, True, False, True])
        ra = RegimeAnalyzer(bootstrap_n=10)
        result = ra.fit_conditional(d_inc, d_tvl, mask, "Regime")
        # Should not raise; ols_in/ols_out may be None
        assert result is not None


# ============================================================================
# PROFITABILITY ANALYZER
# ============================================================================


class TestProfitabilityAnalyzer:
    @pytest.fixture
    def default_irm(self):
        return IRMParams(
            optimal_util=0.80,
            base_rate=0.0,
            slope1=0.04,
            slope2=0.60,
            reserve_factor=0.10,
            initial_borrows_usd=0.0,
        )

    def test_well_utilized_classification(self, default_irm):
        pa = ProfitabilityAnalyzer(irm=default_irm)
        snap = pa.snapshot(tvl=100e6, utilization=0.90, daily_incentives=0.0)
        assert snap.status == MarketStatus.WELL_UTILIZED

    def test_under_utilized_classification(self, default_irm):
        # IRM mode always returns AT_BREAKEVEN/WELL_UTILIZED because NII = TVL*util*r_b*RF > 0.
        # Use direct mode: supply_apr much larger than util*borrow_apr to test UNDER_UTILIZED.
        pa = ProfitabilityAnalyzer()  # no IRM
        # breakeven_u = supply_apr / borrow_apr = 0.04/0.05 = 0.80
        # util=0.10 < 0.80 - 0.05 (band) = 0.75  →  UNDER_UTILIZED
        snap = pa.snapshot(
            tvl=100e6, utilization=0.10, daily_incentives=0.0,
            borrow_apr=0.05, supply_apr=0.04,
        )
        assert snap.status == MarketStatus.UNDER_UTILIZED

    def test_nii_positive_above_breakeven(self, default_irm):
        pa = ProfitabilityAnalyzer(irm=default_irm)
        snap = pa.snapshot(tvl=100e6, utilization=0.85, daily_incentives=0.0)
        assert snap.daily_nii > 0

    def test_nii_negative_below_breakeven(self, default_irm):
        # In direct mode with supply_apr >> util*borrow_apr the protocol runs at a loss.
        # NII = (100e6*0.10*0.05 - 100e6*0.04)/365 = (500_000 - 4_000_000)/365 < 0
        pa = ProfitabilityAnalyzer()  # no IRM
        snap = pa.snapshot(
            tvl=100e6, utilization=0.10, daily_incentives=0.0,
            borrow_apr=0.05, supply_apr=0.04,
        )
        assert snap.daily_nii < 0

    def test_daily_profit_includes_incentives(self, default_irm):
        pa = ProfitabilityAnalyzer(irm=default_irm)
        snap_no_inc = pa.snapshot(tvl=100e6, utilization=0.80, daily_incentives=0.0)
        snap_with_inc = pa.snapshot(tvl=100e6, utilization=0.80, daily_incentives=10_000.0)
        assert snap_no_inc.daily_profit - snap_with_inc.daily_profit == pytest.approx(10_000.0)

    def test_direct_apr_mode(self):
        pa = ProfitabilityAnalyzer()  # No IRM
        snap = pa.snapshot(
            tvl=100e6,
            utilization=0.70,
            daily_incentives=5_000.0,
            borrow_apr=0.06,
            supply_apr=0.035,
        )
        assert snap.borrow_apr == pytest.approx(0.06)
        assert snap.supply_apr == pytest.approx(0.035)

    def test_missing_apr_without_irm_raises(self):
        pa = ProfitabilityAnalyzer()
        with pytest.raises(ValueError, match="borrow_apr"):
            pa.snapshot(tvl=100e6, utilization=0.7, daily_incentives=0.0)

    def test_breakeven_util_property(self, default_irm):
        pa = ProfitabilityAnalyzer(irm=default_irm)
        snap = pa.snapshot(tvl=100e6, utilization=0.70, daily_incentives=0.0)
        # Verify NII ≈ 0 at breakeven_util
        snap_at_be = pa.snapshot(
            tvl=100e6,
            utilization=snap.breakeven_utilization,
            daily_incentives=0.0,
        )
        assert abs(snap_at_be.daily_nii) < 1_000  # within $1k/day

    def test_market_status_labels(self):
        assert MarketStatus.WELL_UTILIZED.label == "Well-Utilized"
        assert MarketStatus.AT_BREAKEVEN.label == "At Breakeven"
        assert MarketStatus.UNDER_UTILIZED.label == "Under-Utilized"

    def test_market_status_colors(self):
        assert "🟢" in MarketStatus.WELL_UTILIZED.color
        assert "🟡" in MarketStatus.AT_BREAKEVEN.color
        assert "🔴" in MarketStatus.UNDER_UTILIZED.color

    def test_snapshot_summary_string(self, default_irm):
        pa = ProfitabilityAnalyzer(irm=default_irm)
        snap = pa.snapshot(tvl=100e6, utilization=0.75, daily_incentives=5000.0)
        s = snap.summary()
        assert "TVL" in s
        assert "NII" in s


# ============================================================================
# GENERATE REPORT
# ============================================================================


class TestGenerateReport:
    def test_returns_analysis_report(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert isinstance(report, AnalysisReport)

    def test_report_has_observations(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.n_observations == 120

    def test_report_has_date_range(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.date_range[0] != ""
        assert report.date_range[1] != ""

    def test_report_has_elasticity(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.elasticity_total is not None

    def test_report_has_decay(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.decay is not None
        assert report.decay.rho is not None

    def test_report_has_independence(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.independence is not None

    def test_report_has_counter_examples(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        assert report.counter_examples is not None

    def test_report_recommended_horizon_floors_at_28(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        horizon = report.recommended_horizon_days()
        if horizon is not None:
            assert horizon >= 28

    def test_organic_floor_fraction_in_range(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        current_tvl = sample_pool_history.tvl_array[-1]
        frac = report.organic_floor_fraction(current_tvl)
        if frac is not None:
            assert 0.0 <= frac <= 1.0

    def test_report_with_irm_has_profitability(self, sample_pool_history):
        irm = IRMParams(
            optimal_util=0.80, base_rate=0.0, slope1=0.04, slope2=0.60,
            reserve_factor=0.10, initial_borrows_usd=0.0,
        )
        report = generate_report(sample_pool_history, irm=irm, current_utilization=0.75)
        assert report.profitability_current is not None

    def test_report_summary_string(self, sample_pool_history):
        report = generate_report(sample_pool_history)
        s = report.summary()
        assert "ANALYSIS REPORT" in s
        assert "Elasticity" in s
        assert "Decay" in s

    def test_too_few_points_returns_empty(self):
        pts = [
            PoolHistoryPoint("2025-01-01", 1e8, 0.05, 0.03, 0.02, 0.0),
            PoolHistoryPoint("2025-01-02", 1e8, 0.05, 0.03, 0.02, 0.0),
        ]
        history = PoolHistory("p", "proj", "SYM", "ETH", pts)
        report = generate_report(history)
        assert report.elasticity_total is None
        assert report.decay is None


# ============================================================================
# ROLLING BACKTEST
# ============================================================================


class TestRollingBacktest:
    def test_correct_window_count(self, sample_pool_history):
        result = run_rolling_backtest(
            sample_pool_history, window_size=60, step_size=20
        )
        # (120 - 60) / 20 + 1 = 4 windows
        assert len(result.windows) == 4

    def test_single_window_large_step(self, sample_pool_history):
        result = run_rolling_backtest(
            sample_pool_history, window_size=100, step_size=200
        )
        assert len(result.windows) == 1

    def test_no_windows_when_history_too_short(self):
        pts = [
            PoolHistoryPoint(f"2025-01-{i+1:02d}", 1e8, 0.05, 0.03, 0.02, 0.0)
            for i in range(10)
        ]
        history = PoolHistory("p", "", "", "", pts)
        result = run_rolling_backtest(history, window_size=90, step_size=30)
        assert len(result.windows) == 0

    def test_windows_have_reports(self, sample_pool_history):
        result = run_rolling_backtest(sample_pool_history, window_size=60, step_size=30)
        for w in result.windows:
            assert isinstance(w.report, AnalysisReport)
            assert w.n_observations == 60

    def test_window_dates_are_ordered(self, sample_pool_history):
        result = run_rolling_backtest(sample_pool_history, window_size=60, step_size=20)
        dates = [w.window_start for w in result.windows]
        assert dates == sorted(dates)

    def test_elasticity_betas_list(self, sample_pool_history):
        result = run_rolling_backtest(sample_pool_history, window_size=60, step_size=20)
        betas = result.elasticity_betas
        assert isinstance(betas, list)

    def test_half_lives_list(self, sample_pool_history):
        result = run_rolling_backtest(sample_pool_history, window_size=60, step_size=20)
        hls = result.half_lives
        assert isinstance(hls, list)
        for hl in hls:
            assert hl > 0

    def test_parameter_stability_keys(self, sample_pool_history):
        result = run_rolling_backtest(sample_pool_history, window_size=60, step_size=20)
        ps = result.parameter_stability()
        assert isinstance(ps, dict)

    def test_summary_string(self, sample_pool_history):
        result = run_rolling_backtest(
            sample_pool_history, window_size=60, step_size=20, bootstrap_n=10
        )
        s = result.summary()
        assert "Rolling backtest" in s
        assert "window" in s.lower()

    def test_backtest_result_fields(self, sample_pool_history):
        result = run_rolling_backtest(
            sample_pool_history, window_size=60, step_size=30, bootstrap_n=10
        )
        assert result.pool_id == "test-pool"
        assert result.window_size == 60
        assert result.step_size == 30
        assert isinstance(result.windows, list)

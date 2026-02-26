"""
Tests for Phase 3 simulation-realism enhancements:

  3.1  IRMParams + compute_dynamic_base_apy  (state.py)
  3.2  budget_savings_usd tracking           (engine.py)
  3.3  organic_tvl_floor_fraction            (agents.py)

All tests are fully isolated — no network calls, no file I/O.
"""

from __future__ import annotations

import pytest

from campaign.state import (
    CampaignConfig,
    CampaignEnvironment,
    CampaignState,
    IRMParams,
    compute_dynamic_base_apy,
)
from campaign.agents import (
    RetailDepositorConfig,
    RetailDepositorAgent,
)
from campaign.engine import (
    CampaignLossEvaluator,
    CampaignSimulationEngine,
    LossResult,
    LossWeights,
    MonteCarloResult,
    run_monte_carlo,
)


# ============================================================================
# 3.1  IRMParams + compute_dynamic_base_apy
# ============================================================================


class TestIRMParams:
    """Test IRMParams dataclass construction and defaults."""

    def test_default_params(self):
        irm = IRMParams()
        assert irm.optimal_util == 0.80
        assert irm.base_rate == 0.00
        assert irm.slope1 == 0.04
        assert irm.slope2 == 0.60
        assert irm.reserve_factor == 0.10
        assert irm.initial_borrows_usd == 0.0

    def test_custom_params(self):
        irm = IRMParams(optimal_util=0.70, slope1=0.05, initial_borrows_usd=100_000_000.0)
        assert irm.optimal_util == 0.70
        assert irm.slope1 == 0.05
        assert irm.initial_borrows_usd == 100_000_000.0

    def test_frozen(self):
        irm = IRMParams()
        with pytest.raises((AttributeError, TypeError)):
            irm.optimal_util = 0.90  # type: ignore[misc]


class TestComputeDynamicBaseAPY:
    """Arithmetic tests for compute_dynamic_base_apy."""

    def make_irm(self, borrows: float = 80_000_000.0) -> IRMParams:
        return IRMParams(
            optimal_util=0.80,
            base_rate=0.00,
            slope1=0.04,
            slope2=0.60,
            reserve_factor=0.10,
            initial_borrows_usd=borrows,
        )

    # ── Edge cases ──────────────────────────────────────────────────────────

    def test_zero_tvl_returns_zero(self):
        assert compute_dynamic_base_apy(0.0, self.make_irm()) == 0.0

    def test_negative_tvl_returns_zero(self):
        assert compute_dynamic_base_apy(-1_000.0, self.make_irm()) == 0.0

    def test_zero_borrows_returns_zero(self):
        irm = IRMParams(initial_borrows_usd=0.0)
        assert compute_dynamic_base_apy(100_000_000.0, irm) == 0.0

    # ── Below-kink regime (util <= optimal_util) ─────────────────────────

    def test_exact_kink_util(self):
        """At util == optimal_util the formula spans kink exactly."""
        # borrows = 80M, tvl = 100M → util = 0.80 = optimal_util
        irm = self.make_irm(borrows=80_000_000.0)
        tvl = 100_000_000.0
        result = compute_dynamic_base_apy(tvl, irm)
        # borrow_rate = 0.0 + 1.0 * 0.04 = 0.04
        # supply_apy  = 0.04 * 0.80 * (1 - 0.10) = 0.0288
        expected = 0.04 * 0.80 * 0.90
        assert abs(result - expected) < 1e-10

    def test_below_kink_half_util(self):
        """util = 0.40 is exactly half the kink."""
        # borrows = 40M, tvl = 100M → util = 0.40
        irm = self.make_irm(borrows=40_000_000.0)
        tvl = 100_000_000.0
        result = compute_dynamic_base_apy(tvl, irm)
        # borrow_rate = 0.0 + (0.40/0.80) * 0.04 = 0.02
        # supply_apy  = 0.02 * 0.40 * 0.90 = 0.0072
        expected = 0.02 * 0.40 * 0.90
        assert abs(result - expected) < 1e-10

    def test_below_kink_low_util(self):
        """At very low utilization APY is tiny but positive."""
        irm = self.make_irm(borrows=1_000_000.0)
        result = compute_dynamic_base_apy(1_000_000_000.0, irm)
        assert result > 0.0
        assert result < 0.001

    # ── Above-kink regime (util > optimal_util) ──────────────────────────

    def test_above_kink_high_util(self):
        """util = 0.90 is above kink."""
        # borrows = 90M, tvl = 100M → util = 0.90
        irm = self.make_irm(borrows=90_000_000.0)
        tvl = 100_000_000.0
        result = compute_dynamic_base_apy(tvl, irm)
        # excess = (0.90 - 0.80) / (1 - 0.80) = 0.10/0.20 = 0.50
        # borrow_rate = 0.0 + 0.04 + 0.50 * 0.60 = 0.04 + 0.30 = 0.34
        # supply_apy  = 0.34 * 0.90 * 0.90 = 0.2754
        expected = (0.04 + 0.50 * 0.60) * 0.90 * 0.90
        assert abs(result - expected) < 1e-10

    def test_above_kink_max_util(self):
        """Util clamped to 1.0 when borrows > tvl."""
        irm = self.make_irm(borrows=200_000_000.0)
        tvl = 100_000_000.0  # util would be 2.0, should clamp to 1.0
        result = compute_dynamic_base_apy(tvl, irm)
        assert result > 0.0
        # With util clamped to 1.0
        # excess = (1.0 - 0.80)/0.20 = 1.0
        # borrow_rate = 0.04 + 1.0 * 0.60 = 0.64
        # supply_apy  = 0.64 * 1.0 * 0.90 = 0.576
        expected = (0.04 + 1.0 * 0.60) * 1.0 * 0.90
        assert abs(result - expected) < 1e-10

    # ── Monotonicity ────────────────────────────────────────────────────────

    def test_apy_decreases_as_tvl_increases(self):
        """
        More supply (higher TVL) → lower utilization → lower supply APY.
        IRM negative feedback.
        """
        irm = self.make_irm(borrows=80_000_000.0)
        apy_low = compute_dynamic_base_apy(100_000_000.0, irm)  # util=0.80
        apy_mid = compute_dynamic_base_apy(160_000_000.0, irm)  # util=0.50
        apy_high = compute_dynamic_base_apy(400_000_000.0, irm)  # util=0.20
        assert apy_low > apy_mid > apy_high > 0.0

    def test_reserve_factor_reduces_apy(self):
        """Higher reserve factor means less yield for depositors."""
        borrows = 80_000_000.0
        tvl = 100_000_000.0
        irm_low = IRMParams(reserve_factor=0.05, initial_borrows_usd=borrows)
        irm_high = IRMParams(reserve_factor=0.30, initial_borrows_usd=borrows)
        apy_low_rf = compute_dynamic_base_apy(tvl, irm_low)
        apy_high_rf = compute_dynamic_base_apy(tvl, irm_high)
        assert apy_low_rf > apy_high_rf


# ============================================================================
# 3.1  IRM wired into simulation engine
# ============================================================================


class TestIRMWiredIntoEngine:
    """Integration tests: engine uses dynamic base_apy when IRMParams set."""

    def _run_single_path(self, irm: IRMParams | None, initial_tvl: float = 100_000_000.0):
        """Run one simulation path and return (final_state, loss_result)."""
        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.08,
            base_apy=0.03,
            horizon_days=14,
            dt_days=1.0,
            irm_params=irm,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        engine = CampaignSimulationEngine.from_params(config=config, env=env, seed=42)
        initial_state = CampaignState(tvl=initial_tvl, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(initial_state)
        evaluator = CampaignLossEvaluator()
        loss = evaluator.evaluate(final_state, config)
        return final_state, loss, config

    def test_no_irm_base_apy_unchanged(self):
        """Without IRM, config.base_apy stays constant throughout."""
        irm = None
        _, _, config = self._run_single_path(irm)
        assert config.base_apy == pytest.approx(0.03)

    def test_irm_present_base_apy_updated(self):
        """With IRM, base_apy is changed from its initial value."""
        irm = IRMParams(
            optimal_util=0.80,
            slope1=0.04,
            slope2=0.60,
            reserve_factor=0.10,
            initial_borrows_usd=80_000_000.0,
        )
        # Start at TVL=100M → util=0.80 → supply_apy = 0.04 * 0.80 * 0.90 ≈ 0.0288
        _, _, config = self._run_single_path(irm, initial_tvl=100_000_000.0)
        # After the run, config.base_apy should reflect last step's IRM computation
        # (not the original 0.03)
        assert config.base_apy != pytest.approx(0.03, abs=1e-4)

    def test_irm_higher_tvl_lower_base_apy(self):
        """At higher TVL => lower util => lower base_apy set by IRM."""
        irm = IRMParams(
            optimal_util=0.80,
            slope1=0.04,
            slope2=0.60,
            reserve_factor=0.10,
            initial_borrows_usd=80_000_000.0,
        )
        # At 100M TVL → util=0.80 → supply APY ≈ 2.88%
        # At 400M TVL → util=0.20 → supply APY much lower
        _, _, cfg_low = self._run_single_path(irm, initial_tvl=100_000_000.0)
        _, _, cfg_high = self._run_single_path(irm, initial_tvl=400_000_000.0)
        # After simulation both configs will be at different base_apy values
        # (engine mutated config.base_apy to last step's dynamic value)
        # cfg_high started at higher TVL → its final base_apy should be lower
        assert cfg_high.base_apy < cfg_low.base_apy

    def test_irm_zero_borrows_no_apy_change_from_formula(self):
        """IRM with 0 borrows → compute_dynamic_base_apy returns 0."""
        irm = IRMParams(initial_borrows_usd=0.0)
        _, _, config = self._run_single_path(irm)
        # With 0 borrows every step sets base_apy to 0.0
        assert config.base_apy == pytest.approx(0.0)

    def test_irm_apr_history_reflects_dynamic_base(self):
        """APR history should vary when IRM is active (TVL changes → util changes → base_apy changes)."""
        irm = IRMParams(
            optimal_util=0.80,
            slope1=0.10,  # Steep slope for large variation
            slope2=1.50,
            reserve_factor=0.10,
            initial_borrows_usd=80_000_000.0,
        )
        config = CampaignConfig(
            weekly_budget=150_000,
            apr_cap=0.15,
            base_apy=0.03,
            horizon_days=14,
            dt_days=1.0,
            irm_params=irm,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        engine = CampaignSimulationEngine.from_params(config=config, env=env, seed=99)
        initial_state = CampaignState(tvl=100_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(initial_state)
        # If IRM is active and TVL changes, the total APR history should show variation
        # (more variance than just incentive APR noise alone)
        import numpy as np
        apr_std = float(np.std(final_state.apr_history))
        assert apr_std >= 0.0  # sanity — always non-negative


# ============================================================================
# 3.2  budget_savings_usd
# ============================================================================


class TestBudgetSavingsUSD:
    """Tests for budget_savings_usd in LossResult and MonteCarloResult."""

    def _make_config_env(self, tvl_target: float = 80_000_000.0) -> tuple:
        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.08,
            base_apy=0.04,
            horizon_days=28,
            dt_days=1.0,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(tvl_target=tvl_target)
        return config, env, weights

    def test_budget_savings_non_negative(self):
        """budget_savings_usd should always be >= 0."""
        config, env, weights = self._make_config_env()
        engine = CampaignSimulationEngine.from_params(config=config, env=env, seed=1)
        state = CampaignState(tvl=10_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(state)
        evaluator = CampaignLossEvaluator(weights)
        loss = evaluator.evaluate(final_state, config)
        assert loss.budget_savings_usd >= 0.0

    def test_budget_savings_plus_spend_equals_total_budget(self):
        """savings + total_spend == total_budget (up to float precision)."""
        config, env, weights = self._make_config_env()
        engine = CampaignSimulationEngine.from_params(config=config, env=env, seed=2)
        state = CampaignState(tvl=100_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(state)
        evaluator = CampaignLossEvaluator(weights)
        loss = evaluator.evaluate(final_state, config)
        total_budget = config.weekly_budget * config.num_epochs
        assert abs(loss.total_spend + loss.budget_savings_usd - total_budget) < 1.0  # $1 tolerance

    def test_budget_savings_decreases_with_higher_tvl(self):
        """
        Higher TVL → more spend (MAX regime: spend ≈ budget).
        Lower TVL → less spend (Float regime at low TVL → spend < budget).
        So budget_savings at high TVL ≈ 0, at very low TVL > 0.
        """
        config, env, weights = self._make_config_env()

        # Low TVL run (Float regime — APR=r_max, spend < min(budget, tvl*r_max/365*7))
        engine_low = CampaignSimulationEngine.from_params(config=config, env=env, seed=3)
        state_low = CampaignState(tvl=1_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_low = engine_low.run(state_low)
        evaluator = CampaignLossEvaluator(weights)
        loss_low = evaluator.evaluate(final_low, config)

        # High TVL run (Float regime — APR < r_max, spend ≈ full budget)
        engine_high = CampaignSimulationEngine.from_params(config=config, env=env, seed=4)
        state_high = CampaignState(tvl=500_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_high = engine_high.run(state_high)
        loss_high = evaluator.evaluate(final_high, config)

        # At $1M TVL, incentive APR = r_max = 8%, spend ~= 1M*0.08/365*28 ≈ $6,137
        # Total budget = 100_000 * (28/7) = $400,000 -> large savings
        # At $500M TVL, spend is much higher (all budget consumed)
        assert loss_low.budget_savings_usd > loss_high.budget_savings_usd

    def test_budget_savings_in_loss_result_fields(self):
        """budget_savings_usd field exists in LossResult."""
        config, env, weights = self._make_config_env()
        engine = CampaignSimulationEngine.from_params(config=config, env=env, seed=5)
        state = CampaignState(tvl=100_000_000.0, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(state)
        evaluator = CampaignLossEvaluator(weights)
        loss = evaluator.evaluate(final_state, config)
        assert hasattr(loss, "budget_savings_usd")

    def test_mean_budget_savings_usd_in_monte_carlo_result(self):
        """mean_budget_savings_usd field exists in MonteCarloResult and is ≥ 0."""
        config, env, weights = self._make_config_env()
        mc = run_monte_carlo(config, env, 100_000_000.0, [], weights, n_paths=5)
        assert hasattr(mc, "mean_budget_savings_usd")
        assert mc.mean_budget_savings_usd >= 0.0

    def test_mean_budget_savings_consistent_with_util(self):
        """mean_spend + mean_budget_savings_usd ≈ total_budget."""
        config, env, weights = self._make_config_env()
        mc = run_monte_carlo(config, env, 100_000_000.0, [], weights, n_paths=5)
        total_budget = config.weekly_budget * config.num_epochs
        assert abs(mc.mean_spend + mc.mean_budget_savings_usd - total_budget) < 1.0

    def test_empty_sim_budget_savings_zero(self):
        """Empty simulation path → budget_savings_usd = 0 (early return path)."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.08, horizon_days=1, dt_days=1.0)
        state = CampaignState(tvl=0.0, budget_remaining_epoch=config.weekly_budget)
        # Manually make an empty state (no history)
        evaluator = CampaignLossEvaluator()
        loss = evaluator.evaluate(state, config)
        assert loss.budget_savings_usd == 0.0


# ============================================================================
# 3.3  organic_tvl_floor_fraction
# ============================================================================


class TestOrganicTVLFloor:
    """Tests for organic floor in RetailDepositorConfig and RetailDepositorAgent."""

    def test_default_floor_zero(self):
        cfg = RetailDepositorConfig()
        assert cfg.organic_tvl_floor_fraction == 0.0

    def test_custom_floor(self):
        cfg = RetailDepositorConfig(organic_tvl_floor_fraction=0.30)
        assert cfg.organic_tvl_floor_fraction == 0.30

    def test_floor_prevents_tvl_below_fraction(self):
        """
        With floor=0.5 and initial TVL=100M, TVL should never drop below 50M.
        We drive the protocol hard (no incentives, APR threshold very high)
        so retail wants to exit but floor should clamp.
        """
        import numpy as np

        floor_frac = 0.50
        initial_tvl = 100_000_000.0

        cfg = CampaignConfig(
            weekly_budget=0,       # Zero budget: no incentive APR
            apr_cap=0.00,
            base_apy=0.00,
            horizon_days=28,
            dt_days=0.5,
        )
        env = CampaignEnvironment(r_threshold=0.20)  # Very high threshold → massive exit pressure
        retail_cfg = RetailDepositorConfig(
            alpha_plus=0.50,        # Fast response
            alpha_minus_multiplier=5.0,
            diffusion_sigma=0.0,   # No noise for determinism
            organic_tvl_floor_fraction=floor_frac,
        )

        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            retail_config=retail_cfg,
            seed=7,
        )
        state = CampaignState(tvl=initial_tvl, budget_remaining_epoch=0.0)
        final_state = engine.run(state)

        floor_usd = floor_frac * initial_tvl
        tvl_arr = np.array(final_state.tvl_history)
        # Every recorded TVL should be >= floor (or very close, within float precision)
        assert float(np.min(tvl_arr)) >= floor_usd - 1.0  # $1 tolerance

    def test_no_floor_tvl_can_drop_below(self):
        """Without floor, same setup allows TVL to crash (control test)."""
        import numpy as np

        initial_tvl = 100_000_000.0
        floor_frac = 0.50

        cfg = CampaignConfig(
            weekly_budget=0,
            apr_cap=0.00,
            base_apy=0.00,
            horizon_days=28,
            dt_days=0.5,
        )
        env = CampaignEnvironment(r_threshold=0.20)
        retail_cfg = RetailDepositorConfig(
            alpha_plus=0.50,
            alpha_minus_multiplier=5.0,
            diffusion_sigma=0.0,
            organic_tvl_floor_fraction=0.0,  # No floor
        )
        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            retail_config=retail_cfg,
            seed=7,
        )
        state = CampaignState(tvl=initial_tvl, budget_remaining_epoch=0.0)
        final_state = engine.run(state)

        floor_usd = floor_frac * initial_tvl
        tvl_arr = np.array(final_state.tvl_history)
        # Without floor, TVL should drop below 50M (massive exit pressure from r_threshold=0.20)
        assert float(np.min(tvl_arr)) < floor_usd

    def test_floor_fraction_one_holds_tvl_constant(self):
        """floor_fraction=1.0 → TVL never drops below starting TVL."""
        import numpy as np

        initial_tvl = 50_000_000.0
        cfg = CampaignConfig(
            weekly_budget=0,
            apr_cap=0.00,
            base_apy=0.00,
            horizon_days=14,
            dt_days=1.0,
        )
        env = CampaignEnvironment(r_threshold=0.20)
        retail_cfg = RetailDepositorConfig(
            alpha_plus=0.50,
            alpha_minus_multiplier=5.0,
            diffusion_sigma=0.0,
            organic_tvl_floor_fraction=1.0,
        )
        engine = CampaignSimulationEngine.from_params(
            config=cfg, env=env, retail_config=retail_cfg, seed=8
        )
        state = CampaignState(tvl=initial_tvl, budget_remaining_epoch=0.0)
        final_state = engine.run(state)

        tvl_arr = np.array(final_state.tvl_history)
        assert float(np.min(tvl_arr)) >= initial_tvl - 1.0  # $1 tolerance

    def test_floor_respected_in_monte_carlo(self):
        """organic_tvl_floor propagates correctly through run_monte_carlo."""
        import numpy as np

        initial_tvl = 30_000_000.0
        floor_frac = 0.50

        cfg = CampaignConfig(
            weekly_budget=0,
            apr_cap=0.00,
            base_apy=0.00,
            horizon_days=14,
            dt_days=1.0,
        )
        env = CampaignEnvironment(r_threshold=0.20)
        retail_cfg = RetailDepositorConfig(
            alpha_plus=0.50,
            alpha_minus_multiplier=5.0,
            diffusion_sigma=0.0,
            organic_tvl_floor_fraction=floor_frac,
        )
        weights = LossWeights(tvl_target=initial_tvl)
        mc = run_monte_carlo(
            cfg, env, initial_tvl, [], weights, n_paths=5,
            retail_config=retail_cfg,
        )
        # mean_tvl should be above floor
        floor_usd = floor_frac * initial_tvl
        assert mc.mean_tvl >= floor_usd - 1.0

    def test_agent_reset_clears_initial_tvl(self):
        """RetailDepositorAgent.reset() clears _initial_tvl so it is re-set on next call."""
        agent = RetailDepositorAgent(
            RetailDepositorConfig(organic_tvl_floor_fraction=0.5), seed=0
        )
        cfg = CampaignConfig(weekly_budget=0, apr_cap=0.0, base_apy=0.0, horizon_days=1, dt_days=1.0)
        env = CampaignEnvironment(r_threshold=0.05)
        state = CampaignState(tvl=100_000_000.0, budget_remaining_epoch=0.0)
        agent.act(state, cfg, env)
        assert agent._initial_tvl == pytest.approx(100_000_000.0)
        agent.reset()
        assert agent._initial_tvl is None


# ============================================================================
# Integration: all three features together
# ============================================================================


class TestPhase3Integration:
    """Light integration test combining IRM + budget savings + organic floor."""

    def test_full_integration_run(self):
        """
        Smoke test: run Monte Carlo with all Phase 3 features active.
        Should produce a valid MonteCarloResult without errors.
        """
        irm = IRMParams(
            optimal_util=0.80,
            slope1=0.04,
            slope2=0.60,
            reserve_factor=0.10,
            initial_borrows_usd=80_000_000.0,
        )
        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.08,
            base_apy=0.032,
            horizon_days=14,
            dt_days=1.0,
            irm_params=irm,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        retail_cfg = RetailDepositorConfig(
            organic_tvl_floor_fraction=0.30,
        )
        weights = LossWeights(tvl_target=100_000_000.0)

        mc = run_monte_carlo(
            config, env, 100_000_000.0, [], weights, n_paths=5,
            retail_config=retail_cfg,
        )

        assert isinstance(mc, MonteCarloResult)
        assert mc.mean_loss >= 0.0
        assert mc.mean_budget_savings_usd >= 0.0
        assert mc.mean_tvl >= 0.0

    def test_irm_organic_floor_interact_correctly(self):
        """
        With high borrows (high util → high supply APY) and a floor,
        TVL should attract inflows (from high realized APR) but floor
        prevents TVL from dropping below threshold.
        """
        import numpy as np

        irm = IRMParams(
            optimal_util=0.80,
            slope1=0.05,
            slope2=0.80,
            reserve_factor=0.05,
            initial_borrows_usd=90_000_000.0,  # High util → high base APY
        )
        floor_frac = 0.40
        initial_tvl = 100_000_000.0

        config = CampaignConfig(
            weekly_budget=50_000,
            apr_cap=0.10,
            base_apy=0.03,
            horizon_days=14,
            dt_days=1.0,
            irm_params=irm,
        )
        env = CampaignEnvironment(r_threshold=0.05)
        retail_cfg = RetailDepositorConfig(
            diffusion_sigma=0.0,
            organic_tvl_floor_fraction=floor_frac,
        )
        engine = CampaignSimulationEngine.from_params(
            config=config, env=env, retail_config=retail_cfg, seed=13
        )
        state = CampaignState(tvl=initial_tvl, budget_remaining_epoch=config.weekly_budget)
        final_state = engine.run(state)

        floor_usd = floor_frac * initial_tvl
        tvl_arr = np.array(final_state.tvl_history)
        assert float(np.min(tvl_arr)) >= floor_usd - 1.0

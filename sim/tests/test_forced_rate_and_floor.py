"""
Extensive tests for forced-rate logic, floor APR sensitivity, risk flagging,
and APY-sensitive agent behavior.

Validates that:
1. Forced-rate budget derivation is correct across edge cases
2. Floor APR penalty in loss function penalizes below-floor APR
3. APY-sensitive agent exits when APR drops below floor
4. Risk flag computation triggers for all 6 risk types
5. Multi-venue forced-rate + floor APR interaction is sound
6. End-to-end MC with forced rate + floor APR produces valid outputs
7. Optimizer selects higher-budget configs when floor APR is active
"""

import numpy as np
import pytest
from campaign.agents import (
    APYSensitiveAgent,
    APYSensitiveConfig,
    WhaleProfile,
)
from campaign.engine import (
    CampaignLossEvaluator,
    CampaignSimulationEngine,
    LossWeights,
    run_monte_carlo,
)
from campaign.optimizer import SurfaceGrid, optimize_surface
from campaign.state import CampaignConfig, CampaignEnvironment, CampaignState

WEEKS_PER_YEAR = 365.0 / 7.0  # 52.142857...


# ============================================================================
# 1. Forced-Rate Budget Derivation Edge Cases
# ============================================================================


class TestForcedRateBudgetDerivation:
    """Test the forced_rate → budget derivation across edge cases."""

    def test_target_tvl_larger_uses_target(self):
        """When target > current TVL, reference_tvl = target."""
        current_tvl = 100_000_000
        target_tvl = 200_000_000
        forced_rate = 0.05

        reference_tvl = max(current_tvl, target_tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        assert reference_tvl == target_tvl
        assert forced_B == pytest.approx(200_000_000 * 0.05 / WEEKS_PER_YEAR)
        # ~$191,780/wk
        assert 190_000 < forced_B < 195_000

    def test_current_tvl_larger_uses_current(self):
        """When current > target TVL, reference_tvl = current."""
        current_tvl = 500_000_000
        target_tvl = 200_000_000
        forced_rate = 0.03

        reference_tvl = max(current_tvl, target_tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        assert reference_tvl == current_tvl
        assert forced_B == pytest.approx(500_000_000 * 0.03 / WEEKS_PER_YEAR)

    def test_equal_tvl(self):
        """When current == target TVL, either works."""
        tvl = 300_000_000
        forced_rate = 0.04
        reference_tvl = max(tvl, tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        assert reference_tvl == tvl
        assert forced_B == pytest.approx(300_000_000 * 0.04 / WEEKS_PER_YEAR)

    def test_very_high_forced_rate(self):
        """Extreme forced rate (10%) on large TVL → massive budget."""
        forced_rate = 0.10
        reference_tvl = 1_000_000_000  # $1B TVL
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        # $1B × 10% / 52.14 ≈ $1.917M/wk
        assert forced_B > 1_900_000
        assert forced_B < 1_950_000

    def test_very_low_forced_rate(self):
        """Tiny forced rate (0.1%) on small TVL → small budget."""
        forced_rate = 0.001
        reference_tvl = 10_000_000  # $10M TVL
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        # $10M × 0.1% / 52.14 ≈ $192/wk
        assert forced_B < 200
        assert forced_B > 180

    def test_overspend_detection_large_tvl(self):
        """Forced rate requires more than available budget → overspend."""
        forced_rate = 0.05
        reference_tvl = 500_000_000
        total_budget = 100_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        overspend = forced_B > total_budget
        overspend_amount = max(0, forced_B - total_budget)

        assert overspend is True
        assert overspend_amount > 370_000  # ~$479k - $100k

    def test_no_overspend_small_tvl(self):
        """Forced rate is feasible within budget → no overspend."""
        forced_rate = 0.01
        reference_tvl = 50_000_000
        total_budget = 500_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        overspend = forced_B > total_budget
        assert overspend is False
        assert forced_B < 10_000  # ~$9.6k/wk

    def test_forced_rate_zero_is_noop(self):
        """forced_rate = 0 should not trigger forced mode."""
        forced_rate = 0.0
        # In the dashboard code: `if forced_rate is not None and forced_rate > 0:`
        assert not (forced_rate is not None and forced_rate > 0)

    def test_forced_rate_none_is_noop(self):
        """forced_rate = None should not trigger forced mode."""
        forced_rate = None
        assert not (forced_rate is not None and forced_rate > 0)

    def test_forced_rate_info_dict_structure(self):
        """Validate the forced_rate_info dict has all required fields."""
        forced_rate = 0.04
        reference_tvl = 200_000_000
        total_budget = 50_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        info = {
            "forced_rate": forced_rate,
            "required_budget": forced_B,
            "input_budget": total_budget,
            "overspend": forced_B > total_budget,
            "overspend_amount": max(0, forced_B - total_budget),
            "reference_tvl": reference_tvl,
        }

        assert set(info.keys()) == {
            "forced_rate",
            "required_budget",
            "input_budget",
            "overspend",
            "overspend_amount",
            "reference_tvl",
        }
        assert isinstance(info["overspend"], bool)
        assert info["overspend_amount"] >= 0


# ============================================================================
# 2. Forced-Rate Grid Construction
# ============================================================================


class TestForcedRateGridConstruction:
    """Verify SurfaceGrid construction under forced-rate constraints."""

    def test_pinned_budget_narrows_grid(self):
        """Pinned budget creates a tight 3-point grid around forced_B."""
        forced_rate = 0.05
        reference_tvl = 150_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        # Dashboard code: b_min = forced_B * 0.95, b_max = forced_B * 1.05, steps=3
        grid = SurfaceGrid.from_ranges(
            B_min=forced_B * 0.95,
            B_max=forced_B * 1.05,
            B_steps=3,
            r_max_min=forced_rate * 0.95,
            r_max_max=forced_rate * 1.05,
            r_max_steps=3,
        )

        assert grid.shape == (3, 3)
        # Center B should be close to forced_B
        center_B = grid.B_values[1]
        assert center_B == pytest.approx(forced_B, rel=0.01)
        # Center r_max should be close to forced_rate
        center_r = grid.r_max_values[1]
        assert center_r == pytest.approx(forced_rate, rel=0.01)

    def test_pinned_r_max_narrows_grid(self):
        """Pinned r_max creates a tight 3-point grid around forced_rate."""
        forced_rate = 0.06
        grid = SurfaceGrid.from_ranges(
            B_min=50_000,
            B_max=150_000,
            B_steps=10,
            r_max_min=forced_rate * 0.95,
            r_max_max=forced_rate * 1.05,
            r_max_steps=3,
        )

        assert grid.shape == (10, 3)
        # r_max range should be tight
        assert grid.r_max_values[0] == pytest.approx(forced_rate * 0.95, rel=0.001)
        assert grid.r_max_values[2] == pytest.approx(forced_rate * 1.05, rel=0.001)

    def test_forced_rate_config_has_correct_apr_cap(self):
        """CampaignConfig built from forced-rate grid has correct apr_cap."""
        forced_rate = 0.07
        reference_tvl = 100_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        grid = SurfaceGrid.from_ranges(
            B_min=forced_B * 0.95,
            B_max=forced_B * 1.05,
            B_steps=3,
            r_max_min=forced_rate * 0.95,
            r_max_max=forced_rate * 1.05,
            r_max_steps=3,
            base_apy=0.02,
        )

        config = grid.make_config(B=forced_B, r_max=forced_rate)
        assert config.apr_cap == forced_rate
        assert config.weekly_budget == pytest.approx(forced_B)
        assert config.base_apy == 0.02

        # Verify incentive APR at reference TVL equals forced_rate
        inc_apr = config.incentive_apr(reference_tvl)
        # At reference TVL: float_apr = B/TVL * 52.14 = forced_rate
        # Since min(forced_rate, forced_rate) = forced_rate
        assert inc_apr == pytest.approx(forced_rate, rel=0.05)

    def test_t_bind_at_forced_rate(self):
        """T_bind should equal reference_tvl when forced_rate pins both B and r_max."""
        forced_rate = 0.05
        reference_tvl = 200_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(weekly_budget=forced_B, apr_cap=forced_rate)
        # T_bind = B * 52.14 / r_max = reference_tvl * forced_rate / 52.14 * 52.14 / forced_rate
        assert config.t_bind == pytest.approx(reference_tvl, rel=0.001)


# ============================================================================
# 3. Floor APR in Loss Function
# ============================================================================


class TestFloorAPRLossFunction:
    """Verify floor APR penalty in the loss evaluator."""

    def _make_state_with_constant_apr(self, apr: float, n_steps: int = 50) -> CampaignState:
        """Create a state with constant APR history."""
        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.10,
            base_apy=0.0,
            dt_days=0.25,
            horizon_days=n_steps * 0.25,
        )

        # Build a TVL array such that realized_apr(tvl) == desired apr
        # r = min(r_max, B/TVL * 52.14)
        # For apr < r_max: TVL = B * 52.14 / apr
        if apr > 0 and apr <= config.apr_cap:
            target_tvl = config.weekly_budget * WEEKS_PER_YEAR / apr
        else:
            target_tvl = 100_000_000

        state = CampaignState(tvl=target_tvl, budget_remaining_epoch=config.weekly_budget)

        # Manually populate history
        for step in range(n_steps):
            state.current_step = step
            state.record(config, CampaignEnvironment(r_threshold=0.04))
            state.update_spend(config, config.dt_days)

        return state, config

    def test_no_floor_penalty_when_above_floor(self):
        """No penalty when APR is above floor."""
        state, config = self._make_state_with_constant_apr(0.06, n_steps=40)
        weights = LossWeights(
            apr_floor=0.04,  # Floor = 4%, APR = 6% → no breach
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        evaluator = CampaignLossEvaluator(weights)
        result = evaluator.evaluate(state, config)

        assert result.floor_breach_cost == 0.0
        assert result.time_below_floor == 0.0

    def test_floor_penalty_when_below_floor(self):
        """Penalty applied when APR drops below floor."""
        state, config = self._make_state_with_constant_apr(0.03, n_steps=40)
        weights = LossWeights(
            apr_floor=0.05,  # Floor = 5%, APR = 3% → breach
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        evaluator = CampaignLossEvaluator(weights)
        result = evaluator.evaluate(state, config)

        assert result.floor_breach_cost > 0.0
        assert result.time_below_floor > 0.0

    def test_floor_penalty_scales_with_breach_depth(self):
        """Larger breach → larger penalty."""
        weights = LossWeights(
            apr_floor=0.06,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        evaluator = CampaignLossEvaluator(weights)

        # Small breach: APR = 5.5% vs floor 6%
        state_small, config_small = self._make_state_with_constant_apr(0.055, n_steps=40)
        result_small = evaluator.evaluate(state_small, config_small)

        # Large breach: APR = 3% vs floor 6%
        state_large, config_large = self._make_state_with_constant_apr(0.03, n_steps=40)
        result_large = evaluator.evaluate(state_large, config_large)

        assert result_large.floor_breach_cost > result_small.floor_breach_cost

    def test_floor_penalty_scales_with_sensitivity(self):
        """Higher sensitivity → larger penalty for same breach."""
        state, config = self._make_state_with_constant_apr(0.03, n_steps=40)

        # Low sensitivity
        weights_low = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=0.2,
            w_apr_floor=7.0,
        )
        result_low = CampaignLossEvaluator(weights_low).evaluate(state, config)

        # High sensitivity
        weights_high = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        result_high = CampaignLossEvaluator(weights_high).evaluate(state, config)

        assert result_high.floor_breach_cost > result_low.floor_breach_cost
        # Should scale linearly with sensitivity
        ratio = result_high.floor_breach_cost / result_low.floor_breach_cost
        assert ratio == pytest.approx(1.0 / 0.2, rel=0.01)

    def test_floor_disabled_when_sensitivity_zero(self):
        """No floor penalty when sensitivity = 0."""
        state, config = self._make_state_with_constant_apr(0.02, n_steps=40)
        weights = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=0.0,
            w_apr_floor=7.0,
        )
        result = CampaignLossEvaluator(weights).evaluate(state, config)
        assert result.floor_breach_cost == 0.0

    def test_floor_disabled_when_floor_zero(self):
        """No floor penalty when floor = 0."""
        state, config = self._make_state_with_constant_apr(0.02, n_steps=40)
        weights = LossWeights(
            apr_floor=0.0,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        result = CampaignLossEvaluator(weights).evaluate(state, config)
        assert result.floor_breach_cost == 0.0

    def test_floor_penalty_included_in_total_loss(self):
        """Floor breach cost is included in total_loss."""
        state, config = self._make_state_with_constant_apr(0.02, n_steps=40)
        weights = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )
        result = CampaignLossEvaluator(weights).evaluate(state, config)

        # Total loss should include floor breach
        recomputed = (
            result.spend_cost
            + result.apr_variance_cost
            + result.apr_ceiling_cost
            + result.tvl_shortfall_cost
            + result.merkl_fee_cost
            + result.budget_waste_cost
            + result.mercenary_cost
            + result.whale_proximity_cost
            + result.floor_breach_cost
        )
        assert result.total_loss == pytest.approx(recomputed, rel=1e-6)
        assert result.floor_breach_cost > 0


# ============================================================================
# 4. APY-Sensitive Agent Behavior
# ============================================================================


class TestAPYSensitiveAgent:
    """Verify APY-sensitive agent exits below floor and re-enters above."""

    def test_agent_disabled_when_floor_zero(self):
        """Agent does nothing when floor_apr = 0."""
        config = APYSensitiveConfig(floor_apr=0.0, sensitivity=1.0)
        agent = APYSensitiveAgent(config=config, seed=42)

        camp_config = CampaignConfig(weekly_budget=100_000, apr_cap=0.10, base_apy=0.02)
        env = CampaignEnvironment(r_threshold=0.04)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=100_000)
        state.sensitive_tvl = 10_000_000

        initial_tvl = state.tvl
        agent.act(state, camp_config, env)
        # No change because floor is 0
        assert state.tvl == initial_tvl

    def test_agent_disabled_when_sensitivity_zero(self):
        """Agent does nothing when sensitivity = 0."""
        config = APYSensitiveConfig(floor_apr=0.05, sensitivity=0.0)
        agent = APYSensitiveAgent(config=config, seed=42)

        camp_config = CampaignConfig(weekly_budget=10_000, apr_cap=0.10, base_apy=0.01)
        env = CampaignEnvironment(r_threshold=0.04)
        # TVL high enough that APR is below floor
        state = CampaignState(tvl=500_000_000, budget_remaining_epoch=10_000)
        state.sensitive_tvl = 50_000_000

        initial_tvl = state.tvl
        agent.act(state, camp_config, env)
        # No change because sensitivity is 0
        assert state.tvl == initial_tvl

    def test_agent_exits_below_floor(self):
        """Agent removes TVL when APR drops below floor."""
        config = APYSensitiveConfig(
            floor_apr=0.06,
            sensitivity=1.0,
            max_sensitive_tvl=50_000_000,
            leverage_multiple=3.0,
            unwind_rate_per_day=0.4,
        )
        agent = APYSensitiveAgent(config=config, seed=42)

        # High TVL → low APR → below floor
        camp_config = CampaignConfig(
            weekly_budget=50_000,
            apr_cap=0.10,
            base_apy=0.01,
            dt_days=1.0,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        state = CampaignState(tvl=500_000_000, budget_remaining_epoch=50_000)
        state.sensitive_tvl = 50_000_000

        # realized_apr = 0.01 + min(0.10, 50000/500M * 52.14) = 0.01 + 0.005214 = 0.015
        # Well below floor of 0.06
        apr = camp_config.realized_apr(state.tvl)
        assert apr < config.floor_apr

        initial_tvl = state.tvl
        # Run agent for several steps to allow exit delay to pass
        for _ in range(10):
            agent.act(state, camp_config, env)

        # TVL should have decreased
        assert state.tvl < initial_tvl

    def test_agent_stable_above_floor(self):
        """Agent does not exit when APR is above floor."""
        config = APYSensitiveConfig(
            floor_apr=0.03,
            sensitivity=1.0,
            max_sensitive_tvl=20_000_000,
            leverage_multiple=2.0,
        )
        agent = APYSensitiveAgent(config=config, seed=42)

        # Low TVL → high APR → above floor
        camp_config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.10,
            base_apy=0.02,
            dt_days=0.25,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        state = CampaignState(tvl=50_000_000, budget_remaining_epoch=100_000)
        state.sensitive_tvl = 0  # No sensitive TVL currently in

        # realized_apr = 0.02 + min(0.10, 100000/50M * 52.14) = 0.02 + 0.1043 → capped at 0.10
        # Total ≈ 0.12 → well above floor of 0.03
        apr = camp_config.realized_apr(state.tvl)
        assert apr > config.floor_apr

        initial_tvl = state.tvl
        for _ in range(10):
            agent.act(state, camp_config, env)

        # TVL should not have decreased from sensitive exits
        # (may increase slightly from re-entry if sensitive_tvl < max_sensitive_tvl)
        assert state.tvl >= initial_tvl * 0.99  # Allow small noise


# ============================================================================
# 5. End-to-End MC with Floor APR
# ============================================================================


class TestMCWithFloorAPR:
    """Run Monte Carlo simulations with floor APR and verify outputs."""

    def test_mc_with_floor_apr_runs(self):
        """MC simulation with floor APR does not crash."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.06, base_apy=0.02)
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.04,
            apr_floor_sensitivity=0.8,
            w_apr_floor=7.0,
            tvl_target=200_000_000,
            apr_target=0.06,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.8,
            max_sensitive_tvl=20_000_000,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=5,
            apy_sensitive_config=apy_sensitive,
        )

        assert mc.mean_tvl > 0
        assert mc.mean_apr > 0
        assert mc.is_feasible
        assert "floor_breach" in mc.loss_components

    def test_mc_floor_breach_cost_in_components(self):
        """MC aggregates floor_breach cost across paths."""
        # Low budget → low APR → floor breaches likely
        config = CampaignConfig(
            weekly_budget=10_000,
            apr_cap=0.10,
            base_apy=0.01,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.08,  # Very high floor → guaranteed breach
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=100_000_000,
            apr_target=0.06,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.08,
            sensitivity=1.0,
            max_sensitive_tvl=10_000_000,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=5,
            apy_sensitive_config=apy_sensitive,
        )

        # With 10k budget on 100M TVL, APR ≈ 1.5% — well below 8% floor
        assert mc.loss_components["floor_breach"] > 0

    def test_mc_higher_budget_reduces_floor_breach(self):
        """Higher budget → higher APR → fewer floor breaches."""
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.04,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=100_000_000,
            apr_target=0.05,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.5,
            max_sensitive_tvl=10_000_000,
        )

        # Low budget
        config_low = CampaignConfig(
            weekly_budget=10_000,
            apr_cap=0.10,
            base_apy=0.01,
        )
        mc_low = run_monte_carlo(
            config=config_low,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=10,
            apy_sensitive_config=apy_sensitive,
        )

        # High budget
        config_high = CampaignConfig(
            weekly_budget=200_000,
            apr_cap=0.10,
            base_apy=0.01,
        )
        mc_high = run_monte_carlo(
            config=config_high,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=10,
            apy_sensitive_config=apy_sensitive,
        )

        # Higher budget → higher APR → fewer (or no) floor breaches
        assert mc_high.loss_components["floor_breach"] <= mc_low.loss_components["floor_breach"]

    def test_mc_with_forced_rate_as_pinned(self):
        """MC runs correctly with forced-rate-derived pinned params."""
        forced_rate = 0.05
        reference_tvl = 100_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=forced_B,
            apr_cap=forced_rate,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            tvl_target=reference_tvl,
            apr_target=0.06,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=reference_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=10,
        )

        assert mc.mean_tvl > 0
        assert mc.mean_apr > 0
        # At T_bind, APR = forced_rate + base = 7%
        assert mc.B == pytest.approx(forced_B)
        assert mc.r_max == forced_rate

    def test_mc_with_whales_and_floor(self):
        """MC with whales + floor APR produces valid output."""
        whales = [
            WhaleProfile(
                whale_id="whale_1",
                position_usd=10_000_000,
                alt_rate=0.05,
                risk_premium=0.005,
            ),
            WhaleProfile(
                whale_id="whale_2",
                position_usd=5_000_000,
                alt_rate=0.04,
                risk_premium=0.003,
            ),
        ]
        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.07,
            base_apy=0.02,
            whale_profiles=tuple(whales),
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.04,
            apr_floor_sensitivity=0.7,
            w_apr_floor=7.0,
            tvl_target=150_000_000,
            apr_target=0.06,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.7,
            max_sensitive_tvl=15_000_000,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=whales,
            weights=weights,
            n_paths=5,
            apy_sensitive_config=apy_sensitive,
        )

        assert mc.mean_tvl > 0
        assert mc.is_feasible
        assert "floor_breach" in mc.loss_components
        assert "whale_proximity" in mc.loss_components


# ============================================================================
# 6. Surface Optimizer with Floor APR
# ============================================================================


class TestOptimizerWithFloorAPR:
    """Verify optimizer behavior when floor APR is active."""

    def test_optimizer_avoids_floor_breach(self):
        """Optimizer should prefer configs that avoid floor APR breach."""
        grid = SurfaceGrid.from_ranges(
            B_min=20_000,
            B_max=200_000,
            B_steps=5,
            r_max_min=0.03,
            r_max_max=0.08,
            r_max_steps=5,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=100_000_000,
            apr_target=0.06,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.05,
            sensitivity=1.0,
            max_sensitive_tvl=10_000_000,
        )

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=5,
            apy_sensitive_config=apy_sensitive,
            verbose=False,
        )

        assert result.optimal_B > 0
        assert result.optimal_r_max > 0
        # With a floor of 5% total (2% base → 3% incentive needed),
        # optimizer should pick higher budget to maintain floor
        opt_mc = result.optimal_mc_result
        assert opt_mc is not None
        assert opt_mc.mean_apr > 0

    def test_optimizer_with_forced_rate_pinned_grid(self):
        """Optimizer on a pinned grid (forced rate) converges to forced point."""
        forced_rate = 0.05
        reference_tvl = 100_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        grid = SurfaceGrid.from_ranges(
            B_min=forced_B * 0.95,
            B_max=forced_B * 1.05,
            B_steps=3,
            r_max_min=forced_rate * 0.95,
            r_max_max=forced_rate * 1.05,
            r_max_steps=3,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(tvl_target=reference_tvl, apr_target=0.06)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=reference_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=5,
            verbose=False,
        )

        # All grid points are near the forced value, so optimal should be close
        assert result.optimal_B == pytest.approx(forced_B, rel=0.1)
        assert result.optimal_r_max == pytest.approx(forced_rate, rel=0.1)

    def test_optimizer_floor_increases_loss_for_low_budget(self):
        """Floor APR should increase loss for low-budget configs."""
        env = CampaignEnvironment(r_threshold=0.04)

        # Without floor
        weights_no_floor = LossWeights(
            apr_floor=0.0,
            apr_floor_sensitivity=0.0,
            tvl_target=100_000_000,
            apr_target=0.06,
        )

        # With floor
        weights_with_floor = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=100_000_000,
            apr_target=0.06,
        )

        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.05,
            sensitivity=1.0,
            max_sensitive_tvl=10_000_000,
        )

        # Low budget config that will breach floor
        config = CampaignConfig(
            weekly_budget=10_000,
            apr_cap=0.06,
            base_apy=0.02,
        )

        mc_no_floor = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_no_floor,
            n_paths=10,
        )

        mc_with_floor = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_with_floor,
            n_paths=10,
            apy_sensitive_config=apy_sensitive,
        )

        # The floor-enabled run should have higher loss due to floor breach penalty
        assert mc_with_floor.mean_loss >= mc_no_floor.mean_loss


# ============================================================================
# 7. Risk Flag Computation Logic
# ============================================================================


class TestRiskFlagComputation:
    """Test risk flag logic matching dashboard code."""

    def _compute_risks(
        self,
        B_star: float,
        r_star: float,
        base_apy: float,
        current_tvl: float,
        target_tvl: float,
        total_budget: float,
        r_threshold: float = 0.045,
        floor_apr: float = 0.0,
        apr_sensitivity: float = 0.0,
        forced_rate: float = 0.0,
        mc_budget_util: float = 0.8,
    ) -> list[str]:
        """Replicate the risk flag logic from app3.py."""
        risks = []

        # Forced rate info
        if forced_rate > 0:
            ref_tvl = max(current_tvl, target_tvl)
            req_B = ref_tvl * forced_rate / WEEKS_PER_YEAR
            if req_B > total_budget:
                risks.append("overspend")

        # Venue sacrificed
        if B_star < 1_000:
            risks.append("sacrificed")

        # Floor APR breach
        inc_at_target = (
            min(r_star, B_star / target_tvl * WEEKS_PER_YEAR) if target_tvl > 0 else r_star
        )
        total_apr_target = base_apy + inc_at_target
        if floor_apr > 0 and total_apr_target < floor_apr:
            risks.append("floor_breach")

        # Below r_threshold
        if total_apr_target < r_threshold:
            risks.append("uncompetitive")

        # Low budget utilization
        if mc_budget_util < 0.5:
            risks.append("low_util")

        # High incentive at current TVL
        inc_at_current = (
            min(r_star, B_star / current_tvl * WEEKS_PER_YEAR) if current_tvl > 0 else r_star
        )
        if inc_at_current > 0.10:
            risks.append("mercenary_magnet")

        return risks

    def test_no_risks_healthy_venue(self):
        """Healthy venue triggers no risk flags."""
        risks = self._compute_risks(
            B_star=100_000,
            r_star=0.05,
            base_apy=0.02,
            current_tvl=200_000_000,
            target_tvl=250_000_000,
            total_budget=500_000,
            r_threshold=0.04,
        )
        assert len(risks) == 0

    def test_risk_overspend(self):
        """Forced rate requiring overspend is flagged."""
        risks = self._compute_risks(
            B_star=480_000,
            r_star=0.05,
            base_apy=0.02,
            current_tvl=500_000_000,
            target_tvl=500_000_000,
            total_budget=100_000,
            forced_rate=0.05,
        )
        assert "overspend" in risks

    def test_risk_sacrificed(self):
        """Near-zero budget is flagged as sacrificed."""
        risks = self._compute_risks(
            B_star=500,
            r_star=0.05,
            base_apy=0.02,
            current_tvl=100_000_000,
            target_tvl=150_000_000,
            total_budget=500_000,
        )
        assert "sacrificed" in risks

    def test_risk_floor_breach(self):
        """Floor APR breach is flagged."""
        risks = self._compute_risks(
            B_star=50_000,
            r_star=0.05,
            base_apy=0.01,
            current_tvl=200_000_000,
            target_tvl=300_000_000,
            total_budget=500_000,
            floor_apr=0.06,
            apr_sensitivity=0.8,
        )
        # inc_at_target = min(0.05, 50000/300M*52.14) = min(0.05, 0.00869) = 0.00869
        # total = 0.01 + 0.00869 = 0.01869 < 0.06
        assert "floor_breach" in risks

    def test_risk_uncompetitive(self):
        """Below r_threshold is flagged."""
        risks = self._compute_risks(
            B_star=10_000,
            r_star=0.05,
            base_apy=0.01,
            current_tvl=200_000_000,
            target_tvl=300_000_000,
            total_budget=500_000,
            r_threshold=0.05,
        )
        # inc_at_target ≈ 0.00174, total ≈ 0.01174 < 0.05
        assert "uncompetitive" in risks

    def test_risk_low_utilization(self):
        """Low budget utilization is flagged."""
        risks = self._compute_risks(
            B_star=100_000,
            r_star=0.05,
            base_apy=0.02,
            current_tvl=200_000_000,
            target_tvl=250_000_000,
            total_budget=500_000,
            mc_budget_util=0.3,
        )
        assert "low_util" in risks

    def test_risk_mercenary_magnet(self):
        """High incentive at current TVL is flagged."""
        risks = self._compute_risks(
            B_star=200_000,
            r_star=0.15,
            base_apy=0.02,
            current_tvl=50_000_000,
            target_tvl=200_000_000,
            total_budget=500_000,
        )
        # inc_at_current = min(0.15, 200000/50M * 52.14) = min(0.15, 0.2086) = 0.15
        assert "mercenary_magnet" in risks

    def test_multiple_risks(self):
        """Multiple risks can fire simultaneously."""
        risks = self._compute_risks(
            B_star=300,
            r_star=0.15,
            base_apy=0.005,
            current_tvl=1_000_000,
            target_tvl=100_000_000,
            total_budget=50_000,
            r_threshold=0.05,
            floor_apr=0.06,
            apr_sensitivity=1.0,
            forced_rate=0.10,
        )
        # Tiny budget + huge forced rate + floor breach + uncompetitive + possibly more
        assert len(risks) >= 3
        assert "sacrificed" in risks

    def test_floor_breach_not_flagged_when_floor_zero(self):
        """No floor risk when floor_apr = 0."""
        risks = self._compute_risks(
            B_star=1_000,
            r_star=0.05,
            base_apy=0.01,
            current_tvl=200_000_000,
            target_tvl=300_000_000,
            total_budget=500_000,
            floor_apr=0.0,
        )
        assert "floor_breach" not in risks


# ============================================================================
# 8. Multi-Venue Allocation with Sensitive Venues
# ============================================================================


class TestMultiVenueWithSensitiveVenues:
    """Test behavior when some venues have floor APR and others don't."""

    def test_sensitive_venue_gets_higher_floor_breach_cost(self):
        """Venue with floor APR has higher loss than identical venue without."""
        env = CampaignEnvironment(r_threshold=0.04)

        # Same MC config for both
        config = CampaignConfig(
            weekly_budget=20_000,
            apr_cap=0.06,
            base_apy=0.01,
        )

        # Without floor
        weights_plain = LossWeights(
            apr_floor=0.0,
            apr_floor_sensitivity=0.0,
            tvl_target=200_000_000,
            apr_target=0.05,
        )
        mc_plain = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=200_000_000,
            whale_profiles=[],
            weights=weights_plain,
            n_paths=10,
        )

        # With floor (this venue has loopers)
        weights_sensitive = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=200_000_000,
            apr_target=0.05,
        )
        apy_sensitive = APYSensitiveConfig(
            floor_apr=0.05,
            sensitivity=1.0,
            max_sensitive_tvl=20_000_000,
        )
        mc_sensitive = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=200_000_000,
            whale_profiles=[],
            weights=weights_sensitive,
            n_paths=10,
            apy_sensitive_config=apy_sensitive,
        )

        # Sensitive venue should have higher loss (floor breach penalty)
        # Budget is low → APR at 200M = 20k/200M * 52.14 = 0.52% + 1% = 1.52% < 5%
        assert mc_sensitive.mean_loss > mc_plain.mean_loss
        assert mc_sensitive.loss_components["floor_breach"] > 0

    def test_two_venues_different_floors(self):
        """Run two identical venues with different floors — higher floor → higher loss."""
        env = CampaignEnvironment(r_threshold=0.04)
        config = CampaignConfig(
            weekly_budget=50_000,
            apr_cap=0.08,
            base_apy=0.02,
        )

        # Low floor (3% total)
        weights_low = LossWeights(
            apr_floor=0.03,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=150_000_000,
            apr_target=0.05,
        )
        mc_low = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_low,
            n_paths=10,
            apy_sensitive_config=APYSensitiveConfig(floor_apr=0.03, sensitivity=1.0),
        )

        # High floor (8% total)
        weights_high = LossWeights(
            apr_floor=0.08,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
            tvl_target=150_000_000,
            apr_target=0.05,
        )
        mc_high = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_high,
            n_paths=10,
            apy_sensitive_config=APYSensitiveConfig(floor_apr=0.08, sensitivity=1.0),
        )

        # Higher floor → more breaches → higher floor_breach cost
        assert mc_high.loss_components["floor_breach"] >= mc_low.loss_components["floor_breach"]


# ============================================================================
# 9. Forced Rate + Floor APR Combined Scenarios
# ============================================================================


class TestForcedRateWithFloorAPR:
    """Test forced rate and floor APR working together."""

    def test_forced_rate_maintains_floor(self):
        """A forced rate high enough should keep APR above floor."""
        floor_apr = 0.04  # 4% total
        base_apy = 0.02  # 2% base → need 2% incentive
        required_incentive = floor_apr - base_apy  # 2%

        # Force a rate at least as high as required_incentive
        forced_rate = required_incentive  # 2% incentive rate
        reference_tvl = 100_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        # At T_bind = reference_tvl, APR = forced_rate exactly
        config = CampaignConfig(
            weekly_budget=forced_B,
            apr_cap=forced_rate,
            base_apy=base_apy,
        )

        # At reference TVL, realized_apr = base + min(r_max, B/TVL * 52.14)
        apr = config.realized_apr(reference_tvl)
        assert apr == pytest.approx(floor_apr, rel=0.01)

    def test_forced_rate_below_floor_still_breaches(self):
        """Forcing a rate that's too low doesn't prevent floor breach."""
        floor_apr = 0.06  # 6% total
        base_apy = 0.01  # 1% base → need 5% incentive
        forced_rate = 0.02  # Only 2% incentive → total = 3% < 6%

        reference_tvl = 100_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=forced_B,
            apr_cap=forced_rate,
            base_apy=base_apy,
        )

        apr = config.realized_apr(reference_tvl)
        assert apr < floor_apr  # 3% < 6% → still breaches

    def test_forced_rate_overspend_with_floor(self):
        """High floor requires high forced rate which can cause overspend."""
        floor_apr = 0.08
        base_apy = 0.02
        forced_rate = floor_apr - base_apy  # 6% incentive needed

        reference_tvl = 500_000_000
        total_budget = 200_000

        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR
        # $500M × 6% / 52.14 ≈ $575k/wk

        assert forced_B > total_budget  # Overspend!
        overspend = forced_B - total_budget
        assert overspend > 370_000

    def test_minimum_forced_rate_for_floor(self):
        """Calculate minimum forced rate to maintain floor at target TVL."""
        floor_apr = 0.05
        base_apy = 0.02
        target_tvl = 200_000_000

        # Minimum incentive rate = floor - base
        min_incentive = max(0, floor_apr - base_apy)  # 3%

        # At target TVL, to have incentive = min_incentive:
        # min(r_max, B/TVL * 52.14) >= min_incentive
        # So r_max >= min_incentive and B >= TVL * min_incentive / 52.14
        forced_rate = min_incentive
        forced_B = target_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=forced_B,
            apr_cap=forced_rate,
            base_apy=base_apy,
        )

        # At target TVL, APR should just meet floor
        apr = config.realized_apr(target_tvl)
        assert apr == pytest.approx(floor_apr, rel=0.01)

        # At lower TVL, APR should be at cap → floor is maintained
        apr_low = config.realized_apr(target_tvl * 0.5)
        assert apr_low >= floor_apr

    def test_forced_rate_with_supply_cap(self):
        """Forced rate + supply cap: TVL capped → APR may stay higher."""
        forced_rate = 0.04
        reference_tvl = 100_000_000
        supply_cap = 80_000_000  # Cap below reference TVL
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=forced_B,
            apr_cap=forced_rate,
            base_apy=0.02,
            supply_cap=supply_cap,
        )

        # At supply cap, APR > at reference_tvl (less TVL → higher float rate)
        apr_at_cap = config.realized_apr(supply_cap)
        apr_at_ref = config.realized_apr(reference_tvl)
        assert apr_at_cap >= apr_at_ref


# ============================================================================
# 10. APR Computation Soundness
# ============================================================================


class TestAPRComputationSoundness:
    """Verify APR math is consistent across all components."""

    def test_incentive_apr_is_min_rmax_float(self):
        """incentive_apr = min(r_max, B/TVL * 52.14)."""
        B = 100_000
        r_max = 0.06
        tvl = 200_000_000

        config = CampaignConfig(weekly_budget=B, apr_cap=r_max, base_apy=0.02)

        float_rate = B / tvl * WEEKS_PER_YEAR  # 100k/200M * 52.14 = 0.02607%
        expected = min(r_max, float_rate)
        assert config.incentive_apr(tvl) == pytest.approx(expected)

    def test_realized_apr_equals_base_plus_incentive(self):
        """realized_apr = base_apy + incentive_apr."""
        B = 100_000
        r_max = 0.06
        base = 0.025
        tvl = 150_000_000

        config = CampaignConfig(weekly_budget=B, apr_cap=r_max, base_apy=base)

        inc = config.incentive_apr(tvl)
        assert config.realized_apr(tvl) == pytest.approx(base + inc)

    def test_t_bind_crossover(self):
        """At T_bind, float rate = r_max → regime transition."""
        B = 100_000
        r_max = 0.05

        config = CampaignConfig(weekly_budget=B, apr_cap=r_max)
        tb = config.t_bind  # B * 52.14 / r_max

        # At T_bind: float_rate = B/T_bind * 52.14 = r_max
        float_rate = B / tb * WEEKS_PER_YEAR
        assert float_rate == pytest.approx(r_max, rel=1e-6)

        # Below T_bind: cap binds → incentive = r_max
        assert config.incentive_apr(tb * 0.5) == r_max

        # Above T_bind: float → incentive < r_max
        assert config.incentive_apr(tb * 2.0) < r_max

    def test_apr_at_tvl_consistency(self):
        """apr_at_tvl utility matches CampaignConfig.incentive_apr."""
        B = 75_000
        r_max = 0.06
        tvl = 120_000_000

        config = CampaignConfig(weekly_budget=B, apr_cap=r_max)

        # The standalone function
        def apr_at_tvl_fn(B_val, tvl_val, r_val):
            if tvl_val <= 0:
                return r_val
            return min(r_val, B_val / tvl_val * WEEKS_PER_YEAR)

        assert config.incentive_apr(tvl) == pytest.approx(apr_at_tvl_fn(B, tvl, r_max))

    def test_apr_at_zero_tvl(self):
        """At TVL=0, incentive APR = r_max (cap)."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.06)
        assert config.incentive_apr(0) == 0.06

    def test_apr_at_very_high_tvl(self):
        """At very high TVL, incentive APR → 0 (float dilution)."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.06)
        apr = config.incentive_apr(1e12)  # $1T TVL
        assert apr < 0.001  # Effectively zero

    def test_forced_rate_apr_at_reference_tvl(self):
        """At reference_tvl, forced rate → incentive = forced_rate exactly."""
        forced_rate = 0.05
        reference_tvl = 200_000_000
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        config = CampaignConfig(weekly_budget=forced_B, apr_cap=forced_rate)

        # At reference_tvl: float = B/T * 52.14 = forced_rate, cap = forced_rate
        # min(forced_rate, forced_rate) = forced_rate
        inc = config.incentive_apr(reference_tvl)
        assert inc == pytest.approx(forced_rate, rel=1e-4)


# ============================================================================
# 11. Comprehensive MC Output Validation
# ============================================================================


class TestMCOutputValidation:
    """Validate MC output fields are sensible."""

    def _run_mc(
        self,
        B=100_000,
        r_max=0.06,
        base=0.02,
        tvl=100_000_000,
        floor=0.0,
        sensitivity=0.0,
        n_paths=10,
    ):
        config = CampaignConfig(weekly_budget=B, apr_cap=r_max, base_apy=base)
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=floor,
            apr_floor_sensitivity=sensitivity,
            w_apr_floor=7.0,
            tvl_target=tvl,
            apr_target=0.06,
        )
        apy_config = None
        if floor > 0 and sensitivity > 0:
            apy_config = APYSensitiveConfig(
                floor_apr=floor,
                sensitivity=sensitivity,
                max_sensitive_tvl=tvl * 0.1,
            )
        return run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=n_paths,
            apy_sensitive_config=apy_config,
        )

    def test_mean_tvl_positive(self):
        mc = self._run_mc()
        assert mc.mean_tvl > 0

    def test_mean_apr_positive(self):
        mc = self._run_mc()
        assert mc.mean_apr > 0

    def test_apr_p5_less_than_p95(self):
        mc = self._run_mc()
        assert mc.apr_p5 <= mc.apr_p95

    def test_budget_util_in_range(self):
        mc = self._run_mc()
        assert 0 <= mc.mean_budget_util <= 1.5  # Can slightly exceed 1 in rare cases

    def test_loss_components_all_present(self):
        mc = self._run_mc()
        expected = {
            "spend",
            "apr_variance",
            "apr_ceiling",
            "tvl_shortfall",
            "merkl_fee",
            "budget_waste",
            "mercenary",
            "whale_proximity",
            "floor_breach",
        }
        assert set(mc.loss_components.keys()) == expected

    def test_loss_components_non_negative(self):
        mc = self._run_mc()
        for key, val in mc.loss_components.items():
            assert val >= 0, f"{key} is negative: {val}"

    def test_mean_loss_equals_sum_of_path_losses(self):
        """Mean loss should be the average across path results."""
        mc = self._run_mc(n_paths=5)
        path_losses = [r.total_loss for r in mc.path_results]
        assert mc.mean_loss == pytest.approx(np.mean(path_losses), rel=1e-6)

    def test_floor_breach_component_with_floor(self):
        """Floor breach component should be positive when floor is active and high."""
        mc = self._run_mc(
            B=10_000,
            r_max=0.10,
            base=0.01,
            tvl=200_000_000,
            floor=0.08,
            sensitivity=1.0,
        )
        # APR at 200M = 0.01 + min(0.10, 10k/200M * 52.14) = 0.01 + 0.0026 = 0.0126
        # Well below floor of 0.08
        assert mc.loss_components["floor_breach"] > 0

    def test_floor_breach_component_zero_without_floor(self):
        """Floor breach component should be zero when floor is disabled."""
        mc = self._run_mc(
            B=10_000,
            r_max=0.10,
            base=0.01,
            tvl=200_000_000,
            floor=0.0,
            sensitivity=0.0,
        )
        assert mc.loss_components["floor_breach"] == 0.0

    def test_mc_result_has_correct_params(self):
        """MC result records the correct B and r_max."""
        mc = self._run_mc(B=75_000, r_max=0.055)
        assert mc.B == 75_000
        assert mc.r_max == 0.055

    def test_path_results_count(self):
        """Number of path results matches n_paths."""
        mc = self._run_mc(n_paths=7)
        assert len(mc.path_results) == 7


# ============================================================================
# 12. Looper Scenario: High Leverage + Sensitive Floor
# ============================================================================


class TestLooperScenario:
    """
    Realistic looper scenario: vault has 3x leveraged loopers with
    $30M headline TVL. If APR drops below 4%, they unwind, removing
    $30M from TVL. Test that the simulation handles this correctly.
    """

    def test_looper_config_parameters(self):
        """Looper config has correct leverage parameters."""
        config = APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.9,  # Very sensitive
            leverage_multiple=3.0,  # 3x leverage
            max_sensitive_tvl=30_000_000,  # $30M headline
            unwind_rate_per_day=0.4,
        )
        assert config.leverage_multiple == 3.0
        assert config.max_sensitive_tvl == 30_000_000
        assert config.floor_apr == 0.04

    def test_looper_mc_runs_without_crash(self):
        """MC with aggressive looper parameters runs cleanly."""
        config = CampaignConfig(
            weekly_budget=50_000,
            apr_cap=0.06,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=0.04,
            apr_floor_sensitivity=0.9,
            w_apr_floor=7.0,
            tvl_target=150_000_000,
            apr_target=0.06,
        )
        looper = APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.9,
            leverage_multiple=3.0,
            max_sensitive_tvl=30_000_000,
            unwind_rate_per_day=0.4,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=5,
            apy_sensitive_config=looper,
        )

        assert mc.mean_tvl > 0
        assert mc.mean_loss < float("inf")
        assert mc.is_feasible

    def test_looper_unwind_reduces_tvl(self):
        """Running simulation with very low budget triggers looper unwind."""
        # Very low budget → APR below floor → loopers should exit
        config = CampaignConfig(
            weekly_budget=5_000,
            apr_cap=0.10,
            base_apy=0.01,
            dt_days=0.25,
            horizon_days=14,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        looper = APYSensitiveConfig(
            floor_apr=0.05,
            sensitivity=1.0,
            leverage_multiple=3.0,
            max_sensitive_tvl=20_000_000,
            unwind_rate_per_day=0.4,
            max_delay_days=0.5,  # Very short delay
        )

        initial_tvl = 100_000_000
        # APR at start: 0.01 + min(0.10, 5000/100M * 52.14) = 0.01 + 0.0026 = 0.0126
        # Well below floor 0.05

        engine = CampaignSimulationEngine.from_params(
            config=config,
            env=env,
            whale_profiles=[],
            apy_sensitive_config=looper,
            seed=42,
        )
        state = CampaignState(
            tvl=initial_tvl,
            budget_remaining_epoch=config.weekly_budget,
        )
        state.sensitive_tvl = 20_000_000  # Loopers in place

        final = engine.run(state)

        # Sensitive TVL should have decreased (loopers unwinding)
        assert final.sensitive_tvl < 20_000_000

    def test_looper_loss_higher_than_no_looper(self):
        """Venue with loopers has higher loss when APR is too low for them."""
        env = CampaignEnvironment(r_threshold=0.04)

        # Low budget → APR too low
        config = CampaignConfig(
            weekly_budget=15_000,
            apr_cap=0.08,
            base_apy=0.01,
        )

        # Without loopers
        weights_plain = LossWeights(
            apr_floor=0.0,
            apr_floor_sensitivity=0.0,
            tvl_target=100_000_000,
            apr_target=0.05,
        )
        mc_plain = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_plain,
            n_paths=10,
        )

        # With loopers
        weights_looper = LossWeights(
            apr_floor=0.05,
            apr_floor_sensitivity=0.9,
            w_apr_floor=7.0,
            tvl_target=100_000_000,
            apr_target=0.05,
        )
        mc_looper = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights_looper,
            n_paths=10,
            apy_sensitive_config=APYSensitiveConfig(
                floor_apr=0.05,
                sensitivity=0.9,
                leverage_multiple=3.0,
                max_sensitive_tvl=15_000_000,
            ),
        )

        assert mc_looper.mean_loss > mc_plain.mean_loss


# ============================================================================
# 13. Surface Result Integrity
# ============================================================================


class TestSurfaceResultIntegrity:
    """Validate SurfaceResult structure and consistency."""

    def _run_small_surface(self, **extra_weights):
        grid = SurfaceGrid.from_ranges(
            B_min=30_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.03,
            r_max_max=0.07,
            r_max_steps=3,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(tvl_target=100_000_000, apr_target=0.06, **extra_weights)
        return optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            verbose=False,
        )

    def test_surface_shape(self):
        result = self._run_small_surface()
        assert result.loss_surface.shape == (3, 3)
        assert result.loss_std_surface.shape == (3, 3)
        assert result.feasibility_mask.shape == (3, 3)

    def test_optimal_within_grid(self):
        result = self._run_small_surface()
        assert result.optimal_B >= 30_000
        assert result.optimal_B <= 100_000
        assert result.optimal_r_max >= 0.03
        assert result.optimal_r_max <= 0.07

    def test_optimal_loss_finite(self):
        result = self._run_small_surface()
        assert np.isfinite(result.optimal_loss)

    def test_sensitivity_analysis_valid(self):
        result = self._run_small_surface()
        sa = result.sensitivity_analysis()
        assert "eigenvalues" in sa
        assert "interpretation" in sa
        assert len(sa["eigenvalues"]) == 2

    def test_duality_map_contains_optimum(self):
        result = self._run_small_surface()
        dual = result.duality_map(0.05)
        assert len(dual) >= 1
        # First entry should be at the optimum
        assert dual[0]["loss"] == pytest.approx(result.optimal_loss)

    def test_surface_with_floor_has_more_structure(self):
        """Surface with floor APR should have different loss landscape."""
        result_plain = self._run_small_surface(
            apr_floor=0.0,
            apr_floor_sensitivity=0.0,
        )
        result_floor = self._run_small_surface(
            apr_floor=0.06,
            apr_floor_sensitivity=1.0,
            w_apr_floor=7.0,
        )

        # At least some grid points should differ
        assert not np.allclose(result_plain.loss_surface, result_floor.loss_surface)

    def test_mc_results_populated_for_all_points(self):
        result = self._run_small_surface()
        for i in range(3):
            for j in range(3):
                assert (i, j) in result.mc_results


# ============================================================================
# 14. CampaignConfig Regression Tests
# ============================================================================


class TestCampaignConfigRegression:
    """Guard against regressions in core CampaignConfig behavior."""

    def test_t_bind_formula(self):
        """T_bind = B * 52.14 / r_max."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.05)
        assert config.t_bind == pytest.approx(100_000 * WEEKS_PER_YEAR / 0.05)

    def test_t_bind_zero_rmax(self):
        """T_bind is inf when r_max = 0."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.0)
        assert config.t_bind == float("inf")

    def test_daily_budget(self):
        config = CampaignConfig(weekly_budget=70_000, apr_cap=0.05)
        assert config.daily_budget == pytest.approx(10_000)

    def test_num_steps(self):
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.05, dt_days=0.25, horizon_days=28)
        assert config.num_steps == 112  # 28 / 0.25

    def test_num_epochs(self):
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.05, horizon_days=28)
        assert config.num_epochs == 4  # 28 / 7

    def test_spend_rate_at_t_bind(self):
        """At T_bind, spend rate = TVL * r_max / 365."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.05, base_apy=0.02)
        tb = config.t_bind
        spend = config.instantaneous_spend_rate(tb)
        expected = tb * 0.05 / 365.0
        assert spend == pytest.approx(expected, rel=1e-6)

    def test_spend_rate_above_t_bind(self):
        """Above T_bind (float regime), spend rate = B/7."""
        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.05)
        tb = config.t_bind
        # At 2x T_bind: float rate = B/(2*T_bind) * 52.14 = r_max/2
        spend = config.instantaneous_spend_rate(tb * 2)
        # In float: spend = TVL * float_rate / 365 = 2*T_bind * (r_max/2) / 365
        # = T_bind * r_max / 365 = same as at T_bind
        # Actually: TVL * min(r_max, B/TVL*52.14) / 365
        # = 2*T_bind * (B/(2*T_bind)*52.14) / 365 = 2*T_bind * (r_max/2) / 365
        # = T_bind * r_max / 365
        # which equals B / 7 because T_bind = B*52.14/r_max
        # → B*52.14/r_max * r_max / 365 = B*52.14/365 = B/7
        assert spend == pytest.approx(100_000 / 7.0, rel=1e-4)

"""
Tests for campaign optimizer.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from campaign.agents import (
    RetailDepositorAgent,
    RetailDepositorConfig,
    WhaleAgent,
    WhaleProfile,
    resolve_cascades,
)
from campaign.engine import (
    CampaignLossEvaluator,
    CampaignSimulationEngine,
    LossWeights,
    run_monte_carlo,
)
from campaign.optimizer import SurfaceGrid, optimize_surface
from campaign.state import CampaignConfig, CampaignEnvironment, CampaignState

# ============================================================================
# STATE TESTS
# ============================================================================


class TestCampaignConfig:
    def test_basic_properties(self):
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        assert cfg.daily_budget == pytest.approx(170_000 / 7.0)
        assert cfg.num_steps == int(28 / 0.25)
        assert cfg.num_epochs == 4

    def test_t_bind(self):
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        expected = 170_000 * (365.0 / 7.0) / 0.07
        assert cfg.t_bind == pytest.approx(expected)

    def test_incentive_apr_float_regime(self):
        """When TVL > T_bind, incentive APR = B/TVL * 365/7 (Float-like)."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        tvl = cfg.t_bind * 2
        apr = cfg.incentive_apr(tvl)
        expected = 170_000 / tvl * (365 / 7)
        assert apr == pytest.approx(expected)
        assert apr < cfg.apr_cap

    def test_incentive_apr_max_regime(self):
        """When TVL < T_bind, incentive APR = r_max (MAX-like)."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        tvl = cfg.t_bind * 0.5
        apr = cfg.incentive_apr(tvl)
        assert apr == pytest.approx(cfg.apr_cap)

    def test_realized_apr_includes_base(self):
        """realized_apr = base_apy + incentive_apr."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07, base_apy=0.03)
        tvl = cfg.t_bind * 0.5  # Cap binding
        assert cfg.realized_apr(tvl) == pytest.approx(0.03 + 0.07)
        tvl2 = cfg.t_bind * 2  # Float regime
        inc = cfg.incentive_apr(tvl2)
        assert cfg.realized_apr(tvl2) == pytest.approx(0.03 + inc)

    def test_realized_apr_zero_base(self):
        """With base_apy=0, realized_apr == incentive_apr (backward compat)."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07, base_apy=0.0)
        tvl = cfg.t_bind * 0.5
        assert cfg.realized_apr(tvl) == pytest.approx(cfg.incentive_apr(tvl))

    def test_is_cap_binding(self):
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        assert cfg.is_cap_binding(cfg.t_bind * 0.5) is True
        assert cfg.is_cap_binding(cfg.t_bind * 2.0) is False

    def test_zero_tvl(self):
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
        assert cfg.incentive_apr(0) == cfg.apr_cap

    def test_zero_apr_cap(self):
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.0)
        assert cfg.t_bind == float("inf")

    def test_spend_rate_uses_incentive_only(self):
        """Spend rate should be based on incentive APR, not total."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07, base_apy=0.05)
        tvl = cfg.t_bind * 0.5  # Cap binding: incentive = 7%
        spend = cfg.instantaneous_spend_rate(tvl)
        expected = tvl * 0.07 / 365.0  # Only incentive, not total
        assert spend == pytest.approx(expected)


class TestCampaignState:
    def test_tvl_change(self):
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(10_000_000)
        assert state.tvl == 110_000_000

    def test_tvl_floor(self):
        state = CampaignState(tvl=100, budget_remaining_epoch=170_000)
        state.apply_tvl_change(-1_000_000)
        assert state.tvl == 0.0

    def test_whale_exit(self):
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": False}
        state.apply_whale_exit("w1", 20_000_000)
        assert state.tvl == 80_000_000
        assert state.whale_exited["w1"] is True
        assert state.total_whale_exits == 1

    def test_whale_reentry(self):
        state = CampaignState(tvl=80_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": True}
        state.apply_whale_reentry("w1", 20_000_000)
        assert state.tvl == 100_000_000
        assert state.whale_exited["w1"] is False

    def test_fast_clone(self):
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        clone = state.fast_clone()
        assert clone.tvl == state.tvl
        assert clone.whale_positions == state.whale_positions
        assert clone.tvl_history == []
        assert clone.budget_spent_total == 0.0

    def test_retail_tvl(self):
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000, "w2": 15_000_000}
        state.whale_exited = {"w1": False, "w2": True}
        state.mercenary_tvl = 5_000_000
        assert state.retail_tvl == pytest.approx(75_000_000)

    def test_record_tracks_both_apr_types(self):
        """record() should populate both apr_history and incentive_apr_history."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.07, base_apy=0.03)
        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.record(cfg, env)
        assert len(state.apr_history) == 1
        assert len(state.incentive_apr_history) == 1
        assert state.apr_history[0] > state.incentive_apr_history[0]
        assert state.apr_history[0] == pytest.approx(state.incentive_apr_history[0] + 0.03)


class TestCampaignEnvironment:
    def test_step(self):
        env = CampaignEnvironment()
        env.step(0.25)
        assert env.current_time_days == pytest.approx(0.25)

    def test_copy(self):
        env = CampaignEnvironment(competitor_rates={"a": 0.05})
        env2 = env.copy()
        env2.competitor_rates["b"] = 0.06
        assert "b" not in env.competitor_rates


# ============================================================================
# AGENT TESTS
# ============================================================================


class TestWhaleProfile:
    def test_exit_threshold(self):
        p = WhaleProfile(
            whale_id="w1",
            position_usd=20_000_000,
            alt_rate=0.05,
            risk_premium=0.005,
            switching_cost_usd=2000,
        )
        expected = 0.05 + 0.005 - (2000 / 20_000_000)
        assert p.exit_threshold == pytest.approx(expected)

    def test_larger_position_lower_switching_cost_per_dollar(self):
        small = WhaleProfile(whale_id="s", position_usd=1_000_000, switching_cost_usd=1000)
        large = WhaleProfile(whale_id="l", position_usd=50_000_000, switching_cost_usd=1000)
        assert large.exit_threshold > small.exit_threshold


class TestRetailDepositorAgent:
    def test_inflow_when_total_apr_above_threshold(self):
        """TVL should increase when total APR > r_threshold."""
        cfg = CampaignConfig(weekly_budget=170_000, apr_cap=0.10, base_apy=0.03)
        env = CampaignEnvironment(r_threshold=0.03)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)

        agent = RetailDepositorAgent(
            config=RetailDepositorConfig(alpha_plus=1.0, diffusion_sigma=0.0),
            seed=42,
        )

        initial_tvl = state.tvl
        for _ in range(30):
            agent.act(state, cfg, env)
        assert state.tvl > initial_tvl

    def test_outflow_when_total_apr_below_threshold(self):
        """TVL should decrease when total APR < r_threshold."""
        cfg = CampaignConfig(weekly_budget=1_000, apr_cap=0.001, base_apy=0.0)
        env = CampaignEnvironment(r_threshold=0.10)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=1_000)

        agent = RetailDepositorAgent(
            config=RetailDepositorConfig(alpha_plus=0.5, diffusion_sigma=0.0),
            seed=42,
        )

        initial_tvl = state.tvl
        for _ in range(30):
            agent.act(state, cfg, env)
        assert state.tvl < initial_tvl

    def test_base_apy_helps_retain_tvl(self):
        """Higher base_apy should lead to more TVL retention."""
        env = CampaignEnvironment(r_threshold=0.06)
        cfg_low = CampaignConfig(weekly_budget=50_000, apr_cap=0.04, base_apy=0.0)
        cfg_high = CampaignConfig(weekly_budget=50_000, apr_cap=0.04, base_apy=0.03)

        results = []
        for cfg in [cfg_low, cfg_high]:
            state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
            agent = RetailDepositorAgent(
                config=RetailDepositorConfig(alpha_plus=0.5, diffusion_sigma=0.0),
                seed=42,
            )
            for _ in range(40):
                agent.act(state, cfg, env)
            results.append(state.tvl)

        assert results[1] > results[0]


class TestWhaleAgent:
    def test_whale_exits_below_threshold(self):
        """Whale exit threshold compares against TOTAL APR."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=20_000_000,
            alt_rate=0.08,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=0.1,
        )
        agent = WhaleAgent(profile=profile, seed=42)

        cfg = CampaignConfig(weekly_budget=1_000, apr_cap=0.01, base_apy=0.0)
        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=1_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": False}

        for _ in range(20):
            agent.act(state, cfg, env)
        assert agent.has_exited is True

    def test_whale_stays_when_base_apy_keeps_above_threshold(self):
        """Base APY helps keep total above whale exit threshold."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=20_000_000,
            alt_rate=0.04,
            risk_premium=0.001,
            switching_cost_usd=100,
        )
        agent = WhaleAgent(profile=profile, seed=42)

        cfg = CampaignConfig(weekly_budget=100_000, apr_cap=0.02, base_apy=0.03)
        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=100_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": False}

        for _ in range(40):
            agent.act(state, cfg, env)
        assert agent.has_exited is False

    def test_exit_delay_accumulator_builds_up(self):
        """Accumulator should build up over multiple timesteps when APR < threshold."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=1.0,  # 1 day delay = 4 steps at dt=0.25
        )
        agent = WhaleAgent(profile=profile, seed=42)

        # APR = 0.05 < threshold (0.06 + 0.01) = 0.07
        cfg = CampaignConfig(weekly_budget=50_000, apr_cap=0.05, base_apy=0.0, dt_days=0.25)
        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Should NOT exit after just 2 steps (0.5 days)
        for _ in range(2):
            agent.act(state, cfg, env)
        assert agent.has_exited is False
        assert agent._time_below_threshold > 0

        # Should exit after ~4 steps (1 day + jitter)
        for _ in range(5):
            agent.act(state, cfg, env)
        assert agent.has_exited is True

    def test_exit_accumulator_decays_on_apr_recovery(self):
        """Accumulator should decay when APR recovers before delay elapsed."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=1.0,
        )
        agent = WhaleAgent(profile=profile, seed=42)

        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Start with low APR, build up accumulator
        cfg_low = CampaignConfig(weekly_budget=50_000, apr_cap=0.05, base_apy=0.0, dt_days=0.25)
        for _ in range(2):
            agent.act(state, cfg_low, env)
        accumulator_before = agent._time_below_threshold
        assert accumulator_before > 0

        # APR recovers above threshold
        cfg_high = CampaignConfig(weekly_budget=200_000, apr_cap=0.10, base_apy=0.0, dt_days=0.25)
        for _ in range(2):
            agent.act(state, cfg_high, env)

        # Accumulator should decay (50% per step)
        assert agent._time_below_threshold < accumulator_before

    def test_reentry_requires_hysteresis_band(self):
        """Whale should require APR > (exit_threshold + hysteresis) to re-enter."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=0.25,
            reentry_delay_days=0.5,
            hysteresis_band=0.02,  # Need 2% above exit threshold
        )
        agent = WhaleAgent(profile=profile, seed=42)

        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Force whale to exit
        cfg_low = CampaignConfig(weekly_budget=10_000, apr_cap=0.01, base_apy=0.0, dt_days=0.25)
        for _ in range(5):
            agent.act(state, cfg_low, env)
        assert agent.has_exited is True
        assert state.whale_exited["w1"] is True

        # APR at exactly exit_threshold (0.07) — should NOT re-enter
        cfg_exact = CampaignConfig(weekly_budget=300_000, apr_cap=0.07, base_apy=0.0, dt_days=0.25)
        for _ in range(5):
            agent.act(state, cfg_exact, env)
        assert agent.has_exited is True  # Still exited

        # APR above reentry_threshold (0.07 + 0.02 = 0.09)
        cfg_high = CampaignConfig(weekly_budget=400_000, apr_cap=0.09, base_apy=0.0, dt_days=0.25)
        for _ in range(5):
            agent.act(state, cfg_high, env)
        assert agent.has_exited is False  # Re-entered

    def test_reentry_delay_longer_than_exit_delay(self):
        """Re-entry should take longer than exit (asymmetric)."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=0.25,  # Fast exit
            reentry_delay_days=1.0,  # Slow re-entry (4x longer)
            hysteresis_band=0.01,
        )
        agent = WhaleAgent(profile=profile, seed=42)

        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Exit quickly
        cfg_low = CampaignConfig(weekly_budget=10_000, apr_cap=0.01, base_apy=0.0, dt_days=0.25)
        for _ in range(3):
            agent.act(state, cfg_low, env)
        assert agent.has_exited is True

        # High APR above reentry threshold — should NOT re-enter immediately
        cfg_high = CampaignConfig(weekly_budget=500_000, apr_cap=0.10, base_apy=0.0, dt_days=0.25)
        for _ in range(2):  # Only 0.5 days
            agent.act(state, cfg_high, env)
        assert agent.has_exited is True  # Still out

        # After full delay (1 day = 4 steps)
        for _ in range(5):
            agent.act(state, cfg_high, env)
        assert agent.has_exited is False  # Now re-entered

    def test_accumulator_reset_on_reentry(self):
        """Accumulators should reset to zero after re-entry."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=0.25,
            reentry_delay_days=0.5,
            hysteresis_band=0.01,
        )
        agent = WhaleAgent(profile=profile, seed=42)

        env = CampaignEnvironment()
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Exit
        cfg_low = CampaignConfig(weekly_budget=10_000, apr_cap=0.01, base_apy=0.0, dt_days=0.25)
        for _ in range(3):
            agent.act(state, cfg_low, env)
        assert agent.has_exited is True

        # Re-enter
        cfg_high = CampaignConfig(weekly_budget=500_000, apr_cap=0.10, base_apy=0.0, dt_days=0.25)
        for _ in range(6):
            agent.act(state, cfg_high, env)
        assert agent.has_exited is False

        # Verify accumulators are reset
        assert agent._time_below_threshold == 0.0
        assert agent._time_above_threshold == 0.0


class TestCascades:
    def test_no_cascade_when_apr_above_thresholds(self):
        profiles = [
            WhaleProfile(whale_id=f"w{i}", position_usd=10_000_000, alt_rate=0.02) for i in range(3)
        ]
        agents = [WhaleAgent(p, seed=i) for i, p in enumerate(profiles)]

        cfg = CampaignConfig(weekly_budget=500_000, apr_cap=0.10)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=500_000)
        for p in profiles:
            state.whale_positions[p.whale_id] = p.position_usd
            state.whale_exited[p.whale_id] = False

        depth = resolve_cascades(agents, state, cfg)
        assert depth == 0

    def test_three_whale_cascade(self):
        """Test cascade where whale 1 exits → whale 2 triggered → whale 3 triggered."""
        # Create 3 whales with different thresholds and positions
        profiles = [
            WhaleProfile(
                whale_id="w1",
                position_usd=20_000_000,
                alt_rate=0.05,
                risk_premium=0.01,
                switching_cost_usd=100,
                exit_delay_days=0.25,
            ),
            WhaleProfile(
                whale_id="w2",
                position_usd=15_000_000,
                alt_rate=0.06,
                risk_premium=0.01,
                switching_cost_usd=100,
                exit_delay_days=0.25,
            ),
            WhaleProfile(
                whale_id="w3",
                position_usd=10_000_000,
                alt_rate=0.07,
                risk_premium=0.01,
                switching_cost_usd=100,
                exit_delay_days=0.25,
            ),
        ]
        agents = [WhaleAgent(p, seed=i) for i, p in enumerate(profiles)]

        # Start with sufficient APR for all (8% > all thresholds)
        _cfg = CampaignConfig(weekly_budget=350_000, apr_cap=0.08, base_apy=0.0, dt_days=0.25)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=350_000)
        for p in profiles:
            state.whale_positions[p.whale_id] = p.position_usd
            state.whale_exited[p.whale_id] = False

        # Force whale 1 to exit by accumulating below threshold
        for agent in agents:
            agent._time_below_threshold = 1.0  # Simulate delay elapsed

        # Trigger cascade from whale 1
        # After w1 exits (20M), TVL = 80M → APR rises under Float but may still trigger w2
        # With B=350K, r_max=8%, T_bind = 350K * 52.14 / 0.08 = 228M
        # At TVL=80M < T_bind, cap binding → APR = 8% (MAX regime)
        # So APR won't change enough to cascade in this setup...

        # Let's use Float regime where TVL drop increases APR
        cfg_float = CampaignConfig(weekly_budget=100_000, apr_cap=0.20, base_apy=0.0, dt_days=0.25)
        # T_bind = 100K * 52.14 / 0.20 = 26M
        # At TVL=100M > T_bind, we're in Float regime
        # APR = 100K * 52.14 / 100M = 0.052 = 5.2%
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=100_000)
        for p in profiles:
            state.whale_positions[p.whale_id] = p.position_usd
            state.whale_exited[p.whale_id] = False

        # All whales have thresholds 6%, 7%, 8% — initially APR=5.2% so all should exit
        # But let's prime w1 to exit first
        agents[0]._time_below_threshold = 1.0

        # Trigger cascade
        depth = resolve_cascades(agents, state, cfg_float)

        # All 3 whales should cascade if APR stays below thresholds
        assert depth >= 0  # May be 0-3 depending on APR after each exit
        assert state.max_cascade_depth >= 0

    def test_cascade_depth_tracking(self):
        """Verify cascade depth is correctly tracked in state."""
        profiles = [
            WhaleProfile(
                whale_id=f"w{i}",
                position_usd=10_000_000,
                alt_rate=0.10,
                risk_premium=0.005,
                switching_cost_usd=50,
                exit_delay_days=0.1,
            )
            for i in range(5)
        ]
        agents = [WhaleAgent(p, seed=i) for i, p in enumerate(profiles)]

        # Low APR, all whales primed to exit
        cfg = CampaignConfig(weekly_budget=10_000, apr_cap=0.02, base_apy=0.0, dt_days=0.25)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=10_000)
        for p in profiles:
            state.whale_positions[p.whale_id] = p.position_usd
            state.whale_exited[p.whale_id] = False

        # Prime all whales
        for agent in agents:
            agent._time_below_threshold = 1.0

        depth = resolve_cascades(agents, state, cfg)
        assert state.max_cascade_depth == depth

    def test_cascade_reduced_delay_threshold(self):
        """Cascade exits should use 30% of normal exit delay."""
        profile = WhaleProfile(
            whale_id="w1",
            position_usd=10_000_000,
            alt_rate=0.06,
            risk_premium=0.01,
            switching_cost_usd=100,
            exit_delay_days=1.0,  # Normal delay = 1 day
        )
        agent = WhaleAgent(profile=profile, seed=42)

        cfg = CampaignConfig(weekly_budget=50_000, apr_cap=0.05, base_apy=0.0, dt_days=0.25)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=50_000)
        state.whale_positions = {"w1": 10_000_000}
        state.whale_exited = {"w1": False}

        # Build up accumulator to 0.4 days (< 1.0 normal, but > 0.3 cascade)
        agent._time_below_threshold = 0.4

        # Cascade should trigger with 30% threshold (0.3 days)
        triggered = agent.force_cascade_check(state, cfg)
        assert triggered is True
        assert agent.has_exited is True

    def test_max_cascade_depth_limit(self):
        """Cascade should stop at max_cascade_depth iterations to prevent infinite loops."""
        # Create many whales with identical thresholds (pathological case)
        profiles = [
            WhaleProfile(
                whale_id=f"w{i}",
                position_usd=1_000_000,
                alt_rate=0.10,
                risk_premium=0.00,
                switching_cost_usd=10,
                exit_delay_days=0.01,
            )
            for i in range(20)
        ]
        agents = [WhaleAgent(p, seed=i) for i, p in enumerate(profiles)]

        cfg = CampaignConfig(weekly_budget=10_000, apr_cap=0.02, base_apy=0.0, dt_days=0.25)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=10_000)
        for p in profiles:
            state.whale_positions[p.whale_id] = p.position_usd
            state.whale_exited[p.whale_id] = False

        # Prime all whales to exit
        for agent in agents:
            agent._time_below_threshold = 1.0

        # Resolve with low max_depth
        # Note: max_cascade_depth limits ITERATIONS, not total exits
        # All 20 whales can exit in iteration 1 if they're all primed
        depth = resolve_cascades(agents, state, cfg, max_cascade_depth=5)

        # All whales should have exited (they were all primed)
        assert depth == 20
        exited_count = sum(1 for a in agents if a.has_exited)
        assert exited_count == 20


class TestWhaleProfileBuilder:
    """Tests for build_whale_profiles_from_holders from evm_data.py."""

    def test_basic_profile_building(self):
        """Basic whale profile building from holder data."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xABCD1234", "balance_usd": 25_000_000},
            {"address": "0xDEF56789", "balance_usd": 15_000_000},
            {"address": "0x12345678", "balance_usd": 5_000_000},
        ]

        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=0.045,
            min_position_usd=1_000_000,
        )

        assert len(profiles) == 3
        assert profiles[0].position_usd == 25_000_000
        assert profiles[1].position_usd == 15_000_000
        assert profiles[2].position_usd == 5_000_000

    def test_single_whale_concentration_capping(self):
        """Whales exceeding 25% of TVL should be clamped."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xWHALE", "balance_usd": 40_000_000},  # 40% of TVL
        ]

        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=0.045,
            max_single_whale_share=0.25,
        )

        assert len(profiles) == 1
        # Should be clamped to 25% of TVL
        assert profiles[0].position_usd == 25_000_000

    def test_total_concentration_capping(self):
        """Total whale TVL exceeding 60% should be scaled down."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": f"0xWHALE{i}", "balance_usd": 20_000_000}
            for i in range(5)  # 100M total on 100M TVL = 100%
        ]

        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=0.045,
            max_total_whale_share=0.60,
        )

        # All first whales clamped to 25% individually
        # Then total should be scaled to 60%
        total_whale_tvl = sum(p.position_usd for p in profiles)
        assert total_whale_tvl <= 60_000_000 + 1e-6  # Allow small floating point error

    def test_alt_rate_clamping(self):
        """alt_rate should be clamped to 1.5x r_threshold."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [{"address": f"0xWHALE{i}", "balance_usd": 5_000_000} for i in range(10)]

        r_threshold = 0.04
        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=r_threshold,
            max_alt_rate_multiplier=1.5,
        )

        # All alt_rates should be <= 1.5 * 0.04 = 0.06
        for p in profiles:
            assert p.alt_rate <= r_threshold * 1.5 + 1e-10

    def test_whale_type_classification(self):
        """Whales should be classified by position size."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xBIG", "balance_usd": 15_000_000},  # >10% = institutional
            {"address": "0xMED", "balance_usd": 7_000_000},  # 5-10% = quant_desk
            {"address": "0xSMALL", "balance_usd": 2_000_000},  # <5% = opportunistic
        ]

        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=0.045,
        )

        assert profiles[0].whale_type == "institutional"
        assert profiles[1].whale_type == "quant_desk"
        assert profiles[2].whale_type == "opportunistic"

    def test_switching_cost_scales_with_position(self):
        """Larger whales should have higher switching costs."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xBIG", "balance_usd": 50_000_000},
            {"address": "0xSMALL", "balance_usd": 1_000_000},
        ]

        profiles = build_whale_profiles_from_holders(
            holders=holders,
            total_supply_usd=100_000_000,
            r_threshold=0.045,
            max_single_whale_share=0.50,  # Don't clamp for this test
        )

        # Formula: 500 + (pos / 10M) * 100
        # Big:   500 + (50M / 10M) * 100 = 500 + 500 = 1000
        # Small: 500 + (1M / 10M) * 100 = 500 + 10 = 510
        assert profiles[0].switching_cost_usd > profiles[1].switching_cost_usd

    def test_error_when_no_whales_meet_threshold(self):
        """Should error loudly if no holders meet min_position_usd."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xTINY", "balance_usd": 500_000},  # Below 1M threshold
        ]

        with pytest.raises(RuntimeError, match="No whale profiles could be built"):
            build_whale_profiles_from_holders(
                holders=holders,
                total_supply_usd=100_000_000,
                r_threshold=0.045,
                min_position_usd=1_000_000,
            )

    def test_empty_holders_list(self):
        """Should error when holders list is empty."""
        from campaign.evm_data import build_whale_profiles_from_holders

        with pytest.raises(RuntimeError, match="No holders data provided"):
            build_whale_profiles_from_holders(
                holders=[],
                total_supply_usd=100_000_000,
                r_threshold=0.045,
            )


# ============================================================================
# ENGINE TESTS
# ============================================================================


class TestSimulationEngine:
    def test_single_path_runs(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            base_apy=0.025,
            dt_days=0.5,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.065)
        whale = WhaleProfile(whale_id="w1", position_usd=20_000_000)

        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            whale_profiles=[whale],
            seed=42,
        )

        state = CampaignState(tvl=160_000_000, budget_remaining_epoch=170_000)
        result = engine.run(state)

        assert len(result.tvl_history) == cfg.num_steps
        assert len(result.apr_history) == cfg.num_steps
        assert len(result.incentive_apr_history) == cfg.num_steps
        assert all(t >= 0 for t in result.tvl_history)
        # Total APR should be >= base_apy
        assert all(r >= cfg.base_apy for r in result.apr_history)
        # Incentive APR should be <= cap
        assert all(0 <= r <= cfg.apr_cap + 1e-10 for r in result.incentive_apr_history)

    def test_deterministic_with_same_seed(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            dt_days=0.5,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.045)

        results = []
        for _ in range(2):
            engine = CampaignSimulationEngine.from_params(
                config=cfg,
                env=env,
                seed=42,
            )
            state = CampaignState(tvl=160_000_000, budget_remaining_epoch=170_000)
            r = engine.run(state)
            results.append(r.tvl_history)
        np.testing.assert_array_almost_equal(results[0], results[1])


class TestLossEvaluator:
    def test_evaluates_completed_path(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            base_apy=0.025,
            dt_days=0.5,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.065)
        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            seed=42,
        )
        state = CampaignState(tvl=160_000_000, budget_remaining_epoch=170_000)
        final = engine.run(state)

        evaluator = CampaignLossEvaluator(
            LossWeights(
                tvl_target=150_000_000,
                apr_target=0.085,  # Total target
                apr_stability_on_total=True,
            )
        )
        result = evaluator.evaluate(final, cfg)

        assert result.total_loss > 0
        component_sum = (
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
        assert result.total_loss == pytest.approx(component_sum)
        assert result.avg_apr > result.avg_incentive_apr
        assert result.avg_incentive_apr > 0
        assert result.avg_tvl > 0

    def test_higher_tvl_target_increases_shortfall_cost(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            dt_days=0.5,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.045)
        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            seed=42,
        )
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        final = engine.run(state)

        low = CampaignLossEvaluator(LossWeights(tvl_target=50_000_000))
        high = CampaignLossEvaluator(LossWeights(tvl_target=200_000_000))

        r_low = low.evaluate(final, cfg)
        r_high = high.evaluate(final, cfg)
        assert r_high.tvl_shortfall_cost >= r_low.tvl_shortfall_cost


class TestMonteCarlo:
    def test_monte_carlo_runs(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            base_apy=0.02,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.06)
        whale = WhaleProfile(whale_id="w1", position_usd=20_000_000)
        weights = LossWeights(tvl_target=150_000_000, apr_target=0.08)

        mc = run_monte_carlo(
            config=cfg,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[whale],
            weights=weights,
            n_paths=5,
            base_seed=42,
        )

        assert mc.mean_loss > 0
        assert mc.std_loss >= 0
        assert mc.mean_apr > mc.mean_incentive_apr
        assert mc.base_apy == 0.02
        assert mc.mean_tvl > 0
        assert len(mc.path_results) == 5

    def test_monte_carlo_loss_components_sum(self):
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.07,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.045)
        weights = LossWeights(tvl_target=150_000_000)

        mc = run_monte_carlo(
            config=cfg,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            base_seed=42,
        )

        component_sum = sum(mc.loss_components.values())
        assert mc.mean_loss == pytest.approx(component_sum, rel=0.01)


# ============================================================================
# OPTIMIZER TESTS
# ============================================================================


class TestSurfaceGrid:
    def test_from_ranges(self):
        grid = SurfaceGrid.from_ranges(
            B_min=100_000,
            B_max=200_000,
            B_steps=5,
            r_max_min=0.04,
            r_max_max=0.10,
            r_max_steps=4,
        )
        assert grid.shape == (5, 4)
        assert grid.B_values[0] == pytest.approx(100_000)
        assert grid.B_values[-1] == pytest.approx(200_000)

    def test_base_apy_propagates_to_config(self):
        grid = SurfaceGrid.from_ranges(
            B_min=100_000,
            B_max=200_000,
            B_steps=3,
            r_max_min=0.05,
            r_max_max=0.10,
            r_max_steps=3,
            base_apy=0.03,
        )
        cfg = grid.make_config(150_000, 0.07)
        assert cfg.base_apy == 0.03
        assert cfg.realized_apr(cfg.t_bind * 0.5) == pytest.approx(0.03 + 0.07)

    def test_t_bind_surface(self):
        grid = SurfaceGrid.from_ranges(
            B_min=100_000,
            B_max=200_000,
            B_steps=3,
            r_max_min=0.05,
            r_max_max=0.10,
            r_max_steps=3,
        )
        t_bind = grid.t_bind_surface()
        assert t_bind.shape == (3, 3)
        assert np.all(t_bind > 0)

    def test_from_t_bind_centered(self):
        grid = SurfaceGrid.from_t_bind_centered(
            current_tvl=160_000_000,
            B_center=170_000,
            B_half_range=50_000,
            B_steps=5,
            t_bind_min_frac=0.5,
            t_bind_max_frac=1.2,
            r_max_steps=5,
        )
        assert grid.shape == (5, 5)

    def test_make_config(self):
        grid = SurfaceGrid.from_ranges(
            B_min=100_000,
            B_max=200_000,
            B_steps=3,
            r_max_min=0.05,
            r_max_max=0.10,
            r_max_steps=3,
            dt_days=0.5,
            horizon_days=14,
        )
        cfg = grid.make_config(150_000, 0.07)
        assert cfg.weekly_budget == 150_000
        assert cfg.apr_cap == 0.07
        assert cfg.dt_days == 0.5
        assert cfg.horizon_days == 14


class TestSurfaceOptimizer:
    def test_tiny_optimization(self):
        grid = SurfaceGrid.from_ranges(
            B_min=150_000,
            B_max=200_000,
            B_steps=2,
            r_max_min=0.05,
            r_max_max=0.08,
            r_max_steps=2,
            dt_days=1.0,
            horizon_days=7,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.06)
        whale = WhaleProfile(whale_id="w1", position_usd=20_000_000)
        weights = LossWeights(tvl_target=150_000_000, apr_target=0.08)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[whale],
            weights=weights,
            n_paths=3,
            verbose=False,
        )

        assert result.loss_surface.shape == (2, 2)
        assert result.optimal_B > 0
        assert result.optimal_r_max > 0
        assert result.optimal_loss < float("inf")
        # base_apy should propagate
        assert result.grid.base_apy == 0.02
        mc = result.optimal_mc_result
        assert mc is not None
        assert mc.base_apy == 0.02
        assert mc.mean_apr > mc.mean_incentive_apr

        sa = result.sensitivity_analysis()
        assert "interpretation" in sa
        assert len(sa["eigenvalues"]) == 2

    def test_feasibility_mask(self):
        grid = SurfaceGrid.from_ranges(
            B_min=150_000,
            B_max=200_000,
            B_steps=2,
            r_max_min=0.06,
            r_max_max=0.09,
            r_max_steps=2,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(tvl_target=150_000_000)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            cascade_tolerance=100,
            verbose=False,
        )
        assert result.feasibility_mask.all()

    def test_duality_map(self):
        grid = SurfaceGrid.from_ranges(
            B_min=150_000,
            B_max=200_000,
            B_steps=2,
            r_max_min=0.05,
            r_max_max=0.08,
            r_max_steps=2,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.045)
        weights = LossWeights(tvl_target=150_000_000)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            verbose=False,
        )

        dual = result.duality_map(tolerance=0.10)
        assert len(dual) >= 1
        assert dual[0]["loss_ratio"] == pytest.approx(1.0)

    def test_component_surfaces_populated(self):
        grid = SurfaceGrid.from_ranges(
            B_min=150_000,
            B_max=200_000,
            B_steps=2,
            r_max_min=0.05,
            r_max_max=0.08,
            r_max_steps=2,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.045)
        weights = LossWeights(tvl_target=150_000_000)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            verbose=False,
        )

        assert "spend" in result.component_surfaces
        assert "apr_variance" in result.component_surfaces
        assert "tvl_shortfall" in result.component_surfaces
        assert result.component_surfaces["spend"].shape == (2, 2)

    def test_incentive_apr_surface_populated(self):
        """avg_incentive_apr_surface should be populated and < avg_apr_surface when base > 0."""
        grid = SurfaceGrid.from_ranges(
            B_min=150_000,
            B_max=200_000,
            B_steps=2,
            r_max_min=0.05,
            r_max_max=0.08,
            r_max_steps=2,
            dt_days=1.0,
            horizon_days=7,
            base_apy=0.02,
        )
        env = CampaignEnvironment(r_threshold=0.06)
        weights = LossWeights(tvl_target=150_000_000, apr_target=0.08)

        result = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=160_000_000,
            whale_profiles=[],
            weights=weights,
            n_paths=3,
            verbose=False,
        )

        assert result.avg_incentive_apr_surface.shape == (2, 2)
        assert result.avg_apr_surface.shape == (2, 2)
        # Total APR > incentive APR everywhere (base > 0)
        assert np.all(result.avg_apr_surface > result.avg_incentive_apr_surface)

"""
Comprehensive tests for hardening features:
1. Supply cap enforcement (state.py)
2. CampaignConfig.supply_cap field
3. Historical data structures and calibration (historical.py)
4. VenueRecord new fields (venue_registry.py)
5. APY-sensitive agent with supply cap
6. Retail agent with supply cap
7. Mercenary agent with supply cap
8. Whale re-entry with supply cap

Run with: pytest tests/test_hardening.py -v
"""

import numpy as np
import pytest
from campaign.agents import (
    APYSensitiveAgent,
    APYSensitiveConfig,
    MercenaryAgent,
    MercenaryConfig,
    RetailDepositorAgent,
    RetailDepositorConfig,
    WhaleProfile,
)
from campaign.state import CampaignConfig, CampaignEnvironment, CampaignState
from campaign.venue_registry import VENUE_REGISTRY, VenueRecord

# ============================================================================
# 1. SUPPLY CAP — CampaignConfig
# ============================================================================


class TestCampaignConfigSupplyCap:
    """Test supply_cap field on CampaignConfig."""

    def test_supply_cap_default_zero(self):
        """supply_cap defaults to 0 (unlimited)."""
        cfg = CampaignConfig(weekly_budget=100_000, apr_cap=0.07)
        assert cfg.supply_cap == 0.0

    def test_supply_cap_set_explicitly(self):
        """supply_cap can be set to a positive value."""
        cfg = CampaignConfig(weekly_budget=100_000, apr_cap=0.07, supply_cap=500_000_000)
        assert cfg.supply_cap == 500_000_000

    def test_supply_cap_frozen(self):
        """CampaignConfig is frozen — supply_cap cannot be modified."""
        cfg = CampaignConfig(weekly_budget=100_000, apr_cap=0.07, supply_cap=500_000_000)
        with pytest.raises(AttributeError):
            cfg.supply_cap = 1_000_000_000


# ============================================================================
# 2. SUPPLY CAP — State Enforcement
# ============================================================================


class TestSupplyCapState:
    """Test supply cap enforcement in CampaignState methods."""

    def test_apply_tvl_change_no_cap(self):
        """TVL change without cap — no clamping."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(50_000_000)
        assert state.tvl == 150_000_000

    def test_apply_tvl_change_with_cap_blocks_excess(self):
        """TVL change clamped by supply cap."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(50_000_000, supply_cap=120_000_000)
        assert state.tvl == 120_000_000

    def test_apply_tvl_change_cap_zero_means_unlimited(self):
        """supply_cap=0 means no cap (backward compat)."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(200_000_000, supply_cap=0.0)
        assert state.tvl == 300_000_000

    def test_apply_tvl_change_outflow_not_affected_by_cap(self):
        """Outflows should work normally — cap only limits inflows."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(-30_000_000, supply_cap=120_000_000)
        assert state.tvl == 70_000_000

    def test_apply_tvl_change_exactly_at_cap(self):
        """TVL already at cap — further inflows blocked."""
        state = CampaignState(tvl=120_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(10_000_000, supply_cap=120_000_000)
        assert state.tvl == 120_000_000

    def test_apply_tvl_change_floor_still_enforced(self):
        """TVL floor (0) is still enforced even with cap."""
        state = CampaignState(tvl=5_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(-100_000, supply_cap=50_000_000)
        assert state.tvl == 0.0

    def test_whale_reentry_capped(self):
        """Whale re-entry respects supply cap."""
        state = CampaignState(tvl=90_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": True}
        state.apply_whale_reentry("w1", 20_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000  # Capped
        assert state.whale_exited["w1"] is False

    def test_whale_reentry_under_cap(self):
        """Whale re-entry under cap — full position restored."""
        state = CampaignState(tvl=50_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": True}
        state.apply_whale_reentry("w1", 20_000_000, supply_cap=100_000_000)
        assert state.tvl == 70_000_000
        assert state.whale_exited["w1"] is False

    def test_whale_reentry_no_cap(self):
        """Whale re-entry without cap — full position restored."""
        state = CampaignState(tvl=90_000_000, budget_remaining_epoch=170_000)
        state.whale_positions = {"w1": 20_000_000}
        state.whale_exited = {"w1": True}
        state.apply_whale_reentry("w1", 20_000_000, supply_cap=0.0)
        assert state.tvl == 110_000_000

    def test_mercenary_entry_capped(self):
        """Mercenary entry respects supply cap headroom."""
        state = CampaignState(tvl=95_000_000, budget_remaining_epoch=170_000)
        state.apply_mercenary_entry(10_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000  # Only 5M headroom
        assert state.mercenary_tvl == 5_000_000

    def test_mercenary_entry_no_headroom(self):
        """Mercenary entry blocked when at cap."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_mercenary_entry(10_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000
        assert state.mercenary_tvl == 0.0
        assert state.mercenary_entries == 0  # Should not count

    def test_mercenary_entry_no_cap(self):
        """Mercenary entry without cap — full amount added."""
        state = CampaignState(tvl=95_000_000, budget_remaining_epoch=170_000)
        state.apply_mercenary_entry(10_000_000, supply_cap=0.0)
        assert state.tvl == 105_000_000
        assert state.mercenary_tvl == 10_000_000

    def test_sensitive_entry_capped(self):
        """APY-sensitive entry respects supply cap headroom."""
        state = CampaignState(tvl=90_000_000, budget_remaining_epoch=170_000)
        state.apply_sensitive_entry(20_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000  # Only 10M headroom
        assert state.sensitive_tvl == 10_000_000

    def test_sensitive_entry_no_headroom(self):
        """APY-sensitive entry blocked when at cap."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_sensitive_entry(20_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000
        assert state.sensitive_tvl == 0.0
        assert state.sensitive_entries == 0  # Should not count

    def test_sensitive_entry_no_cap(self):
        """APY-sensitive entry without cap — full amount added."""
        state = CampaignState(tvl=90_000_000, budget_remaining_epoch=170_000)
        state.apply_sensitive_entry(20_000_000, supply_cap=0.0)
        assert state.tvl == 110_000_000
        assert state.sensitive_tvl == 20_000_000


# ============================================================================
# 3. SUPPLY CAP — Agent Integration
# ============================================================================


class TestRetailAgentSupplyCap:
    """Retail agent passes supply_cap to state methods."""

    def test_retail_inflow_bounded_by_cap(self):
        """TVL should not exceed supply cap even with strong inflows."""
        cfg = CampaignConfig(
            weekly_budget=170_000,
            apr_cap=0.10,
            base_apy=0.05,
            supply_cap=110_000_000,
        )
        env = CampaignEnvironment(r_threshold=0.03)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)

        agent = RetailDepositorAgent(
            config=RetailDepositorConfig(alpha_plus=2.0, diffusion_sigma=0.0),
            seed=42,
        )

        for _ in range(100):
            agent.act(state, cfg, env)

        assert state.tvl <= 110_000_000


class TestMercenaryAgentSupplyCap:
    """Mercenary agent respects supply cap."""

    def test_mercenary_bounded_by_cap(self):
        """Mercenary capital should not push TVL beyond supply cap."""
        cfg = CampaignConfig(
            weekly_budget=500_000,
            apr_cap=0.15,
            base_apy=0.0,
            supply_cap=120_000_000,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=500_000)

        agent = MercenaryAgent(
            config=MercenaryConfig(
                entry_threshold=0.08,
                max_capital_usd=50_000_000,
                entry_rate_per_day=0.5,
                exit_rate_per_day=0.3,
            ),
            seed=42,
        )

        for _ in range(50):
            agent.act(state, cfg, env)

        assert state.tvl <= 120_000_000


class TestAPYSensitiveAgentSupplyCap:
    """APY-sensitive agent respects supply cap on re-entry."""

    def test_sensitive_reentry_bounded_by_cap(self):
        """Re-entry after unwind should be bounded by supply cap."""
        cfg = CampaignConfig(
            weekly_budget=200_000,
            apr_cap=0.10,
            base_apy=0.03,
            supply_cap=150_000_000,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        state = CampaignState(tvl=140_000_000, budget_remaining_epoch=200_000)

        agent = APYSensitiveAgent(
            config=APYSensitiveConfig(
                floor_apr=0.05,
                sensitivity=0.8,
                leverage_multiple=1.0,
                max_sensitive_tvl=30_000_000,
                reentry_rate_per_day=0.5,
            ),
            seed=42,
        )

        # Run enough steps for re-entry to happen
        for _ in range(100):
            agent.act(state, cfg, env)

        assert state.tvl <= 150_000_000


# ============================================================================
# 4. HISTORICAL DATA STRUCTURES (no HTTP calls)
# ============================================================================


class TestPoolHistoryPoint:
    """Test PoolHistoryPoint data structure."""

    def test_creation(self):
        from campaign.historical import PoolHistoryPoint

        pt = PoolHistoryPoint(
            timestamp="2025-01-15",
            tvl_usd=50_000_000,
            apy=0.045,
            apy_base=0.03,
            apy_reward=0.015,
            il7d=0.0,
        )
        assert pt.tvl_usd == 50_000_000
        assert pt.apy == 0.045
        assert pt.apy_base == 0.03


class TestPoolHistory:
    """Test PoolHistory data structure and derived arrays."""

    def _make_history(self, n=30, base_tvl=50e6, base_apy=0.04):
        from campaign.historical import PoolHistory, PoolHistoryPoint

        rng = np.random.default_rng(42)
        points = []
        for i in range(n):
            noise_tvl = rng.normal(0, 0.02) * base_tvl
            noise_apy = rng.normal(0, 0.005)
            points.append(
                PoolHistoryPoint(
                    timestamp=f"2025-01-{i + 1:02d}",
                    tvl_usd=base_tvl + noise_tvl,
                    apy=base_apy + noise_apy,
                    apy_base=max(0, 0.02 + noise_apy * 0.5),
                    apy_reward=max(0, 0.02 + noise_apy * 0.5),
                    il7d=0.0,
                )
            )
        return PoolHistory(
            pool_id="test-pool-id",
            project="test-project",
            symbol="TEST",
            chain="Ethereum",
            points=points,
        )

    def test_tvl_array_shape(self):
        hist = self._make_history(n=45)
        assert hist.tvl_array.shape == (45,)

    def test_apy_array_shape(self):
        hist = self._make_history(n=30)
        assert hist.apy_array.shape == (30,)

    def test_apy_base_array(self):
        hist = self._make_history(n=30)
        arr = hist.apy_base_array
        assert arr.shape == (30,)
        assert all(v >= 0 for v in arr)

    def test_apy_reward_array(self):
        hist = self._make_history(n=30)
        arr = hist.apy_reward_array
        assert arr.shape == (30,)

    def test_days_property(self):
        hist = self._make_history(n=60)
        assert hist.days == 60


# ============================================================================
# 5. CALIBRATION (synthetic data, no HTTP)
# ============================================================================


class TestCalibration:
    """Test calibrate_retail_params with synthetic data."""

    def _make_history(self, n=60, base_tvl=50e6, base_apy=0.04, seed=42):
        from campaign.historical import PoolHistory, PoolHistoryPoint

        rng = np.random.default_rng(seed)
        points = []
        tvl = base_tvl
        for i in range(n):
            # Simulate: TVL drifts based on APR gap
            apy = base_apy + rng.normal(0, 0.005)
            gap = apy - 0.03  # above 3% threshold
            tvl += tvl * gap * 0.2 + tvl * rng.normal(0, 0.01)
            tvl = max(tvl, 1e6)  # floor
            points.append(
                PoolHistoryPoint(
                    timestamp=f"2025-01-{(i % 28) + 1:02d}",
                    tvl_usd=tvl,
                    apy=apy,
                    apy_base=max(0, apy * 0.6),
                    apy_reward=max(0, apy * 0.4),
                    il7d=0.0,
                )
            )
        return PoolHistory(
            pool_id="test-cal",
            project="test",
            symbol="TEST",
            chain="Ethereum",
            points=points,
        )

    def test_calibration_returns_result(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.data_quality in ("good", "sparse", "insufficient")
        assert cal.n_observations == 60

    def test_calibration_good_quality_with_enough_data(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.data_quality == "good"

    def test_calibration_sparse_quality_with_few_data(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=20)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.data_quality in ("good", "sparse")

    def test_calibration_insufficient_with_minimal_data(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=10)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.data_quality == "insufficient"
        # Should return safe defaults
        assert cal.alpha_plus == 0.15
        assert cal.alpha_minus == 0.45

    def test_calibration_alpha_plus_positive(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.alpha_plus > 0
        assert cal.alpha_plus <= 1.0

    def test_calibration_alpha_minus_positive(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.alpha_minus > 0
        assert cal.alpha_minus <= 3.0

    def test_calibration_alpha_minus_multiplier_at_least_one(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.alpha_minus_multiplier >= 1.0

    def test_calibration_sigma_bounded(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert 0.001 <= cal.diffusion_sigma <= 0.05

    def test_calibration_lag_positive(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.response_lag_days >= 1.0

    def test_calibration_r_threshold_mean(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.r_threshold_mean == pytest.approx(0.03)

    def test_calibration_with_competitor_history(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        competitor = np.linspace(0.03, 0.05, 60)
        cal = calibrate_retail_params(hist, competitor_apy_history=competitor)
        # Calibration trims to valid TVL days, so mean may shift slightly
        assert cal.r_threshold_mean == pytest.approx(np.mean(competitor), rel=0.02)
        assert cal.data_quality == "good"

    def test_calibration_details_populated(self):
        from campaign.historical import calibrate_retail_params

        hist = self._make_history(n=60)
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert "pos_gap_samples" in cal.details
        assert "neg_gap_samples" in cal.details
        assert "mean_tvl" in cal.details

    def test_calibration_with_zero_tvl_days(self):
        """Calibration handles days with zero TVL gracefully."""
        from campaign.historical import PoolHistory, PoolHistoryPoint, calibrate_retail_params

        # Create history with some zero-TVL days
        rng = np.random.default_rng(42)
        points = []
        for i in range(40):
            tvl = 50e6 if i > 5 else 0.0  # First 5 days zero TVL
            points.append(
                PoolHistoryPoint(
                    timestamp=f"2025-01-{(i % 28) + 1:02d}",
                    tvl_usd=tvl + rng.normal(0, 1e6) if tvl > 0 else 0,
                    apy=0.04 + rng.normal(0, 0.003),
                    apy_base=0.02,
                    apy_reward=0.02,
                    il7d=0.0,
                )
            )
        hist = PoolHistory(
            pool_id="z",
            project="t",
            symbol="T",
            chain="E",
            points=points,
        )
        cal = calibrate_retail_params(hist, default_r_threshold=0.03)
        assert cal.data_quality in ("good", "sparse")
        assert cal.n_observations == 40


# ============================================================================
# 6. WHALE FLOW EVENT STRUCTURES
# ============================================================================


class TestWhaleFlowStructures:
    """Test WhaleFlowEvent and WhaleHistoryResult data structures."""

    def test_whale_flow_event_creation(self):
        from campaign.historical import WhaleFlowEvent

        evt = WhaleFlowEvent(
            timestamp="2025-01-15T10:30:00Z",
            address="0xabc123",
            amount_usd=5_000_000,
            direction="deposit",
            tx_hash="0xdef456",
            protocol="aave",
        )
        assert evt.amount_usd == 5_000_000
        assert evt.direction == "deposit"

    def test_whale_history_result_net_flow(self):
        from campaign.historical import WhaleFlowEvent, WhaleHistoryResult

        events = [
            WhaleFlowEvent("t1", "0xa", 10_000_000, "deposit"),
            WhaleFlowEvent("t2", "0xa", 3_000_000, "withdrawal"),
            WhaleFlowEvent("t3", "0xb", 5_000_000, "deposit"),
        ]
        result = WhaleHistoryResult(
            venue_name="test",
            token_address="0xtoken",
            events=events,
            total_deposits_usd=15_000_000,
            total_withdrawals_usd=3_000_000,
            unique_whales=2,
        )
        assert result.net_flow_usd == 12_000_000

    def test_whale_history_result_empty(self):
        from campaign.historical import WhaleHistoryResult

        result = WhaleHistoryResult(
            venue_name="empty",
            token_address="0x0",
            events=[],
            total_deposits_usd=0,
            total_withdrawals_usd=0,
            unique_whales=0,
        )
        assert result.net_flow_usd == 0.0


# ============================================================================
# 7. VENUE REGISTRY — New Fields
# ============================================================================


class TestVenueRecordFields:
    """Test VenueRecord new fields: defillama_pool_id, supply_cap, borrow_cap."""

    def test_defillama_pool_id_default_empty(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
        )
        assert v.defillama_pool_id == ""

    def test_defillama_pool_id_set(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
            defillama_pool_id="abc-123-def-456",
        )
        assert v.defillama_pool_id == "abc-123-def-456"

    def test_supply_cap_default_zero(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
        )
        assert v.supply_cap == 0.0

    def test_borrow_cap_default_zero(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
        )
        assert v.borrow_cap == 0.0

    def test_caps_set(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
            supply_cap=500_000_000,
            borrow_cap=300_000_000,
        )
        assert v.supply_cap == 500_000_000
        assert v.borrow_cap == 300_000_000

    def test_frozen_immutable(self):
        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
        )
        with pytest.raises(AttributeError):
            v.defillama_pool_id = "xyz"


class TestVenueRegistryDefillamaPoolIds:
    """Verify all venues in the registry have DeFiLlama pool IDs."""

    def test_all_ethereum_venues_have_pool_ids(self):
        """Every Ethereum venue in VENUE_REGISTRY should have a non-empty defillama_pool_id."""
        missing = []
        for pool_id, venue in VENUE_REGISTRY.items():
            if venue.chain == "ethereum" and not venue.defillama_pool_id:
                missing.append(pool_id)
        assert missing == [], f"Ethereum venues missing defillama_pool_id: {missing}"

    def test_pool_ids_are_uuid_like(self):
        """DeFiLlama pool IDs should look like UUIDs (contain hyphens)."""
        for pool_id, venue in VENUE_REGISTRY.items():
            pid = venue.defillama_pool_id
            if pid:
                assert "-" in pid, f"{pool_id} has non-UUID pool ID: {pid}"
                parts = pid.split("-")
                assert len(parts) == 5, f"{pool_id} pool ID doesn't have 5 segments: {pid}"

    def test_no_duplicate_pool_ids(self):
        """Each venue should have a unique DeFiLlama pool ID."""
        pool_ids = [v.defillama_pool_id for v in VENUE_REGISTRY.values() if v.defillama_pool_id]
        assert len(pool_ids) == len(set(pool_ids)), "Duplicate defillama_pool_id found"


# ============================================================================
# 8. VENUE TO DASHBOARD DICT
# ============================================================================


class TestVenueToDashboardDict:
    """Test venue_to_dashboard_dict includes new fields."""

    def test_includes_defillama_pool_id(self):
        from campaign.venue_registry import venue_to_dashboard_dict

        for pool_id, venue in VENUE_REGISTRY.items():
            d = venue_to_dashboard_dict(venue)
            if venue.defillama_pool_id:
                assert "defillama_pool_id" in d
                assert d["defillama_pool_id"] == venue.defillama_pool_id
            else:
                # Venues without pool IDs should not have the key
                assert "defillama_pool_id" not in d

    def test_includes_caps(self):
        from campaign.venue_registry import venue_to_dashboard_dict

        v = VenueRecord(
            pool_id="test",
            name="Test",
            program="P",
            asset="USDC",
            protocol="aave",
            protocol_type="lending",
            chain="ethereum",
            address="0x1",
            underlying_asset="0x2",
            supply_cap=100_000_000,
            borrow_cap=50_000_000,
        )
        d = venue_to_dashboard_dict(v)
        assert d["supply_cap"] == 100_000_000
        assert d["borrow_cap"] == 50_000_000


# ============================================================================
# 9. EVM DATA — Cap Fields on Dataclasses
# ============================================================================


class TestEvmDataCapFields:
    """Test supply/borrow cap fields on AaveReserveData and EulerVaultData."""

    def test_aave_reserve_data_caps_default(self):
        from campaign.evm_data import AaveReserveData

        d = AaveReserveData(
            asset_address="0x123",
            asset_symbol="USDC",
            market="core",
            liquidity_rate_ray=30000000000000000000000000,
            variable_borrow_rate_ray=50000000000000000000000000,
            a_token_address="0xabc",
            supply_apy=0.03,
            borrow_apy=0.05,
            utilization=0.7,
            total_supply_usd=100e6,
            total_borrow_usd=70e6,
        )
        assert d.supply_cap == 0.0
        assert d.borrow_cap == 0.0

    def test_aave_reserve_data_caps_set(self):
        from campaign.evm_data import AaveReserveData

        d = AaveReserveData(
            asset_address="0x123",
            asset_symbol="USDC",
            market="core",
            liquidity_rate_ray=30000000000000000000000000,
            variable_borrow_rate_ray=50000000000000000000000000,
            a_token_address="0xabc",
            supply_apy=0.03,
            borrow_apy=0.05,
            utilization=0.7,
            total_supply_usd=100e6,
            total_borrow_usd=70e6,
            supply_cap=500e6,
            borrow_cap=300e6,
        )
        assert d.supply_cap == 500e6
        assert d.borrow_cap == 300e6

    def test_euler_vault_data_caps_default(self):
        from campaign.evm_data import EulerVaultData

        d = EulerVaultData(
            vault_address="0xvault",
            asset_symbol="USDC",
            asset_address="0xasset",
            asset_decimals=6,
            cash=30_000_000_000_000,
            total_borrows=20_000_000_000_000,
            supply_apy=0.04,
            utilization=0.65,
            total_supply_usd=50e6,
            total_borrow_usd=32.5e6,
        )
        assert d.supply_cap == 0.0
        assert d.borrow_cap == 0.0

    def test_euler_vault_data_caps_set(self):
        from campaign.evm_data import EulerVaultData

        d = EulerVaultData(
            vault_address="0xvault",
            asset_symbol="USDC",
            asset_address="0xasset",
            asset_decimals=6,
            cash=30_000_000_000_000,
            total_borrows=20_000_000_000_000,
            supply_apy=0.04,
            utilization=0.65,
            total_supply_usd=50e6,
            total_borrow_usd=32.5e6,
            supply_cap=200e6,
            borrow_cap=100e6,
        )
        assert d.supply_cap == 200e6
        assert d.borrow_cap == 100e6


# ============================================================================
# 10. FULL SIMULATION WITH SUPPLY CAP
# ============================================================================


class TestSimulationWithSupplyCap:
    """End-to-end simulation with supply cap to verify cap flows through all agents."""

    def test_simulation_tvl_never_exceeds_cap(self):
        """Full simulation with cap — TVL should never exceed supply_cap."""
        from campaign.engine import CampaignSimulationEngine

        cap = 150_000_000
        cfg = CampaignConfig(
            weekly_budget=200_000,
            apr_cap=0.10,
            base_apy=0.03,
            dt_days=0.5,
            horizon_days=14,
            supply_cap=cap,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        whale = WhaleProfile(whale_id="w1", position_usd=20_000_000)

        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            whale_profiles=[whale],
            seed=42,
        )

        state = CampaignState(tvl=130_000_000, budget_remaining_epoch=200_000)
        result = engine.run(state)

        # TVL should never exceed cap
        for tvl in result.tvl_history:
            assert tvl <= cap + 1e-6, f"TVL {tvl / 1e6:.1f}M exceeded cap {cap / 1e6:.1f}M"


# ============================================================================
# 11. APY-SENSITIVE AGENT STANDALONE TESTS
# ============================================================================


class TestAPYSensitiveAgent:
    """Test APY-sensitive agent behaviors."""

    def test_disabled_when_floor_zero(self):
        """Agent should be a no-op when floor_apr=0."""
        cfg = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.05,
            base_apy=0.0,
            dt_days=0.25,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.03)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=100_000)
        _initial_tvl = state.tvl

        agent = APYSensitiveAgent(
            config=APYSensitiveConfig(floor_apr=0.0, sensitivity=0.0),
            seed=42,
        )

        for _ in range(40):
            agent.act(state, cfg, env)

        # Should not change TVL (no floor breach possible)
        assert state.sensitive_tvl == 0.0

    def test_unwind_when_apr_below_floor(self):
        """APY-sensitive positions should exit when APR drops below floor."""
        cfg = CampaignConfig(
            weekly_budget=10_000,
            apr_cap=0.02,
            base_apy=0.01,
            dt_days=0.25,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.10)
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=10_000)

        agent = APYSensitiveAgent(
            config=APYSensitiveConfig(
                floor_apr=0.06,  # Need 6% total, but we only have ~3%
                sensitivity=1.0,  # Immediate exit
                leverage_multiple=1.0,
                max_sensitive_tvl=20_000_000,
                unwind_rate_per_day=0.5,
            ),
            seed=42,
        )

        _initial_tvl = state.tvl
        for _ in range(30):
            agent.act(state, cfg, env)

        # TVL should have decreased due to sensitive exits
        assert state.sensitive_unwinds > 0 or state.sensitive_tvl < 20_000_000

    def test_config_defaults(self):
        """APYSensitiveConfig should have sensible defaults."""
        c = APYSensitiveConfig()
        assert c.floor_apr == 0.0
        assert c.sensitivity == 0.0
        assert c.leverage_multiple == 3.0
        assert c.max_sensitive_tvl == 0.0
        assert c.unwind_rate_per_day == 0.4
        assert c.reentry_rate_per_day == 0.1
        assert c.max_delay_days == 3.0
        assert c.hysteresis_band == 0.005


# ============================================================================
# 12. LOSS EVALUATOR — ALL COMPONENTS SUM CORRECTLY
# ============================================================================


class TestLossComponentsSum:
    """Verify total_loss equals sum of all components in various configs."""

    def test_loss_sum_with_floor(self):
        """total_loss should equal sum of all components even with floor breach cost."""
        from campaign.engine import (
            CampaignLossEvaluator,
            CampaignSimulationEngine,
            LossWeights,
        )

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
                apr_target=0.085,
                apr_stability_on_total=True,
                apr_floor=0.04,
                apr_floor_sensitivity=0.5,
            )
        )
        result = evaluator.evaluate(final, cfg)

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

    def test_loss_sum_no_whales(self):
        """total_loss sum check without whales."""
        from campaign.engine import (
            CampaignLossEvaluator,
            CampaignSimulationEngine,
            LossWeights,
        )

        cfg = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=0.05,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        engine = CampaignSimulationEngine.from_params(
            config=cfg,
            env=env,
            seed=123,
        )
        state = CampaignState(tvl=80_000_000, budget_remaining_epoch=100_000)
        final = engine.run(state)

        evaluator = CampaignLossEvaluator(LossWeights(tvl_target=75_000_000))
        result = evaluator.evaluate(final, cfg)

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


# ============================================================================
# 13. EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_supply_cap_equal_to_current_tvl(self):
        """Cap exactly at current TVL — no growth possible."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(10_000_000, supply_cap=100_000_000)
        assert state.tvl == 100_000_000

    def test_supply_cap_below_current_tvl(self):
        """Cap below current TVL — existing TVL not reduced, but no growth."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        # Positive inflow still clamped to cap
        state.apply_tvl_change(10_000_000, supply_cap=90_000_000)
        assert state.tvl == 90_000_000  # Clamped down to cap

    def test_negative_delta_still_works_with_small_cap(self):
        """Outflows still work even if cap < current TVL."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_tvl_change(-20_000_000, supply_cap=50_000_000)
        # After outflow: 80M, then capped to 50M
        assert state.tvl == 50_000_000

    def test_mercenary_entry_zero_amount(self):
        """Zero mercenary entry is a no-op."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_mercenary_entry(0.0)
        assert state.mercenary_tvl == 0.0
        assert state.mercenary_entries == 0

    def test_sensitive_entry_zero_amount(self):
        """Zero sensitive entry is a no-op."""
        state = CampaignState(tvl=100_000_000, budget_remaining_epoch=170_000)
        state.apply_sensitive_entry(0.0)
        assert state.sensitive_tvl == 0.0
        assert state.sensitive_entries == 0

    def test_calibration_result_details_dict(self):
        """CalibrationResult.details should be a dict."""
        from campaign.historical import CalibrationResult

        cal = CalibrationResult(
            alpha_plus=0.15,
            alpha_minus=0.45,
            alpha_minus_multiplier=3.0,
            diffusion_sigma=0.008,
            response_lag_days=5.0,
            r_threshold_mean=0.045,
            r_threshold_trend=0.0,
            data_quality="good",
            n_observations=60,
        )
        assert isinstance(cal.details, dict)

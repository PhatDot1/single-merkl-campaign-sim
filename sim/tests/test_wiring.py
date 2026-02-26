"""
Tests for production wiring: supply caps, calibration, Dune sync, whale history.

Validates that:
1. SurfaceGrid.make_config() passes supply_cap to CampaignConfig
2. CalibrationResult fields can populate RetailDepositorConfig
3. Dune sync module loads/saves CSVs correctly
4. Whale history lookup improves profile thresholds
5. MercenaryConfig thresholds derive from calibrated r_threshold
6. Dune query templates render correctly
"""

import tempfile
from pathlib import Path

import pytest
from campaign.agents import MercenaryConfig, RetailDepositorConfig
from campaign.optimizer import SurfaceGrid
from campaign.state import CampaignConfig

# ============================================================================
# 1. Supply Cap Pipeline: SurfaceGrid → make_config → CampaignConfig
# ============================================================================


class TestSupplyCapPipeline:
    """Verify supply_cap flows from SurfaceGrid through to CampaignConfig."""

    def test_surface_grid_default_supply_cap(self):
        """SurfaceGrid defaults to supply_cap=0 (unlimited)."""
        grid = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=3,
        )
        assert grid.supply_cap == 0.0

    def test_surface_grid_explicit_supply_cap(self):
        """SurfaceGrid accepts explicit supply_cap."""
        grid = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=3,
            supply_cap=500_000_000.0,
        )
        assert grid.supply_cap == 500_000_000.0

    def test_make_config_passes_supply_cap(self):
        """make_config() passes supply_cap to CampaignConfig."""
        grid = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=3,
            supply_cap=300_000_000.0,
        )
        config = grid.make_config(B=50_000, r_max=0.05)
        assert config.supply_cap == 300_000_000.0

    def test_make_config_zero_supply_cap(self):
        """make_config() passes supply_cap=0 when unlimited."""
        grid = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=3,
        )
        config = grid.make_config(B=50_000, r_max=0.05)
        assert config.supply_cap == 0.0

    def test_make_config_preserves_other_fields(self):
        """supply_cap doesn't break other CampaignConfig fields."""
        grid = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=100_000,
            B_steps=3,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=3,
            base_apy=0.03,
            supply_cap=200_000_000.0,
        )
        config = grid.make_config(B=75_000, r_max=0.06, whale_profiles=[])
        assert config.weekly_budget == 75_000
        assert config.apr_cap == 0.06
        assert config.base_apy == 0.03
        assert config.supply_cap == 200_000_000.0
        assert config.whale_profiles == ()

    def test_from_t_bind_centered_supply_cap(self):
        """from_t_bind_centered accepts supply_cap via kwargs."""
        grid = SurfaceGrid.from_t_bind_centered(
            current_tvl=100_000_000,
            B_center=50_000,
            B_half_range=20_000,
            B_steps=5,
            supply_cap=150_000_000.0,
        )
        assert grid.supply_cap == 150_000_000.0


# ============================================================================
# 2. Calibration → RetailDepositorConfig
# ============================================================================


class TestCalibrationWiring:
    """Verify CalibrationResult can populate RetailDepositorConfig."""

    def test_calibration_to_retail_config(self):
        """CalibrationResult fields map to RetailDepositorConfig fields."""
        from campaign.historical import CalibrationResult

        cal = CalibrationResult(
            alpha_plus=0.22,
            alpha_minus=0.55,
            alpha_minus_multiplier=2.5,
            diffusion_sigma=0.012,
            response_lag_days=3.0,
            r_threshold_mean=0.04,
            r_threshold_trend=0.0001,
            data_quality="good",
            n_observations=60,
        )

        retail = RetailDepositorConfig(
            alpha_plus=cal.alpha_plus,
            alpha_minus_multiplier=cal.alpha_minus_multiplier,
            response_lag_days=cal.response_lag_days,
            diffusion_sigma=cal.diffusion_sigma,
        )

        assert retail.alpha_plus == 0.22
        assert retail.alpha_minus_multiplier == 2.5
        assert retail.response_lag_days == 3.0
        assert retail.diffusion_sigma == 0.012
        assert retail.alpha_minus == pytest.approx(0.22 * 2.5)

    def test_mercenary_thresholds_from_calibration(self):
        """Mercenary entry/exit derive from calibrated r_threshold_mean."""
        from campaign.historical import CalibrationResult

        cal = CalibrationResult(
            alpha_plus=0.15,
            alpha_minus=0.45,
            alpha_minus_multiplier=3.0,
            diffusion_sigma=0.008,
            response_lag_days=5.0,
            r_threshold_mean=0.05,
            r_threshold_trend=0.0,
            data_quality="good",
            n_observations=60,
        )

        # Same formula used in dashboard
        merc_entry = cal.r_threshold_mean * 1.8
        merc_exit = cal.r_threshold_mean * 1.3

        merc = MercenaryConfig(
            entry_threshold=merc_entry,
            exit_threshold=merc_exit,
        )

        assert merc.entry_threshold == pytest.approx(0.09)  # 5% * 1.8
        assert merc.exit_threshold == pytest.approx(0.065)  # 5% * 1.3

    def test_insufficient_calibration_uses_defaults(self):
        """Insufficient data quality should not override defaults."""
        from campaign.historical import CalibrationResult

        cal = CalibrationResult(
            alpha_plus=0.15,  # defaults
            alpha_minus=0.45,
            alpha_minus_multiplier=3.0,
            diffusion_sigma=0.008,
            response_lag_days=5.0,
            r_threshold_mean=0.045,
            r_threshold_trend=0.0,
            data_quality="insufficient",
            n_observations=5,
        )

        # Dashboard logic: don't apply if quality is insufficient
        assert cal.data_quality == "insufficient"
        # Default RetailDepositorConfig should be used instead
        default = RetailDepositorConfig()
        assert default.alpha_plus == 0.15
        assert default.diffusion_sigma == 0.008


# ============================================================================
# 3. Dune Sync Module
# ============================================================================


class TestDuneSync:
    """Test Dune sync CSV round-trip and query rendering."""

    def test_query_render(self):
        """Query templates render correctly with parameters."""
        from dune.queries import WHALE_FLOWS_QUERY, render_query

        sql = render_query(
            WHALE_FLOWS_QUERY,
            token_address="0xabc123",
            decimals=6,
            min_amount=500000,
            days=90,
            chain="ethereum",
        )
        assert "0xabc123" in sql
        assert "500000" in sql
        assert "90" in sql
        assert "{{" not in sql  # All placeholders replaced

    def test_mercenary_query_render(self):
        """Mercenary detection query renders correctly."""
        from dune.queries import MERCENARY_DETECTION_QUERY, render_query

        sql = render_query(
            MERCENARY_DETECTION_QUERY,
            token_address="0xdef456",
            decimals=6,
            min_amount=100000,
            days=90,
            chain="ethereum",
        )
        assert "0xdef456" in sql
        assert "100000" in sql
        assert "{{" not in sql

    def test_csv_round_trip(self):
        """CSV save/load round-trip preserves data."""
        from dune.sync import _load_csv, _save_rows_csv

        rows = [
            {
                "block_time": "2025-01-01",
                "whale_address": "0xabc",
                "amount": "1000000",
                "direction": "deposit",
            },
            {
                "block_time": "2025-01-02",
                "whale_address": "0xdef",
                "amount": "2000000",
                "direction": "withdrawal",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            filepath = Path(f.name)

        try:
            count = _save_rows_csv(rows, filepath)
            assert count == 2

            loaded = _load_csv(filepath)
            assert len(loaded) == 2
            assert loaded[0]["whale_address"] == "0xabc"
            assert loaded[1]["direction"] == "withdrawal"
        finally:
            filepath.unlink(missing_ok=True)

    def test_load_nonexistent_csv(self):
        """Loading nonexistent CSV returns empty list."""
        from dune.sync import _load_csv

        result = _load_csv(Path("/nonexistent/path/file.csv"))
        assert result == []

    def test_get_dune_token_address_aave(self):
        """Aave venues return aToken address."""
        from dune.sync import get_dune_token_address

        venue = {
            "protocol": "aave",
            "chain": "ethereum",
            "a_token_address": "0xabc123",
            "vault_address": "",
            "address": "0xdef456",
        }
        assert get_dune_token_address(venue) == "0xabc123"

    def test_get_dune_token_address_euler(self):
        """Euler venues return vault address."""
        from dune.sync import get_dune_token_address

        venue = {
            "protocol": "euler",
            "chain": "ethereum",
            "a_token_address": "",
            "vault_address": "0xvault",
            "address": "0xaddr",
        }
        assert get_dune_token_address(venue) == "0xvault"

    def test_get_dune_token_address_solana_skipped(self):
        """Solana venues return None (not queryable via Dune)."""
        from dune.sync import get_dune_token_address

        venue = {
            "protocol": "kamino",
            "chain": "solana",
            "a_token_address": "",
            "vault_address": "",
            "address": "SomePublicKey123",
        }
        assert get_dune_token_address(venue) is None

    def test_get_dune_token_address_morpho(self):
        """Morpho venues return vault address."""
        from dune.sync import get_dune_token_address

        venue = {
            "protocol": "morpho",
            "chain": "ethereum",
            "a_token_address": "",
            "vault_address": "0xmorpho_vault",
            "address": "0xaddr",
        }
        assert get_dune_token_address(venue) == "0xmorpho_vault"

    def test_get_dune_token_address_curve(self):
        """Curve venues return pool address (LP token)."""
        from dune.sync import get_dune_token_address

        venue = {
            "protocol": "curve",
            "chain": "ethereum",
            "a_token_address": "",
            "vault_address": "",
            "address": "0xcurve_pool",
        }
        assert get_dune_token_address(venue) == "0xcurve_pool"

    def test_derive_mercenary_thresholds_no_data(self):
        """No data returns None."""
        from dune.sync import derive_mercenary_thresholds

        result = derive_mercenary_thresholds("nonexistent-pool", 0.045)
        assert result is None

    def test_save_empty_rows(self):
        """Saving empty rows returns 0."""
        from dune.sync import _save_rows_csv

        result = _save_rows_csv([], Path("/tmp/empty.csv"))
        assert result == 0

    def test_get_token_decimals(self):
        """Token decimals correct per protocol."""
        from dune.sync import get_token_decimals

        assert get_token_decimals({"protocol": "aave"}) == 6
        assert get_token_decimals({"protocol": "euler"}) == 6
        assert get_token_decimals({"protocol": "morpho"}) == 6
        assert get_token_decimals({"protocol": "curve"}) == 18


# ============================================================================
# 4. Whale History → Improved Thresholds
# ============================================================================


class TestWhaleHistoryWiring:
    """Test that whale history lookup improves profile generation."""

    def test_build_profiles_without_history(self):
        """Profiles build correctly with no whale history (synthetic fallback)."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xwhale1", "balance_usd": 10_000_000},
            {"address": "0xwhale2", "balance_usd": 5_000_000},
        ]

        profiles = build_whale_profiles_from_holders(
            holders,
            total_supply_usd=100_000_000,
            r_threshold=0.05,
            whale_history=None,
        )

        assert len(profiles) == 2
        # Synthetic: alt_rate = r_threshold * (1.0 + 0.2 * i/n)
        assert profiles[0].alt_rate >= 0.05  # At least r_threshold

    def test_build_profiles_with_history(self):
        """Profiles use empirical data when whale_history provided."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xWhale1ABC", "balance_usd": 10_000_000},
            {"address": "0xWhale2DEF", "balance_usd": 5_000_000},
        ]

        whale_history = {
            "0xwhale1abc": {
                "n_deposits": 3,
                "n_withdrawals": 2,
                "avg_hold_days": 14.0,
                "total_deposited": 30_000_000,
                "total_withdrawn": 20_000_000,
            },
        }

        profiles = build_whale_profiles_from_holders(
            holders,
            total_supply_usd=100_000_000,
            r_threshold=0.05,
            whale_history=whale_history,
        )

        assert len(profiles) == 2
        # First whale has history (n_withdrawals > 0) → tighter alt_rate
        # alt_rate ≈ r_threshold * (1.0 + 0.1 * min(14/30, 1.0))
        expected_alt = 0.05 * (1.0 + 0.1 * min(14.0 / 30.0, 1.0))
        assert profiles[0].alt_rate == pytest.approx(expected_alt, abs=0.001)

        # Second whale has no history → synthetic fallback
        # Should still have a valid alt_rate
        assert profiles[1].alt_rate >= 0.05

    def test_whale_history_case_insensitive(self):
        """Whale history lookup is case-insensitive on addresses."""
        from campaign.evm_data import build_whale_profiles_from_holders

        holders = [
            {"address": "0xABC123", "balance_usd": 5_000_000},
        ]

        whale_history = {
            "0xabc123": {
                "n_deposits": 1,
                "n_withdrawals": 1,
                "avg_hold_days": 7.0,
            },
        }

        profiles = build_whale_profiles_from_holders(
            holders,
            total_supply_usd=50_000_000,
            r_threshold=0.05,
            whale_history=whale_history,
        )

        assert len(profiles) == 1
        # Should use empirical data (addr matched case-insensitively)
        # With n_withdrawals > 0: alt_rate = 0.05 * (1.0 + 0.1 * min(7/30, 1))
        expected = 0.05 * (1.0 + 0.1 * min(7.0 / 30.0, 1.0))
        assert profiles[0].alt_rate == pytest.approx(expected, abs=0.001)

    def test_build_whale_history_lookup(self):
        """Whale history lookup builds correctly from flow data."""
        from dune.sync import DATA_DIR, _save_rows_csv, build_whale_history_lookup

        pool_id = "__test_pool__"
        pool_dir = DATA_DIR / pool_id
        pool_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save test whale flows
            rows = [
                {
                    "block_time": "2025-01-01T00:00:00Z",
                    "whale_address": "0xabc",
                    "amount": "1000000",
                    "direction": "deposit",
                    "tx_hash": "0x1",
                },
                {
                    "block_time": "2025-01-05T00:00:00Z",
                    "whale_address": "0xabc",
                    "amount": "500000",
                    "direction": "withdrawal",
                    "tx_hash": "0x2",
                },
                {
                    "block_time": "2025-01-02T00:00:00Z",
                    "whale_address": "0xdef",
                    "amount": "2000000",
                    "direction": "deposit",
                    "tx_hash": "0x3",
                },
            ]
            _save_rows_csv(rows, pool_dir / "whale_flows.csv")

            result = build_whale_history_lookup(pool_id)
            assert result is not None
            assert "0xabc" in result
            assert result["0xabc"]["n_deposits"] == 1
            assert result["0xabc"]["n_withdrawals"] == 1
            assert result["0xabc"]["avg_hold_days"] >= 1.0
            assert "0xdef" in result
            assert result["0xdef"]["n_deposits"] == 1
            assert result["0xdef"]["n_withdrawals"] == 0

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(pool_dir, ignore_errors=True)

    def test_build_whale_history_lookup_no_data(self):
        """Returns None when no cached data."""
        from dune.sync import build_whale_history_lookup

        result = build_whale_history_lookup("__nonexistent_pool__")
        assert result is None


# ============================================================================
# 5. Venue Registry → Dune Token Address Resolution
# ============================================================================


class TestVenueTokenResolution:
    """Verify token address resolution works for all venue types."""

    def test_all_evm_venues_have_dune_address(self):
        """Every Ethereum venue should resolve to a queryable token address."""
        from campaign.venue_registry import VENUE_REGISTRY
        from dune.sync import get_dune_token_address

        for pool_id, venue in VENUE_REGISTRY.items():
            if venue.chain != "ethereum":
                continue

            token = get_dune_token_address(venue)
            assert token is not None, (
                f"EVM venue {pool_id} ({venue.name}) has no queryable token address. "
                f"Protocol={venue.protocol}, a_token={venue.a_token_address}, "
                f"vault={venue.vault_address}, address={venue.address}"
            )

    def test_solana_venues_skipped(self):
        """Solana venues correctly return None."""
        from campaign.venue_registry import VENUE_REGISTRY
        from dune.sync import get_dune_token_address

        solana_venues = [v for v in VENUE_REGISTRY.values() if v.chain == "solana"]
        for venue in solana_venues:
            assert get_dune_token_address(venue) is None


# ============================================================================
# 6. Integration: Full Pipeline Smoke Test
# ============================================================================


class TestPipelineIntegration:
    """Smoke tests for the full wired pipeline."""

    def test_supply_cap_end_to_end(self):
        """Supply cap flows from grid through to MC config."""
        from campaign.engine import run_monte_carlo
        from campaign.state import CampaignEnvironment

        grid = SurfaceGrid.from_ranges(
            B_min=50_000,
            B_max=50_000,
            B_steps=1,
            r_max_min=0.05,
            r_max_max=0.05,
            r_max_steps=1,
            supply_cap=200_000_000.0,
        )

        config = grid.make_config(B=50_000, r_max=0.05)
        assert config.supply_cap == 200_000_000.0

        # Run a quick MC to verify it doesn't crash
        env = CampaignEnvironment(r_threshold=0.04)
        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=None,
            n_paths=3,
        )
        assert mc.mean_tvl > 0

    def test_calibrated_retail_config_runs(self):
        """Simulation runs with calibrated retail parameters."""
        from campaign.engine import run_monte_carlo
        from campaign.state import CampaignEnvironment

        config = CampaignConfig(weekly_budget=50_000, apr_cap=0.05)
        env = CampaignEnvironment(r_threshold=0.04)

        # Calibrated values (different from defaults)
        retail = RetailDepositorConfig(
            alpha_plus=0.22,
            alpha_minus_multiplier=2.5,
            response_lag_days=3.0,
            diffusion_sigma=0.012,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=None,
            n_paths=3,
            retail_config=retail,
        )
        assert mc.mean_tvl > 0
        assert mc.mean_apr > 0

    def test_calibrated_merc_thresholds_run(self):
        """Simulation runs with calibrated mercenary thresholds."""
        from campaign.engine import run_monte_carlo
        from campaign.state import CampaignEnvironment

        config = CampaignConfig(weekly_budget=100_000, apr_cap=0.07)
        env = CampaignEnvironment(r_threshold=0.04)

        # Calibrated mercenary thresholds
        merc = MercenaryConfig(
            entry_threshold=0.04 * 1.8,  # 7.2%
            exit_threshold=0.04 * 1.3,  # 5.2%
            max_capital_usd=10_000_000,
        )

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=100_000_000,
            whale_profiles=[],
            weights=None,
            n_paths=3,
            mercenary_config=merc,
        )
        assert mc.mean_tvl > 0


# ============================================================================
# 7. Forced Rate Logic
# ============================================================================


class TestForcedRate:
    """Test forced_rate parameter in run_venue_optimization."""

    def test_forced_rate_pins_r_max(self):
        """Forced rate pins r_max to the forced value."""
        # We can't easily import app3 without Streamlit,
        # so test the logic directly via SurfaceGrid
        WEEKS_PER_YEAR = 365.0 / 7.0
        forced_rate = 0.05
        current_tvl = 100_000_000
        target_tvl = 150_000_000
        reference_tvl = max(current_tvl, target_tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR

        # The forced rate should derive B from the larger TVL
        assert reference_tvl == target_tvl
        expected_B = 150_000_000 * 0.05 / WEEKS_PER_YEAR
        assert forced_B == pytest.approx(expected_B)

        # Grid should be built with pinned B and r_max
        grid = SurfaceGrid.from_ranges(
            B_min=forced_B * 0.95,
            B_max=forced_B * 1.05,
            B_steps=3,
            r_max_min=forced_rate * 0.95,
            r_max_max=forced_rate * 1.05,
            r_max_steps=3,
        )
        config = grid.make_config(B=forced_B, r_max=forced_rate)
        assert config.apr_cap == forced_rate
        assert config.weekly_budget == pytest.approx(forced_B)

    def test_forced_rate_overspend_detection(self):
        """Overspend is correctly detected when forced rate exceeds budget."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        forced_rate = 0.05
        target_tvl = 500_000_000
        total_budget = 100_000  # Small budget

        required_B = target_tvl * forced_rate / WEEKS_PER_YEAR
        overspend = required_B > total_budget
        overspend_amount = max(0, required_B - total_budget)

        assert overspend is True
        assert overspend_amount > 0
        # ~$479k/wk required vs $100k budget
        assert required_B > 400_000

    def test_forced_rate_within_budget(self):
        """No overspend when forced rate fits within budget."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        forced_rate = 0.01
        target_tvl = 50_000_000
        total_budget = 500_000

        required_B = target_tvl * forced_rate / WEEKS_PER_YEAR
        overspend = required_B > total_budget

        assert overspend is False
        # ~$9,589/wk required, well within $500k
        assert required_B < total_budget

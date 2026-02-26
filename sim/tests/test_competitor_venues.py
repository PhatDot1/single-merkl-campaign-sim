"""
Tests for CompetitorVenue, fetch_competitor_venues (mocked), and
compute_whale_aware_r_threshold.

All tests use synthetic/mock data — no network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from campaign.agents import WhaleProfile
from campaign.data import (
    CompetitorRate,
    CompetitorVenue,
    _estimate_swap_cost_bps,
    compute_whale_aware_r_threshold,
    fetch_competitor_venues,
)

# ============================================================================
# Helpers
# ============================================================================


def _cv(
    venue: str = "test_proto",
    pool_id: str = "pool-001",
    tvl: float = 50_000_000,
    apy_total: float = 0.04,
    supply_cap: float = 0.0,
    available: float = 0.0,
    chain: str = "Ethereum",
    swap_bps: float = 2.0,
) -> CompetitorVenue:
    return CompetitorVenue(
        venue=venue,
        pool_id=pool_id,
        symbol="TESTUSDC",
        tvl_usd=tvl,
        apy_base=apy_total,
        apy_reward=0.0,
        apy_total=apy_total,
        chain=chain,
        protocol=venue,
        supply_cap_usd=supply_cap,
        available_capacity_usd=available,
        swap_cost_bps=swap_bps,
        gas_cost_usd=5.0,
    )


def _whale(position_usd: float) -> WhaleProfile:
    return WhaleProfile(
        whale_id="test_whale",
        position_usd=position_usd,
        alt_rate=0.05,
        risk_premium=0.003,
        switching_cost_usd=500.0,
        exit_delay_days=2.0,
        reentry_delay_days=7.0,
        hysteresis_band=0.005,
        whale_type="institutional",
    )


# ============================================================================
# 1. CompetitorVenue dataclass
# ============================================================================


class TestCompetitorVenueDataclass:
    def test_effective_apy_subtracts_swap_cost(self):
        """effective_apy = apy_total - swap_cost_bps/10000."""
        cv = _cv(apy_total=0.05, swap_bps=10.0)  # 10 bps = 0.001
        assert cv.effective_apy == pytest.approx(0.049)

    def test_effective_apy_never_negative(self):
        """Swap cost can't make effective_apy go below zero."""
        cv = _cv(apy_total=0.001, swap_bps=500.0)  # massive swap cost
        assert cv.effective_apy == 0.0

    def test_to_competitor_rate_roundtrip(self):
        """to_competitor_rate() preserves all CompetitorRate fields."""
        cv = _cv(apy_total=0.04, venue="morpho", pool_id="pool-xyz", tvl=100e6)
        cr = cv.to_competitor_rate()
        assert isinstance(cr, CompetitorRate)
        assert cr.venue == "morpho"
        assert cr.pool_id == "pool-xyz"
        assert cr.tvl_usd == 100e6
        assert cr.apy_total == pytest.approx(0.04)

    def test_from_competitor_rate_factory(self):
        """from_competitor_rate() upgrades a CompetitorRate to CompetitorVenue."""
        cr = CompetitorRate(
            venue="aave", pool_id="p1", symbol="USDC", tvl_usd=200e6,
            apy_base=0.03, apy_reward=0.01, apy_total=0.04,
        )
        cv = CompetitorVenue.from_competitor_rate(cr, chain="Ethereum", swap_cost_bps=2.0)
        assert cv.venue == "aave"
        assert cv.chain == "Ethereum"
        assert cv.apy_total == pytest.approx(0.04)

    def test_unlimited_cap_is_zero(self):
        """supply_cap_usd=0 and available_capacity_usd=0 mean unlimited."""
        cv = _cv(supply_cap=0.0, available=0.0)
        assert cv.supply_cap_usd == 0.0
        assert cv.available_capacity_usd == 0.0


# ============================================================================
# 2. Swap cost heuristics
# ============================================================================


class TestEstimateSwapCostBps:
    def test_ethereum_lending_is_cheap(self):
        assert _estimate_swap_cost_bps("Ethereum", "aave") == pytest.approx(2.0)

    def test_ethereum_curve_is_cheapest(self):
        assert _estimate_swap_cost_bps("Ethereum", "curve") == pytest.approx(1.0)

    def test_cross_chain_is_expensive(self):
        assert _estimate_swap_cost_bps("Arbitrum", "aave") == pytest.approx(15.0)

    def test_eth_case_insensitive(self):
        assert _estimate_swap_cost_bps("eth", "euler") == pytest.approx(2.0)

    def test_solana_is_cross_chain(self):
        assert _estimate_swap_cost_bps("Solana", "kamino") == pytest.approx(15.0)


# ============================================================================
# 3. compute_whale_aware_r_threshold
# ============================================================================


class TestComputeWhaleAwareRThreshold:
    def test_no_competitors_returns_fallback(self):
        whales = [_whale(50e6)]
        result = compute_whale_aware_r_threshold([], whales, fallback_threshold=0.04)
        assert result == pytest.approx(0.04)

    def test_all_capped_out_returns_fallback(self):
        """When all competitors are too full to absorb even 10% of the whale, fallback."""
        whale_pos = 50_000_000
        # Each competitor only has $4M left (< 10% of $50M whale)
        comps = [
            _cv(apy_total=0.06, available=4_000_000, supply_cap=50_000_000),
            _cv(apy_total=0.05, available=3_000_000, supply_cap=50_000_000),
        ]
        whales = [_whale(whale_pos)]
        result = compute_whale_aware_r_threshold(comps, whales, fallback_threshold=0.035)
        assert result == pytest.approx(0.035)

    def test_unlimited_competitors_get_full_weight(self):
        """Competitors with available_capacity_usd=0 (unlimited) get full weight."""
        comps = [
            _cv(apy_total=0.04, available=0.0, supply_cap=0.0),  # unlimited
            _cv(apy_total=0.06, available=0.0, supply_cap=0.0),  # unlimited
        ]
        whales = [_whale(50e6)]
        result = compute_whale_aware_r_threshold(comps, whales)
        # Both unlimited → weighted by position (/position = 1.0 each)
        assert 0.04 < result < 0.07  # between the two rates

    def test_single_viable_competitor(self):
        """One capped-out, one viable → result driven by viable one."""
        whale_pos = 10_000_000
        comps = [
            _cv(venue="capped", apy_total=0.08, available=500_000, supply_cap=100e6),  # < 10% → filtered
            _cv(venue="open", apy_total=0.05, available=20_000_000, supply_cap=100e6),  # viable
        ]
        whales = [_whale(whale_pos)]
        result = compute_whale_aware_r_threshold(comps, whales)
        # Only open venue survives; its effective_apy (~0.05 - 2bps) ≈ 0.0498
        assert result == pytest.approx(comps[1].effective_apy, abs=1e-4)

    def test_capacity_filter_disabled(self):
        """capacity_filter=False: capped-out competitors still get weighted."""
        whale_pos = 10_000_000
        comps = [
            _cv(venue="capped", apy_total=0.08, available=100_000, supply_cap=50e6),
            _cv(venue="open", apy_total=0.04, available=30e6, supply_cap=50e6),
        ]
        whales = [_whale(whale_pos)]
        result_filtered = compute_whale_aware_r_threshold(comps, whales, capacity_filter=True)
        result_unfiltered = compute_whale_aware_r_threshold(comps, whales, capacity_filter=False)
        # Without filter, the high-APY capped competitor pulls the result up
        assert result_unfiltered > result_filtered

    def test_no_whale_data_falls_back_to_tvl_weighted(self):
        """Empty whale list → TVL-weighted average (no capacity filtering)."""
        comps = [
            _cv(apy_total=0.06, tvl=100e6, available=10e6),
            _cv(apy_total=0.04, tvl=100e6, available=10e6),
        ]
        result = compute_whale_aware_r_threshold(comps, whale_profiles=[])
        # TVL-weighted of effective APYs (both TVL equal → simple average)
        expected = (comps[0].effective_apy + comps[1].effective_apy) / 2
        assert result == pytest.approx(expected, abs=1e-4)

    def test_result_capped_at_8_percent(self):
        """Result is capped at 8% regardless of competitor rates."""
        comps = [_cv(apy_total=0.20, available=0.0, swap_bps=0.0)]
        result = compute_whale_aware_r_threshold(comps, [_whale(1e6)])
        assert result <= 0.08

    def test_swap_cost_reduces_effective_threshold(self):
        """swap_cost_annualised=True should give lower result than False."""
        comps = [_cv(apy_total=0.05, swap_bps=50.0, available=0.0)]  # 50bps = 0.005
        whales = [_whale(5e6)]
        with_cost = compute_whale_aware_r_threshold(comps, whales, swap_cost_annualised=True)
        without_cost = compute_whale_aware_r_threshold(comps, whales, swap_cost_annualised=False)
        assert with_cost < without_cost


# ============================================================================
# 4. fetch_competitor_venues (mocked DeFiLlama)
# ============================================================================


_MOCK_DEFILLAMA_RESPONSE = {
    "data": [
        {
            "pool": "pool-aave-pyusd",
            "project": "aave-v3",
            "symbol": "PYUSD",
            "chain": "Ethereum",
            "tvlUsd": 100_000_000,
            "apyBase": 3.5,
            "apyReward": 0.5,
            "underlyingTokens": [],
        },
        {
            "pool": "pool-morpho-pyusd",
            "project": "morpho",
            "symbol": "PYUSD",
            "chain": "Ethereum",
            "tvlUsd": 50_000_000,
            "apyBase": 4.0,
            "apyReward": 0.0,
            "underlyingTokens": [],
        },
        {
            "pool": "pool-unrelated",
            "project": "some-dex",
            "symbol": "USDC-WETH",  # Different asset — should be excluded
            "chain": "Ethereum",
            "tvlUsd": 200_000_000,
            "apyBase": 2.0,
            "apyReward": 0.0,
            "underlyingTokens": [],
        },
        {
            "pool": "pool-dust",
            "project": "tiny-protocol",
            "symbol": "PYUSD",
            "chain": "Ethereum",
            "tvlUsd": 100_000_000,
            "apyBase": 0.05,  # < 0.1% total → excluded as dust
            "apyReward": 0.0,
            "underlyingTokens": [],
        },
        {
            "pool": "pool-cross-chain",
            "project": "morpho",
            "symbol": "PYUSD",
            "chain": "Arbitrum",
            "tvlUsd": 20_000_000,
            "apyBase": 4.5,
            "apyReward": 0.0,
            "underlyingTokens": [],
        },
    ]
}


def _mock_get_no_own_venues(*args, **kwargs):
    """Mock requests.get returning DeFiLlama-like response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _MOCK_DEFILLAMA_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestFetchCompetitorVenues:
    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    @patch("campaign.data.fetch_competitor_venues.__wrapped__", None, create=True)
    def test_returns_competitor_venue_instances(self, mock_get):
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        assert all(isinstance(v, CompetitorVenue) for v in venues)

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_filters_unrelated_asset(self, mock_get):
        """Pools without PYUSD in symbol should be excluded."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        symbols = [v.symbol for v in venues]
        assert all("PYUSD" in s for s in symbols)

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_filters_dust_pools(self, mock_get):
        """Pools with total APY < 0.1% should be excluded."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        assert all(v.apy_total >= 0.001 for v in venues)

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_chain_propagated(self, mock_get):
        """Chain field should be populated from DeFiLlama metadata."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        chains = {v.chain for v in venues}
        assert "Ethereum" in chains

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_cross_chain_gets_higher_swap_cost(self, mock_get):
        """Arbitrum venue should have higher swap_cost_bps than Ethereum venue."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        eth_venues = [v for v in venues if v.chain == "Ethereum"]
        arb_venues = [v for v in venues if v.chain == "Arbitrum"]
        if eth_venues and arb_venues:
            assert arb_venues[0].swap_cost_bps > eth_venues[0].swap_cost_bps

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_apy_converted_to_decimal(self, mock_get):
        """DeFiLlama returns percentage (3.5), output should be decimal (0.035)."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        for v in venues:
            assert v.apy_total < 1.0, f"APY not decimal: {v.apy_total}"

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_supply_cap_defaults_to_zero_unlimited(self, mock_get):
        """DeFiLlama doesn't provide caps — should default to 0 (unlimited)."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        assert all(v.supply_cap_usd == 0.0 for v in venues)

    @patch("campaign.data.requests.get", side_effect=_mock_get_no_own_venues)
    def test_to_competitor_rate_works_on_all_entries(self, mock_get):
        """All CompetitorVenue entries should be downcastable to CompetitorRate."""
        with patch("campaign.data.get_all_venue_addresses", return_value=set()):
            venues = fetch_competitor_venues("PYUSD", exclude_all_own_venues=False)
        for v in venues:
            cr = v.to_competitor_rate()
            assert isinstance(cr, CompetitorRate)
            assert cr.apy_total == pytest.approx(v.apy_total)

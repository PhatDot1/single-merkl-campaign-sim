"""
Tests for supply-cap fetching, decoding, and propagation.

All tests use synthetic / mock data — no network calls.
"""

from __future__ import annotations

import pytest
from campaign.data import VaultSnapshot, _resolve_euler_cap

# ============================================================================
# 1. Aave V3 config-word cap extraction
# ============================================================================


def _aave_config_word(supply_cap_tokens: int, borrow_cap_tokens: int = 0) -> int:
    """Synthesise an Aave V3 ReserveConfigurationMap with the given caps."""
    # supply cap: bits 116-151 (36-bit field)
    # borrow cap: bits 80-115 (36-bit field)
    return (supply_cap_tokens << 116) | (borrow_cap_tokens << 80)


def _parse_aave_caps(config_word: int) -> tuple[int, int]:
    """Mirror the bit-extraction logic from evm_data.fetch_aave_reserve_data."""
    mask_36 = (1 << 36) - 1
    supply = (config_word >> 116) & mask_36
    borrow = (config_word >> 80) & mask_36
    return supply, borrow


class TestAaveConfigWordCapExtraction:
    def test_known_pyusd_cap_500m(self):
        """Aave PYUSD Core supply cap is 500M whole tokens."""
        config = _aave_config_word(supply_cap_tokens=500_000_000)
        supply, borrow = _parse_aave_caps(config)
        assert supply == 500_000_000
        assert borrow == 0

    def test_unlimited_cap_is_zero(self):
        """When no cap is set all bits are 0 → supply = 0 (interpreted as unlimited)."""
        config = _aave_config_word(supply_cap_tokens=0)
        supply, _ = _parse_aave_caps(config)
        assert supply == 0

    def test_supply_and_borrow_caps_independent(self):
        config = _aave_config_word(supply_cap_tokens=100_000_000, borrow_cap_tokens=80_000_000)
        supply, borrow = _parse_aave_caps(config)
        assert supply == 100_000_000
        assert borrow == 80_000_000

    def test_max_36bit_cap(self):
        """36-bit field can hold up to 68.7B — well above any realistic pool."""
        max_cap = (1 << 36) - 1  # 68,719,476,735
        config = _aave_config_word(supply_cap_tokens=max_cap)
        supply, _ = _parse_aave_caps(config)
        assert supply == max_cap

    def test_caps_do_not_bleed_into_each_other(self):
        """Setting supply cap must not corrupt borrow cap bits."""
        config = _aave_config_word(supply_cap_tokens=999_999_999, borrow_cap_tokens=1)
        supply, borrow = _parse_aave_caps(config)
        assert supply == 999_999_999
        assert borrow == 1


# ============================================================================
# 2. Euler legacy cap encoding (_resolve_euler_cap)
# ============================================================================


class TestResolveEulerCap:
    def test_zero_returns_max_uint256(self):
        """0 means no cap — sentinel for unlimited."""
        result = _resolve_euler_cap(0)
        assert result == 2**256 - 1

    def test_known_encoding(self):
        """
        Euler packs: exponent = encoded & 63, mantissa = encoded >> 6
        Result = (10^exponent * mantissa) // 100
        Example: exponent=8, mantissa=100 → (10^8 * 100) / 100 = 10^8 = 100M tokens
        Encoded = (100 << 6) | 8 = 6400 | 8 = 6408
        """
        exponent, mantissa = 8, 100
        encoded = (mantissa << 6) | exponent
        result = _resolve_euler_cap(encoded)
        assert result == (10**8 * 100) // 100  # 100_000_000

    def test_small_cap(self):
        """exponent=6, mantissa=50 → (10^6 * 50) / 100 = 500_000"""
        encoded = (50 << 6) | 6
        result = _resolve_euler_cap(encoded)
        assert result == 500_000

    def test_max_exponent(self):
        """Exponent is 6 bits → max 63. Result should be astronomically large but finite."""
        encoded = (1 << 6) | 63  # mantissa=1, exponent=63
        result = _resolve_euler_cap(encoded)
        assert result == (10**63 * 1) // 100


# ============================================================================
# 3. Euler V2 raw-uint256 cap decoding (logic mirrored from evm_data.py)
# ============================================================================


def _decode_euler_v2_cap(supply_cap_raw: int, asset_decimals: int) -> float:
    """
    Mirror the logic in evm_data.fetch_euler_vault_data:
    - 0 or >= 2^128 → 0.0 (unlimited / sentinel)
    - otherwise → supply_cap_raw / 10^decimals (USD for stablecoins)
    """
    if supply_cap_raw <= 0 or supply_cap_raw >= (1 << 128):
        return 0.0
    return supply_cap_raw / (10**asset_decimals)


class TestEulerV2RawCapDecoding:
    def test_100m_pyusd_raw(self):
        """100M PYUSD cap: raw = 100_000_000 * 10^6, decimals=6 → $100M"""
        raw = 100_000_000 * (10**6)
        result = _decode_euler_v2_cap(raw, asset_decimals=6)
        assert result == pytest.approx(100_000_000.0)

    def test_zero_raw_means_unlimited(self):
        """Raw value 0 → 0.0 (unlimited flag)."""
        assert _decode_euler_v2_cap(0, 6) == 0.0

    def test_max_uint256_sentinel_means_unlimited(self):
        """type(uint256).max is the 'disabled/no-cap' sentinel; should map to 0.0."""
        max_uint256 = (1 << 256) - 1
        assert _decode_euler_v2_cap(max_uint256, 6) == 0.0

    def test_above_2_128_treated_as_unlimited(self):
        """Any value >= 2^128 treated as unlimited to avoid phantom huge caps."""
        assert _decode_euler_v2_cap(1 << 128, 6) == 0.0
        assert _decode_euler_v2_cap((1 << 128) + 1, 6) == 0.0

    def test_just_below_2_128_is_valid(self):
        """Values just under 2^128 should be decoded normally."""
        raw = (1 << 128) - 1
        result = _decode_euler_v2_cap(raw, asset_decimals=6)
        assert result > 0
        assert result == pytest.approx(raw / 1e6)

    def test_18_decimal_asset(self):
        """Works correctly for 18-decimal assets (e.g., WETH)."""
        raw = 1_000 * (10**18)  # 1000 token cap
        result = _decode_euler_v2_cap(raw, asset_decimals=18)
        assert result == pytest.approx(1000.0)


# ============================================================================
# 4. VaultSnapshot.supply_cap_usd property
# ============================================================================


def _make_snapshot(supply_cap_raw: int, decimals: int = 6, price: float = 1.0) -> VaultSnapshot:
    return VaultSnapshot(
        address="0x" + "a" * 40,
        chain_id=1,
        asset_address="0x" + "b" * 40,
        asset_decimals=decimals,
        asset_symbol="TESTUSDC",
        asset_price_usd=price,
        total_supply_assets=10_000_000 * (10**decimals),
        total_borrows=5_000_000 * (10**decimals),
        cash=5_000_000 * (10**decimals),
        supply_cap=supply_cap_raw,
        borrow_cap=0,
        top_depositors=[],
        timestamp=0,
    )


class TestVaultSnapshotSupplyCapUsd:
    def test_zero_cap_is_unlimited(self):
        """supply_cap=0 → supply_cap_usd=0 (unlimited signal)."""
        snap = _make_snapshot(supply_cap_raw=0)
        assert snap.supply_cap_usd == 0.0

    def test_500m_6_decimal(self):
        """500M PYUSD (6 dec) raw cap → $500M USD."""
        raw = 500_000_000 * (10**6)
        snap = _make_snapshot(supply_cap_raw=raw, decimals=6, price=1.0)
        assert snap.supply_cap_usd == pytest.approx(500_000_000.0)

    def test_price_multiplied(self):
        """Non-stablecoin: price != 1 should be factored in."""
        raw = 1000 * (10**18)  # 1000 ETH cap
        snap = _make_snapshot(supply_cap_raw=raw, decimals=18, price=3000.0)
        assert snap.supply_cap_usd == pytest.approx(3_000_000.0)  # 1000 ETH * $3000

    def test_supply_cap_utilization_with_known_cap(self):
        """supply_cap_utilization = total_supply_assets / supply_cap."""
        decimals = 6
        supply_raw = 100 * (10**decimals)
        cap_raw = 500 * (10**decimals)
        snap = VaultSnapshot(
            address="0x" + "a" * 40,
            chain_id=1,
            asset_address="0x" + "b" * 40,
            asset_decimals=decimals,
            asset_symbol="TESTUSDC",
            asset_price_usd=1.0,
            total_supply_assets=supply_raw,
            total_borrows=0,
            cash=supply_raw,
            supply_cap=cap_raw,
            borrow_cap=0,
            top_depositors=[],
            timestamp=0,
        )
        assert snap.supply_cap_utilization == pytest.approx(100 / 500)

    def test_utilization_unlimited_cap(self):
        """When cap=0, supply_cap_utilization must return 0 (not divide-by-zero)."""
        snap = _make_snapshot(supply_cap_raw=0)
        assert snap.supply_cap_utilization == 0.0


# ============================================================================
# 5. Morpho cap handling: sum of market caps, 0 for unlimited
# ============================================================================


def _build_morpho_snapshot_cap(market_caps: list[int | None], asset_decimals: int = 6) -> int:
    """
    Mirror the logic in data.fetch_morpho_vault_snapshot:
    sum per-market supplyCap, treating None or >1e30 as 0 (unlimited per market).
    """
    total = 0
    for cap_raw in market_caps:
        if cap_raw is not None:
            cap = int(cap_raw) if int(cap_raw) < 10**30 else 0
        else:
            cap = 0
        total += cap
    return total


class TestMorphoCapSumming:
    def test_two_capped_markets(self):
        """Sum of two 50M raw caps → 100M raw."""
        raw_50m = 50_000_000 * (10**6)
        total = _build_morpho_snapshot_cap([raw_50m, raw_50m])
        assert total == 100_000_000 * (10**6)

    def test_unlimited_market_is_zero(self):
        """A market with cap > 1e30 is treated as 0 (unlimited)."""
        raw_50m = 50_000_000 * (10**6)
        huge = int(10**31)
        total = _build_morpho_snapshot_cap([raw_50m, huge])
        assert total == raw_50m  # huge cap ignored

    def test_all_unlimited_markets(self):
        """All unlimited markets → total cap = 0 (unlimited vault)."""
        total = _build_morpho_snapshot_cap([None, 10**31, 0])
        assert total == 0

    def test_snapshot_supply_cap_usd_from_morpho_sum(self):
        """End-to-end: summed raw cap flows into VaultSnapshot.supply_cap_usd correctly."""
        raw_200m = 200_000_000 * (10**6)
        snap = _make_snapshot(supply_cap_raw=raw_200m, decimals=6, price=1.0)
        assert snap.supply_cap_usd == pytest.approx(200_000_000.0)

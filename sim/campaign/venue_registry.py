"""
Canonical venue registry — single source of truth for all pool/vault addresses.

Every venue in the optimizer references this registry. No addresses are
hardcoded in dashboard/app.py, evm_data.py, or kamino_data.py.

This file is the ONLY place you edit when adding/removing venues.

Usage:
    from campaign.venue_registry import VENUE_REGISTRY, get_venue, get_program_venues
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VenueRecord:
    """Immutable record for a single venue (pool/vault)."""

    # Identity
    pool_id: str  # Unique key (e.g. "pyusd-aave-v3-core")
    name: str  # Display name (e.g. "AAVE Core Market PYUSD")
    program: str  # Budget program grouping (e.g. "PYUSD", "RLUSD Core")
    asset: str  # Tracked stablecoin symbol

    # Protocol
    protocol: str  # "aave", "euler", "morpho", "curve", "kamino"
    protocol_type: str  # "lending", "dex", "kvault"
    chain: str  # "ethereum" or "solana"

    # Addresses
    address: str  # Primary contract/reserve address
    underlying_asset: str  # Underlying token address/mint
    a_token_address: str = ""  # aToken / eToken / shares mint (for whale scanning)
    variable_debt_token: str = ""  # Variable debt token (Aave)
    vault_address: str = ""  # Morpho V2 vault, Euler vault, Kamino vault pubkey

    # Aave-specific
    aave_market: str = ""  # "core" or "horizon"
    aave_pool_contract: str = ""  # Pool proxy contract address

    # Kamino-specific
    kamino_market_name: str = ""  # "main", "jlp", "maple"
    kamino_market_pubkey: str = ""  # Lending market pubkey
    kamino_vault_pubkey: str = ""  # kVault pubkey (for Earn/CLMM)

    # DeFiLlama
    defillama_project: str = ""  # DeFiLlama project name for APY/competitor lookup
    defillama_pool_id: str = ""  # DeFiLlama pool UUID for historical chart data

    # On-chain caps (fetched live, optional static fallback)
    supply_cap: float = 0.0  # USD (0 = unlimited / unknown)
    borrow_cap: float = 0.0  # USD (0 = unlimited / unknown)

    # Campaign targets (human-judgment inputs)
    current_tvl: float = 0.0
    target_tvl: float = 0.0
    target_util: float = 0.0
    budget_min: float = 0.0
    budget_max: float = 0.0
    r_max_range: tuple[float, float] = (0.03, 0.10)

    # UX
    ux_url: str = ""


# ============================================================================
# AAVE POOL CONTRACTS (Core vs Horizon — separate proxy contracts)
# ============================================================================

AAVE_CORE_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_HORIZON_POOL = "0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8"

# ============================================================================
# KNOWN STABLECOIN ADDRESSES (Ethereum)
# ============================================================================

PYUSD_ETH = "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8"
RLUSD_ETH = "0x8292Bb45bf1Ee4d140127049757C0C38e47a8A75"
# Note: Aave uses a different address encoding for RLUSD:
RLUSD_AAVE = "0x8292bb45bf1ee4d140127049757c2e0ff06317ed"
USDC_ETH = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"

# ============================================================================
# KNOWN SOLANA MINTS
# ============================================================================

PYUSD_SOL = "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo"
USDC_SOL = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# ============================================================================
# KAMINO MARKET PUBKEYS
# ============================================================================

KAMINO_MAIN_MARKET = "7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF"
KAMINO_JLP_MARKET = "DxXdAyU3kCjnyggvHmY5nAwg5cRbbmdyX3npfDMjjMek"
KAMINO_ALTCOIN_MARKET = "ByYiZxp8QrdN9qbdtaAiePN8AAr3qvTPppNJDpf5DVJ5"
KAMINO_MAPLE_MARKET = "6WEGfej9B9wjxRs6t4BYpb9iCXd8CpTpJ8fVSNzHCC5y"

# ============================================================================
# VENUE REGISTRY
# ============================================================================

VENUE_REGISTRY: dict[str, VenueRecord] = {}


# ============================================================================
# r_threshold CONFIGURATION PER PROGRAM
# ============================================================================
# Mode A: "auto"    — use DeFiLlama competitors for this asset (PYUSD has many)
# Mode B: "manual"  — operator sets r_threshold directly
# Mode C: "blended" — stablecoin-class benchmark (USDC/USDT/DAI weighted avg)
#                     minus a friction discount for conversion


@dataclass(frozen=True)
class RThresholdConfig:
    """Per-program r_threshold calibration settings."""

    mode: str = "auto"  # "auto", "manual", "blended"
    manual_value: float = 0.04  # Used only when mode="manual"
    # Blended mode settings
    proxy_assets: tuple[str, ...] = ("USDC", "USDT", "DAI", "USDS")
    friction_discount: float = 0.005  # Swap/bridge cost discount
    min_pool_tvl: float = 50_000_000  # Only large pools for benchmark
    top_n_pools: int = 20  # Number of pools for TVL-weighted avg


PROGRAM_R_THRESHOLD_CONFIG: dict[str, RThresholdConfig] = {
    "RLUSD Core": RThresholdConfig(
        mode="blended",
        proxy_assets=("USDC", "USDT", "DAI", "USDS"),
        friction_discount=0.005,  # 0.5% for RLUSD→USDC swap cost
    ),
    "RLUSD Horizon": RThresholdConfig(
        mode="blended",
        proxy_assets=("USDC", "USDT", "DAI", "USDS"),
        friction_discount=0.005,
    ),
    "PYUSD": RThresholdConfig(
        mode="auto",  # PYUSD has many external competitors
    ),
}


# ============================================================================
# GLOBAL r_max CEILING & PER-PROTOCOL DEFAULTS
# ============================================================================

GLOBAL_R_MAX_CEILING = 0.08  # 8% absolute maximum incentive APR

PROTOCOL_R_MAX_DEFAULTS: dict[str, tuple[float, float]] = {
    "aave": (0.02, 0.06),  # Conservative, large pools
    "euler": (0.03, 0.07),  # Slightly higher, smaller pools
    "morpho": (0.03, 0.08),  # Medium
    "curve": (0.02, 0.06),  # DEX, should not be too high
    "kamino": (0.02, 0.06),  # Similar to Aave
}


def _r(v: VenueRecord) -> VenueRecord:
    """Register and return."""
    VENUE_REGISTRY[v.pool_id] = v
    return v


# ── RLUSD Core Program ──

_r(
    VenueRecord(
        pool_id="rlusd-aave-v3-core",
        name="AAVE RLUSD Core",
        program="RLUSD Core",
        asset="RLUSD",
        protocol="aave",
        protocol_type="lending",
        chain="ethereum",
        address=RLUSD_AAVE,
        underlying_asset=RLUSD_AAVE,
        a_token_address="0xFa82580c16A31D0c1bC632A36F82e83EfEF3Eec0",
        aave_market="core",
        aave_pool_contract=AAVE_CORE_POOL,
        defillama_project="aave-v3",
        defillama_pool_id="85fc6934-c94d-4ebe-9c60-66beb363669f",
        current_tvl=600_000_000,
        target_tvl=600_000_000,
        target_util=0.40,
        budget_min=300_000,
        budget_max=700_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://app.aave.com/reserve-overview/?underlyingAsset=0x8292bb45bf1ee4d140127049757c2e0ff06317ed&marketName=proto_mainnet_v3",
    )
)

_r(
    VenueRecord(
        pool_id="rlusd-euler-v2-sentora",
        name="Euler Sentora RLUSD",
        program="RLUSD Core",
        asset="RLUSD",
        protocol="euler",
        protocol_type="lending",
        chain="ethereum",
        address="0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2",
        underlying_asset=RLUSD_AAVE,
        vault_address="0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2",
        defillama_project="euler-v2",
        defillama_pool_id="73e933a7-73b2-43ec-b1e9-d5d1d42ce2de",
        current_tvl=190_000_000,
        target_tvl=190_000_000,
        target_util=0.385,
        budget_min=100_000,
        budget_max=350_000,
        r_max_range=(0.04, 0.10),
        ux_url="https://app.euler.finance/vault/0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2?network=ethereum",
    )
)

_r(
    VenueRecord(
        pool_id="rlusd-curve-v2-rlusd-usdc",
        name="Curve RLUSD-USDC",
        program="RLUSD Core",
        asset="RLUSD",
        protocol="curve",
        protocol_type="dex",
        chain="ethereum",
        address="0xD001aE433f254283FeCE51d4ACcE8c53263aa186",
        underlying_asset=RLUSD_AAVE,
        defillama_project="curve-dex",
        defillama_pool_id="e91e23af-9099-45d9-8ba5-ea5b4638e453",
        current_tvl=75_000_000,
        target_tvl=75_000_000,
        target_util=0.50,
        budget_min=40_000,
        budget_max=150_000,
        r_max_range=(0.04, 0.10),
        ux_url="https://www.curve.finance/dex/ethereum/pools/factory-stable-ng-327/deposit",
    )
)

# ── RLUSD Horizon Program ──

_r(
    VenueRecord(
        pool_id="rlusd-aave-v3-horizon",
        name="AAVE Horizon RLUSD",
        program="RLUSD Horizon",
        asset="RLUSD",
        protocol="aave",
        protocol_type="lending",
        chain="ethereum",
        address=RLUSD_AAVE,
        underlying_asset=RLUSD_AAVE,
        a_token_address="0xe3190143eb552456f88464662f0c0c4ac67a77eb",
        aave_market="horizon",
        aave_pool_contract=AAVE_HORIZON_POOL,
        defillama_project="aave-v3",
        defillama_pool_id="98d07333-f5e4-4a48-8061-cfb4b73ccf79",
        current_tvl=221_500_000,
        target_tvl=221_500_000,
        target_util=0.60,
        budget_min=100_000,
        budget_max=250_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://app.aave.com/reserve-overview/?underlyingAsset=0x8292bb45bf1ee4d140127049757c2e0ff06317ed&marketName=proto_horizon_v3",
    )
)

# ── PYUSD Program ──

_r(
    VenueRecord(
        pool_id="pyusd-aave-v3-core",
        name="AAVE Core Market PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="aave",
        protocol_type="lending",
        chain="ethereum",
        address=PYUSD_ETH,
        underlying_asset=PYUSD_ETH,
        a_token_address="0x0c0d01abf3e6adfca0989ebba9d6e85dd58eab1e",
        aave_market="core",
        aave_pool_contract=AAVE_CORE_POOL,
        defillama_project="aave-v3",
        defillama_pool_id="d118f505-e75f-4152-bad3-49a2dc7482bf",
        current_tvl=400_000_000,
        target_tvl=400_000_000,
        target_util=0.60,
        budget_min=200_000,
        budget_max=500_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://app.aave.com/reserve-overview/?underlyingAsset=0x6c3ea9036406852006290770bedfcaba0e23a0e8&marketName=proto_mainnet_v3",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-kamino-main-market",
        name="Kamino Main Market PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="kamino",
        protocol_type="lending",
        chain="solana",
        address="2gc9Dm1eB6UgVYFBUN9bWks6Kes9PbWSaPaa9DqyvEiN",
        underlying_asset=PYUSD_SOL,
        kamino_market_name="main",
        kamino_market_pubkey=KAMINO_MAIN_MARKET,
        defillama_project="kamino-lend",
        defillama_pool_id="eaece65e-ffb2-4631-a95e-7f267cb2f1ba",
        current_tvl=461_500_000,
        target_tvl=461_500_000,
        target_util=0.315,
        budget_min=100_000,
        budget_max=300_000,
        r_max_range=(0.02, 0.06),
        ux_url="https://kamino.com/borrow/reserve/7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF/2gc9Dm1eB6UgVYFBUN9bWks6Kes9PbWSaPaa9DqyvEiN",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-kamino-maple-market",
        name="Kamino Maple Market PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="kamino",
        protocol_type="lending",
        chain="solana",
        address="92qeAka3ZzCGPfJriDXrE7tiNqfATVCAM6ZjjctR3TrS",
        underlying_asset=PYUSD_SOL,
        kamino_market_name="maple",
        kamino_market_pubkey=KAMINO_MAPLE_MARKET,
        defillama_project="kamino-lend",
        current_tvl=50_000_000,
        target_tvl=50_000_000,
        target_util=0.40,
        budget_min=20_000,
        budget_max=100_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://kamino.com/borrow/reserve/6WEGfej9B9wjxRs6t4BYpb9iCXd8CpTpJ8fVSNzHCC5y/92qeAka3ZzCGPfJriDXrE7tiNqfATVCAM6ZjjctR3TrS",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-kamino-jlp-market",
        name="Kamino JLP Market PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="kamino",
        protocol_type="lending",
        chain="solana",
        address="FswUCVjvfAuzHCgPDF95eLKscGsLHyJmD6hzkhq26CLe",
        underlying_asset=PYUSD_SOL,
        kamino_market_name="jlp",
        kamino_market_pubkey=KAMINO_JLP_MARKET,
        defillama_project="kamino-lend",
        current_tvl=80_000_000,
        target_tvl=80_000_000,
        target_util=0.50,
        budget_min=30_000,
        budget_max=120_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://kamino.com/borrow/reserve/DxXdAyU3kCjnyggvHmY5nAwg5cRbbmdyX3npfDMjjMek/FswUCVjvfAuzHCgPDF95eLKscGsLHyJmD6hzkhq26CLe",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-kamino-earn-vault",
        name="Kamino Earn Vault PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="kamino",
        protocol_type="kvault",
        chain="solana",
        address="A2wsxhA7pF4B2UKVfXocb6TAAP9ipfPJam6oMKgDE5BK",
        underlying_asset=PYUSD_SOL,
        kamino_vault_pubkey="A2wsxhA7pF4B2UKVfXocb6TAAP9ipfPJam6oMKgDE5BK",
        defillama_project="kamino",
        current_tvl=352_000_000,
        target_tvl=352_000_000,
        target_util=0.42,
        budget_min=150_000,
        budget_max=450_000,
        r_max_range=(0.03, 0.08),
        ux_url="https://app.kamino.finance/earn/A2wsxhA7pF4B2UKVfXocb6TAAP9ipfPJam6oMKgDE5BK",
    )
)

# NOTE: Kamino CLMM vault pubkey not in registry doc — load from env or add when known
_CLMM_PUBKEY = os.environ.get("KAMINO_PYUSD_CLMM_VAULT_PUBKEY", "")
_r(
    VenueRecord(
        pool_id="pyusd-kamino-clmm",
        name="Kamino CLMM PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="kamino",
        protocol_type="kvault",
        chain="solana",
        address=_CLMM_PUBKEY,
        underlying_asset=PYUSD_SOL,
        kamino_vault_pubkey=_CLMM_PUBKEY,
        defillama_project="kamino-liquidity",
        current_tvl=30_000_000,
        target_tvl=30_000_000,
        target_util=0.50,
        budget_min=20_000,
        budget_max=60_000,
        r_max_range=(0.04, 0.10),
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-euler-v2-sentora",
        name="Euler Sentora PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="euler",
        protocol_type="lending",
        chain="ethereum",
        address="0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5",
        underlying_asset=PYUSD_ETH,
        vault_address="0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5",
        defillama_project="euler-v2",
        defillama_pool_id="fa55aa2b-e244-4ce4-ab00-9e96b39df32b",
        current_tvl=250_000_000,
        target_tvl=250_000_000,
        target_util=0.52,
        budget_min=200_000,
        budget_max=400_000,
        r_max_range=(0.04, 0.10),
        ux_url="https://app.euler.finance/vault/0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5?network=ethereum",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-morpho-sentora",
        name="Morpho PYUSD",
        program="PYUSD",
        asset="PYUSD",
        protocol="morpho",
        protocol_type="lending",
        chain="ethereum",
        address="0xb576765fB15505433aF24FEe2c0325895C559FB2",
        underlying_asset=PYUSD_ETH,
        vault_address="0xb576765fB15505433aF24FEe2c0325895C559FB2",
        defillama_project="morpho",
        defillama_pool_id="699f25fe-09f4-4f82-8f58-baa5b0af8fa4",
        current_tvl=195_000_000,
        target_tvl=100_000_000,
        target_util=0.90,
        budget_min=50_000,
        budget_max=200_000,
        r_max_range=(0.04, 0.12),
        ux_url="https://app.morpho.org/ethereum/vault/0xb576765fB15505433aF24FEe2c0325895C559FB2/sentora-pyusd-main",
    )
)

_r(
    VenueRecord(
        pool_id="pyusd-curve-v2-pyusd-usdc",
        name="Curve PYUSD-USDC",
        program="PYUSD",
        asset="PYUSD",
        protocol="curve",
        protocol_type="dex",
        chain="ethereum",
        address="0x383E6b4437b59fff47B619CBA855CA29342A8559",
        underlying_asset=PYUSD_ETH,
        defillama_project="curve-dex",
        defillama_pool_id="14681aee-05c9-4733-acd0-7b2c84616209",
        current_tvl=30_000_000,
        target_tvl=30_000_000,
        target_util=0.50,
        budget_min=15_000,
        budget_max=60_000,
        r_max_range=(0.03, 0.10),
        ux_url="https://www.curve.finance/dex/ethereum/pools/factory-stable-ng-43/deposit",
    )
)


# ============================================================================
# PROGRAM DEFINITIONS (budget envelopes)
# ============================================================================

PROGRAM_BUDGETS = {
    "RLUSD Core": 869_700,
    "RLUSD Horizon": 180_500,
    "PYUSD": 1_300_000,
}


# ============================================================================
# QUERY HELPERS
# ============================================================================


def get_venue(pool_id: str) -> VenueRecord:
    """Get a venue by pool_id. Errors loudly if not found."""
    if pool_id not in VENUE_REGISTRY:
        raise KeyError(f"Unknown pool_id '{pool_id}'. Known: {sorted(VENUE_REGISTRY.keys())}")
    return VENUE_REGISTRY[pool_id]


def get_program_venues(program: str) -> list[VenueRecord]:
    """Get all venues in a program, ordered by registry insertion."""
    return [v for v in VENUE_REGISTRY.values() if v.program == program]


def get_all_venue_addresses(asset: str) -> set[str]:
    """
    Get ALL addresses for venues tracking a given asset.
    Used to EXCLUDE our own venues from competitor rate queries.
    Returns lowercase set of all address variants.
    """
    addrs = set()
    for v in VENUE_REGISTRY.values():
        if v.asset != asset:
            continue
        for addr in [
            v.address,
            v.a_token_address,
            v.vault_address,
            v.underlying_asset,
            v.kamino_vault_pubkey,
        ]:
            if addr:
                addrs.add(addr.lower())
    return addrs


def venue_to_dashboard_dict(v: VenueRecord) -> dict:
    """Convert a VenueRecord to the dict format expected by dashboard/app.py."""
    # Apply global r_max ceiling
    r_lo, r_hi = v.r_max_range
    r_hi = min(r_hi, GLOBAL_R_MAX_CEILING)
    d = {
        "name": v.name,
        "pool_id": v.pool_id,
        "asset": v.asset,
        "protocol": v.protocol,
        "chain": v.chain.capitalize() if v.chain == "ethereum" else v.chain.capitalize(),
        "current_tvl": v.current_tvl,
        "target_tvl": v.target_tvl,
        "target_util": v.target_util,
        "budget_min": v.budget_min,
        "budget_max": v.budget_max,
        "r_max_range": (r_lo, r_hi),
        "defillama_project": v.defillama_project,
    }
    if v.defillama_pool_id:
        d["defillama_pool_id"] = v.defillama_pool_id
    if v.supply_cap > 0:
        d["supply_cap"] = v.supply_cap
    if v.borrow_cap > 0:
        d["borrow_cap"] = v.borrow_cap
    if v.aave_market:
        d["aave_market"] = v.aave_market
    if v.vault_address:
        d["vault_address"] = v.vault_address
    if v.kamino_market_name:
        d["kamino_market_name"] = v.kamino_market_name
    if v.kamino_vault_pubkey:
        d["kamino_vault_pubkey"] = v.kamino_vault_pubkey
    return d


def get_program_dashboard_config() -> dict:
    """
    Build the full PROGRAMS dict for dashboard/app.py from the registry.

    Returns:
        {program_name: {"total_budget": ..., "venues": [...]}}
    """
    programs = {}
    for prog_name, budget in PROGRAM_BUDGETS.items():
        venues = get_program_venues(prog_name)
        programs[prog_name] = {
            "total_budget": budget,
            "venues": [venue_to_dashboard_dict(v) for v in venues],
        }
    return programs

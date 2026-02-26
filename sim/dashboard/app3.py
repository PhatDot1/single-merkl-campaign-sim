"""
Campaign Optimizer Dashboard

Production configuration:
- All RPC URLs loaded from .env (ALCHEMY_ETH_RPC_URL, HELIUS_SOLANA_RPC_URL)
- Kamino vault pubkeys from .env (KAMINO_PYUSD_EARN_VAULT_PUBKEY, etc.)
- Aave Horizon as separate pool (0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8)
- Whale fetching auto-triggered at optimization time via Alchemy / Helius
- Zero fallbacks to static/empty whale data — errors loudly on failure

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure the sim/ directory is on sys.path so `from campaign.X` resolves
_sim_dir = str(Path(__file__).resolve().parent.parent)
if _sim_dir not in sys.path:
    sys.path.insert(0, _sim_dir)

# Load .env at import time (before any campaign imports that read env vars)
# Search upward from this file's directory to find all .env files
# (the root .env has DUNE_API_KEY, floatVmaxVhybrid/.env has RPC keys)
_this_dir = Path(__file__).resolve().parent
for _p in [_this_dir, _this_dir.parent, _this_dir.parent.parent, _this_dir.parent.parent.parent]:
    _env_file = _p / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)  # don't clobber already-loaded keys

from campaign.agents import (  # noqa: E402
    APYSensitiveConfig,
    MercenaryConfig,
    RetailDepositorConfig,
    WhaleProfile,
)
from campaign.base_apy import BaseAPYResult, fetch_all_base_apys  # noqa: E402
from campaign.data import (  # noqa: E402
    compute_r_threshold,
    fetch_competitor_rates,
    fetch_stablecoin_class_benchmark,
)
from campaign.engine import LossWeights  # noqa: E402
from campaign.optimizer import SurfaceGrid, SurfaceResult, optimize_surface  # noqa: E402
from campaign.state import CampaignEnvironment  # noqa: E402
from campaign.venue_registry import (  # noqa: E402
    GLOBAL_R_MAX_CEILING,
    PROGRAM_R_THRESHOLD_CONFIG,
    PROTOCOL_R_MAX_DEFAULTS,
    RThresholdConfig,
)

st.set_page_config(page_title="Campaign Optimizer", page_icon="🎯", layout="wide")


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================


def _validate_env():
    """Check that required env vars are set. Warn in sidebar if not."""
    required = {
        "ALCHEMY_ETH_RPC_URL": "Ethereum RPC (Alchemy) — needed for Aave/Euler on-chain data + whales",
        "HELIUS_SOLANA_RPC_URL": "Solana RPC (Helius) — needed for Kamino whale data",
    }
    optional = {
        "KAMINO_PYUSD_EARN_VAULT_PUBKEY": "Kamino PYUSD Earn vault pubkey",
        "KAMINO_PYUSD_CLMM_VAULT_PUBKEY": "Kamino PYUSD CLMM vault pubkey",
        "KAMINO_MAPLE_MARKET_PUBKEY": "Kamino Maple lending market pubkey",
    }
    missing_required = []
    missing_optional = []

    for key, desc in required.items():
        if not os.environ.get(key):
            missing_required.append(f"**{key}**: {desc}")

    for key, desc in optional.items():
        if not os.environ.get(key):
            missing_optional.append(f"**{key}**: {desc}")

    if missing_required:
        st.sidebar.error(
            "**Missing required env vars:**\n\n"
            + "\n\n".join(missing_required)
            + "\n\nSet these in your `.env` file."
        )

    if missing_optional:
        st.sidebar.warning(
            "**Missing optional env vars:**\n\n"
            + "\n\n".join(missing_optional)
            + "\n\nSome Kamino venues will use DeFiLlama fallback for APY."
        )


# ============================================================================
# WHALE FETCHING (auto-triggered, no static fallback)
# ============================================================================


def _fetch_whales_for_venue(
    venue: dict, r_threshold: float, whale_history: dict | None = None
) -> list[WhaleProfile]:
    """
    Fetch whale profiles for a venue using the appropriate production method.

    - Aave (Core/Horizon): Alchemy indexed transfer API → aToken holders
    - Euler: Alchemy indexed transfer API → eToken holders
    - Morpho: GraphQL V2 positions API
    - Kamino (Earn/CLMM): Helius getTokenLargestAccounts → share holders
    - Curve / Kamino Lend: DeFiLlama competitor landscape (no per-depositor data)

    When whale_history is provided (from Dune sync), it's passed through to
    build_whale_profiles_from_holders() for empirical threshold estimation.

    ERRORS LOUDLY if fetch fails for protocols that support whale fetching.
    Returns empty list ONLY for protocols where whale data is unavailable
    by design (Curve pools, Kamino lending markets).
    """
    protocol = venue["protocol"]
    asset = venue["asset"]

    if protocol == "aave":
        from campaign.evm_data import fetch_aave_whales

        market = venue.get("aave_market", "core")
        print(f"  Fetching Aave {market} whales for {asset}...")
        return fetch_aave_whales(
            asset_symbol=asset,
            market=market,
            r_threshold=r_threshold,
        )

    elif protocol == "euler":
        from campaign.evm_data import fetch_euler_whales

        print(f"  Fetching Euler whales for {asset}...")
        try:
            return fetch_euler_whales(
                asset_symbol=asset,
                r_threshold=r_threshold,
            )
        except Exception as e:
            print(f"  ⚠️ Euler whale fetch failed for {asset}: {e}")
            print("  Continuing without whale profiles for this venue.")
            return []

    elif protocol == "morpho":
        from campaign.data import fetch_morpho_vault_snapshot
        from campaign.evm_data import build_whale_profiles_from_holders

        vault_address = venue.get("vault_address")
        if not vault_address:
            raise RuntimeError(f"Morpho venue {venue['name']} has no vault_address")
        print(f"  Fetching Morpho whales via GraphQL for {vault_address[:10]}...")
        snapshot = fetch_morpho_vault_snapshot(vault_address, chain_id=1)
        return build_whale_profiles_from_holders(
            snapshot.top_depositors,
            snapshot.total_supply_usd,
            r_threshold,
            whale_history=whale_history,
        )

    elif protocol == "kamino" and venue.get("kamino_vault_pubkey"):
        from campaign.evm_data import build_whale_profiles_from_holders
        from campaign.kamino_data import fetch_kamino_vault_metrics, fetch_kamino_vault_whales

        pubkey = venue["kamino_vault_pubkey"]
        print(f"  Fetching Kamino vault whales for {pubkey[:10]}...")
        holders = fetch_kamino_vault_whales(pubkey)
        metrics = fetch_kamino_vault_metrics(pubkey)
        return build_whale_profiles_from_holders(
            holders,
            metrics.total_tvl_usd,
            r_threshold,
            min_position_usd=100_000,
            whale_history=whale_history,
        )

    elif protocol == "kamino" and venue.get("kamino_strategy_pubkey"):
        from campaign.evm_data import build_whale_profiles_from_holders
        from campaign.kamino_data import fetch_kamino_strategy_metrics, fetch_kamino_strategy_whales

        pubkey = venue["kamino_strategy_pubkey"]
        print(f"  Fetching Kamino CLMM strategy whales for {pubkey[:10]}...")
        holders = fetch_kamino_strategy_whales(pubkey)
        metrics = fetch_kamino_strategy_metrics(pubkey)
        return build_whale_profiles_from_holders(
            holders,
            metrics.total_value_locked,
            r_threshold,
            min_position_usd=100_000,
            whale_history=whale_history,
        )

    elif protocol == "curve":
        # Curve pools: whale data not available without subgraph
        print("  Curve pool — no per-depositor whale data available")
        return []

    elif protocol == "kamino":
        # Kamino lending markets: no per-depositor API
        print("  Kamino lending market — no per-depositor whale data available")
        return []

    else:
        raise RuntimeError(
            f"Unknown protocol '{protocol}' for whale fetching in venue {venue['name']}"
        )


def _fetch_current_tvl_and_util(venue: dict) -> tuple[float, float]:
    """
    Fetch live current TVL and utilization for a venue.
    Returns (current_tvl_usd, current_utilization_decimal).

    Falls back to hardcoded values if fetch fails (with warning).
    """
    protocol = venue["protocol"]
    asset = venue["asset"]

    print(f"\n[FETCH] {venue['name']} ({protocol.upper()}, {asset})")

    try:
        if protocol == "aave":
            # Use AAVE V3 on-chain RPC — requires correct asset address
            from campaign.evm_data import (
                AAVE_V3_POOLS,
                STABLECOIN_ADDRESSES,
                fetch_aave_reserve_data,
            )

            market = venue.get("aave_market", "core")
            asset_address = STABLECOIN_ADDRESSES.get(asset)
            if not asset_address:
                raise RuntimeError(f"Unknown asset address for {asset}")
            pool_address = AAVE_V3_POOLS.get(market)
            if not pool_address:
                raise RuntimeError(f"Unknown AAVE market: {market}")

            print(f"  → Calling fetch_aave_reserve_data(asset={asset}, market={market})")
            data = fetch_aave_reserve_data(asset_address, asset, pool_address, market)
            print(
                f"  ✓ Fetched: TVL=${data.total_supply_usd / 1e6:.2f}M, Util={data.utilization:.1%}"
            )
            return data.total_supply_usd, data.utilization

        elif protocol == "euler":
            from campaign.evm_data import EULER_VAULTS, fetch_euler_vault_data

            vault_address = EULER_VAULTS.get(asset)
            if not vault_address:
                raise RuntimeError(f"Unknown Euler vault address for {asset}")
            print(f"  → Calling fetch_euler_vault_data(asset={asset})")
            data = fetch_euler_vault_data(vault_address=vault_address, asset_symbol=asset)
            print(
                f"  ✓ Fetched: TVL=${data.total_supply_usd / 1e6:.2f}M, Util={data.utilization:.1%}"
            )
            return data.total_supply_usd, data.utilization

        elif protocol == "morpho":
            from campaign.data import fetch_morpho_vault_snapshot

            vault_address = venue.get("vault_address")
            if not vault_address:
                raise RuntimeError(f"Morpho venue {venue['name']} has no vault_address")
            print(f"  → Calling fetch_morpho_vault_snapshot(vault={vault_address[:10]}...)")
            snapshot = fetch_morpho_vault_snapshot(vault_address, chain_id=1)
            print(
                f"  ✓ Fetched: TVL=${snapshot.total_supply_usd / 1e6:.2f}M, Util={snapshot.utilization:.1%}"
            )
            return snapshot.total_supply_usd, snapshot.utilization

        elif protocol == "kamino" and venue.get("kamino_vault_pubkey"):
            from campaign.kamino_data import fetch_kamino_vault_metrics

            pubkey = venue["kamino_vault_pubkey"]
            print(f"  → Calling fetch_kamino_vault_metrics(pubkey={pubkey[:10]}...)")
            metrics = fetch_kamino_vault_metrics(pubkey)
            print(f"  ✓ Fetched: TVL=${metrics.total_tvl_usd / 1e6:.2f}M, Util=N/A (vault)")
            # Kamino vaults (Earn) don't have utilization, default to 0
            return metrics.total_tvl_usd, 0.0

        elif protocol == "kamino" and venue.get("kamino_reserve_pubkey"):
            from campaign.kamino_data import fetch_kamino_lend_snapshot

            reserve_pubkey = venue["kamino_reserve_pubkey"]
            market = venue.get("kamino_market_name", "main")
            print(
                f"  → Calling fetch_kamino_lend_snapshot(reserve={reserve_pubkey[:10]}..., market={market})"
            )
            snapshot = fetch_kamino_lend_snapshot(reserve_pubkey, market)
            print(
                f"  ✓ Fetched: TVL=${snapshot['total_supply_usd'] / 1e6:.2f}M, Util={snapshot['utilization']:.1%}"
            )
            return snapshot["total_supply_usd"], snapshot["utilization"]

        elif protocol == "kamino" and venue.get("kamino_market_name"):
            # Kamino lending market - look up reserve by asset symbol
            from campaign.kamino_data import fetch_kamino_reserve_for_asset

            market_name = venue["kamino_market_name"]
            print(
                f"  → Calling fetch_kamino_reserve_for_asset(asset={asset}, market={market_name})"
            )
            reserve = fetch_kamino_reserve_for_asset(asset_symbol=asset, market_name=market_name)
            if not reserve:
                raise RuntimeError(f"No {asset} reserve found in Kamino {market_name} market")
            print(
                f"  ✓ Fetched: TVL=${reserve.total_supply_usd / 1e6:.2f}M, Util={reserve.utilization:.1%}"
            )
            return reserve.total_supply_usd, reserve.utilization

        elif protocol == "kamino" and venue.get("kamino_strategy_pubkey"):
            from campaign.kamino_data import fetch_kamino_strategy_metrics

            pubkey = venue["kamino_strategy_pubkey"]
            print(f"  → Calling fetch_kamino_strategy_metrics(pubkey={pubkey[:10]}...)")
            metrics = fetch_kamino_strategy_metrics(pubkey)
            print(f"  ✓ Fetched: TVL=${metrics.total_value_locked / 1e6:.2f}M, Util=N/A (CLMM)")
            # CLMM strategies don't have utilization
            return metrics.total_value_locked, 0.0

        elif protocol == "curve":
            # Fetch Curve pool data via DeFiLlama yields API
            import requests

            venue_name = venue.get("name", "")

            print(f"  → Querying DeFiLlama yields API for Curve pool '{venue_name}'...")

            resp = requests.get("https://yields.llama.fi/pools", timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"DeFiLlama API failed: HTTP {resp.status_code}")

            pools = resp.json().get("data", [])

            # Build a symbol pattern from asset and venue name
            # e.g., venue "Curve RLUSD-USDC" -> match DeFiLlama symbol "USDC-RLUSD" or "RLUSD-USDC"
            # We match Curve pools on Ethereum that contain our asset in the symbol
            curve_pools = [
                p
                for p in pools
                if p.get("project") == "curve-dex"
                and p.get("chain") == "Ethereum"
                and asset.lower() in (p.get("symbol") or "").lower()
            ]

            if not curve_pools:
                raise RuntimeError(f"No Curve pool found on DeFiLlama for asset={asset}")

            # Try to match the specific pair from venue name
            # venue names like "Curve RLUSD-USDC" or "Curve PYUSD-USDC"
            # DeFiLlama symbols like "USDC-RLUSD" or "PYUSD-USDC"
            best_match = None
            for p in curve_pools:
                dl_symbol = (p.get("symbol") or "").upper()
                dl_tokens = set(dl_symbol.split("-"))
                # Extract pair tokens from venue name (e.g., "Curve RLUSD-USDC" -> {"RLUSD", "USDC"})
                name_parts = venue_name.replace("Curve ", "").upper().split("-")
                name_tokens = set(name_parts)
                if dl_tokens == name_tokens:
                    best_match = p
                    break

            if not best_match:
                # Fall back to the largest Curve pool for this asset
                curve_pools.sort(key=lambda p: p.get("tvlUsd", 0), reverse=True)
                best_match = curve_pools[0]
                print(f"  ⚠️ No exact pair match, using largest pool: {best_match.get('symbol')}")

            tvl_usd = best_match.get("tvlUsd", 0)
            print(
                f"  ✓ Fetched from DeFiLlama: {best_match.get('symbol')} TVL=${tvl_usd / 1e6:.2f}M"
            )

            # Curve pools don't have traditional utilization (swap pools, not lending)
            return tvl_usd, 0.0

        elif protocol == "kamino":
            # Kamino without identifying info
            print(
                "  ⚠️ Kamino venue missing identifying info (need vault_pubkey, strategy_pubkey, or market_name+asset)"
            )
            raise RuntimeError(
                f"Kamino venue {venue['name']} needs vault_pubkey, strategy_pubkey, or market_name"
            )

        else:
            # Unknown protocol - ERROR, don't use hardcoded fallback
            raise RuntimeError(f"Unknown protocol '{protocol}' for venue {venue['name']}")

    except Exception as e:
        print(f"  ❌ FAILED to fetch current TVL/util for {venue['name']}: {e}")
        import traceback

        traceback.print_exc()
        # RE-RAISE the error - no silent fallbacks
        raise RuntimeError(f"Live data fetch failed for {venue['name']}: {e}") from e


# ============================================================================
# VENUE DEFINITIONS
# ============================================================================
# Only things that require human judgment:
#   - current_tvl, target_tvl, target_util
# Base APY, r_threshold, and whale profiles are ALL fetched dynamically.
# Budget ranges and r_max ranges are configured in the sidebar.

# Kamino vault pubkeys loaded from env at runtime
_PYUSD_EARN_VAULT = os.environ.get("KAMINO_PYUSD_EARN_VAULT_PUBKEY")
_PYUSD_CLMM_VAULT = os.environ.get("KAMINO_PYUSD_CLMM_VAULT_PUBKEY")

PROGRAMS = {
    "RLUSD Core": {
        "total_budget": 1_000_000,
        "venues": [
            {
                "name": "AAVE Core Market RLUSD",
                "asset": "RLUSD",
                "protocol": "aave",
                "aave_market": "core",
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 600_000_000,
                "target_tvl": 600_000_000,
                "target_util": 0.40,
            },
            {
                "name": "Euler Sentora RLUSD",
                "asset": "RLUSD",
                "protocol": "euler",
                "defillama_project": "euler-v2",
                "chain": "Ethereum",
                "current_tvl": 190_000_000,
                "target_tvl": 190_000_000,
                "target_util": 0.385,
            },
            {
                "name": "Curve RLUSD-USDC",
                "asset": "RLUSD",
                "protocol": "curve",
                "defillama_project": "curve-dex",
                "pool_address": "0x8e0D210a6B95E7a4CF3e7a94d17B7e992fA1d57f",  # RLUSD-USDC pool
                "chain": "Ethereum",
                "current_tvl": 75_000_000,
                "target_tvl": 75_000_000,
                "target_util": 0.50,
            },
        ],
    },
    "RLUSD Horizon": {
        "total_budget": 180_500,
        "venues": [
            {
                "name": "AAVE Horizon RLUSD",
                "asset": "RLUSD",
                "protocol": "aave",
                "aave_market": "horizon",  # Uses separate pool: 0xAe05Cd22...
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 221_500_000,
                "target_tvl": 221_500_000,
                "target_util": 0.60,
            },
        ],
    },
    "PYUSD": {
        "total_budget": 1_110_000,
        "venues": [
            {
                "name": "AAVE Core Market PYUSD",
                "asset": "PYUSD",
                "protocol": "aave",
                "aave_market": "core",
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 400_000_000,
                "target_tvl": 400_000_000,
                "target_util": 0.60,
            },
            {
                "name": "Kamino Main Market PYUSD",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-lend",
                "kamino_market_name": "main",
                "chain": "Solana",
                "current_tvl": 461_500_000,
                "target_tvl": 461_500_000,
                "target_util": 0.315,
            },
            {
                "name": "Kamino Maple Market PYUSD",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-lend",
                "kamino_market_name": "maple",  # Loaded from KAMINO_MAPLE_MARKET_PUBKEY
                "chain": "Solana",
                "current_tvl": 50_000_000,
                "target_tvl": 50_000_000,
                "target_util": 0.40,
            },
            {
                "name": "Kamino JLP Market PYUSD",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-lend",
                "kamino_market_name": "jlp",
                "chain": "Solana",
                "current_tvl": 80_000_000,
                "target_tvl": 80_000_000,
                "target_util": 0.50,
            },
            {
                "name": "Kamino Earn Vault PYUSD",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino",
                "kamino_vault_pubkey": _PYUSD_EARN_VAULT,  # From env
                "chain": "Solana",
                "current_tvl": 352_000_000,
                "target_tvl": 352_000_000,
                "target_util": 0.42,
            },
            {
                "name": "Kamino CLMM PYUSD",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-liquidity",
                "kamino_strategy_pubkey": _PYUSD_CLMM_VAULT,  # From env — CLMM strategy, not kVault
                "chain": "Solana",
                "current_tvl": 30_000_000,
                "target_tvl": 30_000_000,
                "target_util": 0.50,
            },
            {
                "name": "Euler Sentora PYUSD",
                "asset": "PYUSD",
                "protocol": "euler",
                "defillama_project": "euler-v2",
                "chain": "Ethereum",
                "current_tvl": 250_000_000,
                "target_tvl": 250_000_000,
                "target_util": 0.52,
            },
            {
                "name": "Morpho PYUSD",
                "asset": "PYUSD",
                "protocol": "morpho",
                "vault_address": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
                "chain": "Ethereum",
                "current_tvl": 195_000_000,
                "target_tvl": 100_000_000,
                "target_util": 0.60,
            },
            {
                "name": "Curve PYUSD-USDC",
                "asset": "PYUSD",
                "protocol": "curve",
                "defillama_project": "curve-dex",
                "pool_address": "0x383E6b4437b59fff47B619CBA855CA29342A8559",  # PYUSD-USDC pool
                "chain": "Ethereum",
                "current_tvl": 30_000_000,
                "target_tvl": 30_000_000,
                "target_util": 0.50,
            },
        ],
    },
}


# ============================================================================
# SIMULATION DEFAULTS
# ============================================================================

GRID_B_STEPS = 10
GRID_R_STEPS = 12
MC_PATHS_DEFAULT = 50
HORIZON_DAYS = 28
DT_DAYS = 0.5


# ============================================================================
# HELPERS
# ============================================================================


def _optimal_r_max_at_budget(sr: SurfaceResult, B: float) -> tuple[float, float, int, int]:
    """
    Given a surface and a budget level, find the best r_max and its loss.

    Returns (r_max*, loss*, i, j) at the closest grid B.
    """
    B_vals = sr.grid.B_values
    r_vals = sr.grid.r_max_values
    L = sr.loss_surface
    feas = sr.feasibility_mask

    i = int(np.argmin(np.abs(B_vals - B)))
    row = np.where(feas[i], L[i], np.inf)
    j = int(np.argmin(row))
    return float(r_vals[j]), float(L[i, j]), i, j


def _marginal_loss(sr: SurfaceResult, B: float) -> float:
    """
    Estimate dL/dB at a given budget level on a surface.

    Uses centered finite difference at the optimal r_max for each B.
    """
    B_vals = sr.grid.B_values
    L = sr.loss_surface
    feas = sr.feasibility_mask

    i = int(np.argmin(np.abs(B_vals - B)))
    # optimal r_max at this B
    row = np.where(feas[i], L[i], np.inf)
    j = int(np.argmin(row))

    if i == 0:
        i1 = min(i + 1, L.shape[0] - 1)
        dL = L[i1, j] - L[i, j]
        dB = B_vals[i1] - B_vals[i]
    elif i == L.shape[0] - 1:
        i0 = max(i - 1, 0)
        dL = L[i, j] - L[i0, j]
        dB = B_vals[i] - B_vals[i0]
    else:
        dL = L[i + 1, j] - L[i - 1, j]
        dB = B_vals[i + 1] - B_vals[i - 1]
    return dL / dB if abs(dB) > 0 else 0.0


def _allocate_budget_across_venues(
    results: dict,
    total_budget: float,
    venues: list[dict],
) -> dict:
    """
    Allocate total budget across venues via marginal equalization.

    Algorithm:
    1. Each venue has a pre-computed loss surface L_i(B, r_max).
    2. For each venue, compute dL_i/dB weighted by TVL share — larger
       venues need gentler marginals to justify budget, preventing small
       pools from absorbing disproportionate allocations.
    3. Find lambda* (Lagrange multiplier) via bisection (~60 iterations).
    4. Cap each venue at its max deployable budget (target_tvl ×
       protocol_r_max_ceiling / 52.14).  Redistribute excess.
    5. Floor r_max at the float rate, respecting per-protocol ceilings
       (not just the global 8% ceiling).

    Stores 'adjusted_B' and 'adjusted_r_max' in each result dict.
    """
    active = [
        v
        for v in venues
        if v["name"] in results and results[v["name"]]["overrides"].get("pinned_budget") is None
    ]
    pinned = [
        v
        for v in venues
        if v["name"] in results and results[v["name"]]["overrides"].get("pinned_budget") is not None
    ]

    # Set pinned venues' budgets
    WEEKS_PER_YEAR_PIN = 365.0 / 7.0
    pinned_total = 0.0
    for v in pinned:
        ov = results[v["name"]]["overrides"]
        pb = ov["pinned_budget"]
        results[v["name"]]["adjusted_B"] = pb
        sr = results[v["name"]]["surface"]
        r_max_surface, _, _, _ = _optimal_r_max_at_budget(sr, pb)

        # If user pinned r_max, use that; otherwise floor at float rate
        if ov.get("pinned_r_max"):
            r_max = ov["pinned_r_max"]
        else:
            target_tvl_pin = ov["target_tvl"]
            float_rate_pin = pb / max(target_tvl_pin, 1.0) * WEEKS_PER_YEAR_PIN
            # Respect per-protocol ceiling (not just global 8%)
            proto = v.get("protocol", "").lower()
            _, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(proto, (0.02, GLOBAL_R_MAX_CEILING))
            venue_ceiling = min(proto_hi, GLOBAL_R_MAX_CEILING)
            r_max = max(r_max_surface, min(float_rate_pin, venue_ceiling))
        results[v["name"]]["adjusted_r_max"] = r_max
        pinned_total += pb

    remaining = total_budget - pinned_total

    if not active or remaining <= 0:
        for v in active:
            results[v["name"]]["adjusted_B"] = 0.0
            results[v["name"]]["adjusted_r_max"] = results[v["name"]]["surface"].optimal_r_max
        return results

    # ── TVL weights for marginal equalization ──
    # Weight marginals by target TVL so larger venues get proportionally
    # more budget.  Without this, small pools with steep marginals
    # (each $ produces more APR) absorb disproportionate share.
    WEEKS_PER_YEAR = 365.0 / 7.0
    total_target_tvl = sum(results[v["name"]]["overrides"]["target_tvl"] for v in active)
    n_active = len(active)
    avg_target_tvl = total_target_tvl / max(n_active, 1)
    TVL_WEIGHT_EXPONENT = 0.5  # 0 = no weighting, 1 = full TVL proportionality
    tvl_weights: dict[str, float] = {}
    for v in active:
        target_tvl = results[v["name"]]["overrides"]["target_tvl"]
        tvl_weights[v["name"]] = (target_tvl / max(avg_target_tvl, 1.0)) ** TVL_WEIGHT_EXPONENT

    # ── Per-venue max deployable budget ──
    # Cap at what the protocol-level r_max ceiling allows at target TVL.
    # Budget beyond this is wasted (cap binds, Merkl refunds).
    max_deployable: dict[str, float] = {}
    for v in active:
        target_tvl = results[v["name"]]["overrides"]["target_tvl"]
        proto = v.get("protocol", "").lower()
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(proto, (0.02, GLOBAL_R_MAX_CEILING))
        venue_ceiling = min(proto_hi, GLOBAL_R_MAX_CEILING)
        max_deployable[v["name"]] = target_tvl * venue_ceiling / WEEKS_PER_YEAR

    # Compute marginal loss range across all active venues
    marginals = []
    for v in active:
        sr = results[v["name"]]["surface"]
        for B_val in sr.grid.B_values:
            marginals.append(_marginal_loss(sr, B_val))

    # Bisection bounds — lambda is in the marginal-loss space
    lambda_lo = min(marginals) * 2.0
    lambda_hi = max(marginals) * 0.5
    if lambda_lo >= lambda_hi:
        lambda_lo, lambda_hi = (
            lambda_hi - abs(lambda_hi) - 1e-6,
            lambda_lo + abs(lambda_lo) + 1e-6,
        )

    def _budget_at_lambda(lam: float) -> tuple[float, dict]:
        """For each venue find B where TVL-weighted dL/dB ≈ lam."""
        bmap: dict[str, float] = {}
        total = 0.0
        for v in active:
            sr = results[v["name"]]["surface"]
            w_i = tvl_weights[v["name"]]
            # Large venues (w_i > 1) need gentler slope → get more budget.
            # Small venues (w_i < 1) need steeper slope → get less budget.
            adjusted_lam = lam / max(w_i, 0.01)
            best_B = sr.grid.B_values[0]
            best_dist = float("inf")
            for B_val in sr.grid.B_values:
                m = _marginal_loss(sr, B_val)
                dist = abs(m - adjusted_lam)
                if dist < best_dist:
                    best_dist = dist
                    best_B = float(B_val)
            bmap[v["name"]] = best_B
            total += best_B
        return total, bmap

    # Bisect to find lambda* such that sum B_i(lambda*) = remaining
    best_bmap: dict[str, float] = {}
    for _ in range(60):
        lambda_mid = (lambda_lo + lambda_hi) / 2.0
        consumed, bmap = _budget_at_lambda(lambda_mid)
        best_bmap = bmap
        if consumed > remaining:
            lambda_hi = lambda_mid
        else:
            lambda_lo = lambda_mid
        if abs(consumed - remaining) < remaining * 0.005:
            break

    # Scale to exactly match remaining budget
    alloc_total = sum(best_bmap.values())
    if alloc_total > 0:
        scale = remaining / alloc_total
        best_bmap = {k: v * scale for k, v in best_bmap.items()}

    # ── Post-allocation cap: enforce max deployable budget ──
    # If a venue's allocation exceeds what its protocol r_max ceiling
    # allows at target TVL, cap it and redistribute the excess.
    # Track permanently capped venues to prevent oscillation.
    permanent_caps: set[str] = set()
    for _round in range(5):  # iterate — capping one may push another over
        excess_total = 0.0
        for v in active:
            if v["name"] in permanent_caps:
                continue
            cap = max_deployable[v["name"]]
            if best_bmap[v["name"]] > cap:
                excess_total += best_bmap[v["name"]] - cap
                best_bmap[v["name"]] = cap
                permanent_caps.add(v["name"])
        if excess_total <= 0:
            break
        uncapped = [v for v in active if v["name"] not in permanent_caps]
        if not uncapped:
            break  # All venues at cap — excess is undeployable
        uncapped_total = sum(best_bmap[v["name"]] for v in uncapped)
        for v in uncapped:
            if uncapped_total > 0:
                best_bmap[v["name"]] += excess_total * best_bmap[v["name"]] / uncapped_total
            else:
                best_bmap[v["name"]] += excess_total / len(uncapped)

    # Store results
    for v in active:
        B_alloc = best_bmap.get(v["name"], 0.0)
        sr = results[v["name"]]["surface"]
        r_max_surface, _, _, _ = _optimal_r_max_at_budget(sr, B_alloc)

        # Floor r_max at the float rate so allocated budget isn't wasted.
        # Respect per-protocol ceiling (not just global 8%).
        target_tvl = results[v["name"]]["overrides"]["target_tvl"]
        float_rate = B_alloc / max(target_tvl, 1.0) * WEEKS_PER_YEAR
        proto = v.get("protocol", "").lower()
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(proto, (0.02, GLOBAL_R_MAX_CEILING))
        venue_ceiling = min(proto_hi, GLOBAL_R_MAX_CEILING)
        r_max = max(r_max_surface, min(float_rate, venue_ceiling))

        results[v["name"]]["adjusted_B"] = B_alloc
        results[v["name"]]["adjusted_r_max"] = r_max

    return results


def apr_at_tvl(B: float, tvl: float, r_max: float) -> float:
    if tvl <= 0:
        return r_max
    return min(r_max, B / tvl * 365.0 / 7.0)


def t_bind(B: float, r_max: float) -> float:
    if r_max <= 0:
        return float("inf")
    return B * 365.0 / 7.0 / r_max


def fetch_dynamic_r_threshold(
    asset_symbol: str,
    exclude_vault: str | None = None,
    program_name: str | None = None,
) -> dict:
    """
    Fetch competitor rates from DeFiLlama and compute r_threshold.

    Supports three modes configured per-program in venue_registry:
      - Mode A ("auto"):    Direct asset competitors from DeFiLlama
      - Mode B ("manual"):  Operator-set fixed value
      - Mode C ("blended"): Stablecoin class benchmark (USDC/USDT/DAI/USDS
                            TVL-weighted) minus friction discount

    Falls back to Mode C if Mode A finds no competitors.
    """
    # Look up config for this program
    cfg = PROGRAM_R_THRESHOLD_CONFIG.get(program_name or "", RThresholdConfig())

    # ── Mode B: Manual override ──
    if cfg.mode == "manual":
        return {
            "r_threshold": cfg.manual_value,
            "r_threshold_lo": cfg.manual_value * 0.8,
            "r_threshold_hi": cfg.manual_value * 1.2,
            "competitors": [],
            "source": f"manual ({cfg.manual_value:.2%})",
            "mode": "manual",
        }

    # ── Mode A: Auto (direct asset competitors) ──
    if cfg.mode == "auto":
        try:
            competitors = fetch_competitor_rates(
                asset_symbol=asset_symbol,
                min_tvl=1_000_000,
                exclude_vault_address=exclude_vault,
            )
            # compute_r_threshold handles 0/few/enough competitors internally
            # (with outlier removal, blending, and cap)
            thresholds = compute_r_threshold(competitors)
            thresholds["competitors"] = competitors
            rt_source = thresholds.get("r_threshold_source", "asset_peers")
            thresholds["source"] = f"{asset_symbol} — {rt_source}"
            thresholds["mode"] = "auto"
            return thresholds
        except RuntimeError:
            pass  # Fall through to blended

    # ── Mode C: Blended (stablecoin class benchmark) ──
    # Also reached as fallback when Mode A finds no competitors
    try:
        result = fetch_stablecoin_class_benchmark(
            proxy_assets=cfg.proxy_assets,
            min_pool_tvl=cfg.min_pool_tvl,
            friction_discount=cfg.friction_discount,
            top_n=cfg.top_n_pools,
        )
        return {
            "r_threshold": result["r_threshold"],
            "r_threshold_lo": result["r_threshold"] * 0.8,
            "r_threshold_hi": result["r_threshold"] * 1.2,
            "competitors": [],
            "source": (
                f"stablecoin class benchmark ({result['n_pools']} pools, "
                f"raw={result['raw_benchmark']:.2%}, "
                f"discount={result['friction_discount']:.2%})"
            ),
            "mode": "blended",
            "benchmark_detail": result.get("per_asset", {}),
        }
    except RuntimeError as e:
        print(f"  ⚠ Stablecoin benchmark failed: {e}")

    # Last resort
    return {
        "r_threshold": 0.035,
        "r_threshold_lo": 0.025,
        "r_threshold_hi": 0.045,
        "competitors": [],
        "source": "hardcoded fallback (no DeFiLlama data)",
        "mode": "fallback",
        "error": f"No competitors found for {asset_symbol} or stablecoin fallback",
    }


def run_venue_optimization(
    venue: dict,
    base_apy: float,
    target_tvl: float,
    target_util: float,
    r_threshold: float,
    whale_profiles: list[WhaleProfile],
    total_budget: float,
    pinned_budget: float | None = None,
    pinned_r_max: float | None = None,
    forced_rate: float | None = None,
    n_paths: int = MC_PATHS_DEFAULT,
    weights: LossWeights | None = None,
    retail_config: RetailDepositorConfig | None = None,
    mercenary_config: MercenaryConfig | None = None,
    apy_sensitive_config: APYSensitiveConfig | None = None,
    r_max_range: tuple[float, float] = (0.02, 0.15),
    supply_cap: float = 0.0,
) -> SurfaceResult:
    """Run full MC surface optimization for one venue.

    When forced_rate is set, the optimizer pins r_max = forced_rate and
    derives the required budget B = forced_rate × max(current_tvl, target_tvl) / 52.14.
    If B exceeds total_budget, we still run at the forced rate (overspend)
    but flag it via the returned SurfaceResult metadata.
    """
    # Budget range: derived from venue TVL and r_max range.
    # At budget B with TVL T, the float incentive rate is B/T * 52.14.
    # So for a given r_max range, the meaningful budget range is:
    #   B_lo = T * r_lo / 52.14  (rate = r_lo at target TVL)
    #   B_hi = T * r_hi / 52.14  (rate = r_hi at target TVL, but capped)
    # Also clamp to [10k, total_budget].
    WEEKS_PER_YEAR = 365.0 / 7.0  # 52.14

    # ── Forced rate mode ──
    # When forced_rate is set, we pin r_max to the forced rate and derive
    # the required budget. The forced rate must hold at BOTH current TVL
    # and target TVL — so we use the larger TVL to compute B.
    if forced_rate is not None and forced_rate > 0:
        reference_tvl = max(venue["current_tvl"], target_tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR
        pinned_r_max = forced_rate
        pinned_budget = forced_B
        # Log budget feasibility
        if forced_B > total_budget:
            print(
                f"  [{venue['name']}] ⚠️ FORCED RATE {forced_rate:.2%} requires "
                f"${forced_B:,.0f}/wk but budget is ${total_budget:,.0f}/wk — OVERSPEND"
            )
        else:
            print(
                f"  [{venue['name']}] Forced rate {forced_rate:.2%} → "
                f"B=${forced_B:,.0f}/wk (within budget)"
            )

    # r_max grid range: start with user-provided range, then clamp by
    # per-protocol defaults and global ceiling. Do NOT clamp by budget
    # feasibility — r_max is the APR cap set in Merkl, independent of
    # how much budget is allocated. A high r_max with low budget simply
    # means the cap never binds (pure Float regime), which is valid.
    r_lo_eff, r_hi_eff = r_max_range

    # Apply per-protocol r_max defaults (tighter for established protocols)
    proto = venue.get("protocol", "").lower()
    if proto in PROTOCOL_R_MAX_DEFAULTS:
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS[proto]
        r_lo_eff = max(r_lo_eff, proto_lo)
        r_hi_eff = min(r_hi_eff, proto_hi)

    # Global ceiling
    r_hi_eff = min(r_hi_eff, GLOBAL_R_MAX_CEILING)

    r_hi_eff = max(r_hi_eff, r_lo_eff + 0.005)  # Ensure some range

    if pinned_budget is not None:
        b_min, b_max, b_steps = pinned_budget * 0.95, pinned_budget * 1.05, 3
    else:
        # Budget that gives r_lo at target TVL (lower bound of search)
        b_min = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        # Budget that gives r_hi at target TVL (generous upper bound)
        b_max = min(total_budget, target_tvl * r_hi_eff / WEEKS_PER_YEAR * 1.5)
        # Sanity: ensure b_max > b_min
        b_max = max(b_max, b_min * 2)
        b_steps = GRID_B_STEPS

        # ── Floor-aware budget floor ──
        # When a floor APR is set, the MINIMUM sensible budget is the one
        # that maintains the floor at target TVL.  If this is within the
        # user's budget, raise b_min so the optimizer doesn't waste grid
        # points on configs that guarantee a floor breach.
        if apy_sensitive_config and apy_sensitive_config.floor_apr > 0:
            min_inc_for_floor = max(0, apy_sensitive_config.floor_apr - base_apy)
            floor_budget = target_tvl * min_inc_for_floor / WEEKS_PER_YEAR
            if floor_budget <= total_budget:
                # Floor is achievable — set grid floor to 85% of floor budget
                # (gives optimizer some room to explore just below)
                b_min = max(b_min, floor_budget * 0.85)
                print(
                    f"  [{venue['name']}] Floor-aware grid: "
                    f"floor={apy_sensitive_config.floor_apr:.2%} needs "
                    f"${floor_budget:,.0f}/wk → b_min raised to ${b_min:,.0f}"
                )
            else:
                print(
                    f"  [{venue['name']}] ⚠ Floor {apy_sensitive_config.floor_apr:.2%} "
                    f"needs ${floor_budget:,.0f}/wk but budget is only "
                    f"${total_budget:,.0f}/wk — floor is unachievable"
                )

    if pinned_r_max is not None:
        r_lo, r_hi, r_steps = pinned_r_max * 0.95, pinned_r_max * 1.05, 3
    else:
        r_lo, r_hi = r_lo_eff, r_hi_eff
        r_steps = GRID_R_STEPS

    print(
        f"  [{venue['name']}] Grid: B=[${b_min:,.0f}, ${b_max:,.0f}] ({b_steps} pts), "
        f"r_max=[{r_lo:.2%}, {r_hi:.2%}] ({r_steps} pts)"
    )

    grid = SurfaceGrid.from_ranges(
        B_min=b_min,
        B_max=b_max,
        B_steps=b_steps,
        r_max_min=r_lo,
        r_max_max=r_hi,
        r_max_steps=r_steps,
        dt_days=DT_DAYS,
        horizon_days=HORIZON_DAYS,
        base_apy=base_apy,
        supply_cap=supply_cap,
    )

    env = CampaignEnvironment(r_threshold=r_threshold)

    if weights is None:
        apr_target_total = r_threshold * 1.2
        weights = LossWeights(
            w_spend=1.0,
            w_spend_waste_penalty=2.0,
            w_apr_variance=3.0,
            w_apr_ceiling=5.0,
            w_tvl_shortfall=8.0,
            w_budget_waste=5.0,
            w_mercenary=6.0,
            w_whale_proximity=6.0,
            w_apr_floor=7.0,
            apr_target=apr_target_total,
            apr_ceiling=0.10,
            tvl_target=target_tvl,
            apr_stability_on_total=True,
            apr_floor=apy_sensitive_config.floor_apr if apy_sensitive_config else 0.0,
            apr_floor_sensitivity=apy_sensitive_config.sensitivity if apy_sensitive_config else 0.0,
        )
    else:
        # Use provided weights but override per-venue target fields
        weights = LossWeights(
            w_spend=weights.w_spend,
            w_spend_waste_penalty=weights.w_spend_waste_penalty,
            w_apr_variance=weights.w_apr_variance,
            w_apr_ceiling=weights.w_apr_ceiling,
            w_tvl_shortfall=weights.w_tvl_shortfall,
            w_budget_waste=weights.w_budget_waste,
            w_mercenary=weights.w_mercenary,
            w_whale_proximity=weights.w_whale_proximity,
            w_apr_floor=weights.w_apr_floor,
            apr_target=weights.apr_target,
            apr_ceiling=weights.apr_ceiling,
            tvl_target=target_tvl,
            apr_stability_on_total=weights.apr_stability_on_total,
            apr_floor=weights.apr_floor,
            apr_floor_sensitivity=weights.apr_floor_sensitivity,
        )

    retail = retail_config or RetailDepositorConfig()
    merc = mercenary_config or MercenaryConfig(
        max_capital_usd=target_tvl * 0.1,
    )

    return optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=venue["current_tvl"],
        whale_profiles=whale_profiles,
        weights=weights,
        n_paths=n_paths,
        retail_config=retail,
        mercenary_config=merc,
        apy_sensitive_config=apy_sensitive_config,
        verbose=False,
    )


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("🎯 Merkl Campaign Optimizer")
    st.caption(
        "Full MC simulation engine. Optimizer searches the (B, r_max) hybrid surface. "
        "Base APY fetched on-chain. r_threshold fetched from DeFiLlama. "
        "Whale profiles fetched live via Alchemy/Helius. "
        "Incentive rate is an **output**, not an input."
    )

    _validate_env()

    # ── Program selector ──
    program_name = st.sidebar.selectbox(
        "Program",
        list(PROGRAMS.keys()),
        help="Select the incentive program to optimize. Each program has a set of venues and a total weekly budget.",
    )
    prog = PROGRAMS[program_name]
    venues = prog["venues"]

    total_budget = st.sidebar.number_input(
        "Total Weekly Budget ($)",
        value=int(prog["total_budget"]),
        step=10_000,
        min_value=0,
        key="total_budget",
        help="Total weekly incentive budget across all venues in this program. "
        "The optimizer distributes this across venues to maximize TVL per dollar.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider(
        "MC Paths",
        10,
        200,
        MC_PATHS_DEFAULT,
        10,
        help="More paths = more reliable surfaces. Min 50 recommended.",
    )

    # ── Loss Weights ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Loss Weights")
    with st.sidebar.expander("Loss Function Parameters", expanded=False):
        w_spend = st.number_input(
            "w_spend",
            value=1.0,
            step=0.1,
            min_value=0.0,
            key="w_spend",
            help="Penalty on total dollar spend. Lowest priority — only breaks ties "
            "between campaigns that equally hit TVL target. Default 1.0.",
        )
        w_spend_waste = st.number_input(
            "w_spend_waste_penalty",
            value=2.0,
            step=0.5,
            min_value=0.0,
            key="w_spend_waste",
            help="Extra spend penalty applied when TVL is below target — "
            "penalizes spending money without achieving the TVL goal.",
        )
        w_apr_var = st.number_input(
            "w_apr_variance",
            value=3.0,
            step=0.5,
            min_value=0.0,
            key="w_apr_var",
            help="Penalizes APR volatility (normalized: deviation² / target²). "
            "Higher = optimizer prefers stable, predictable rates.",
        )
        w_apr_ceil = st.number_input(
            "w_apr_ceiling",
            value=5.0,
            step=0.5,
            min_value=0.0,
            key="w_apr_ceil",
            help="Penalty when APR exceeds the ceiling. Normalized to O(1). "
            "5.0 = strong penalty for ceiling breaches.",
        )
        w_tvl_short = st.number_input(
            "w_tvl_shortfall",
            value=8.0,
            step=0.5,
            min_value=0.0,
            key="w_tvl_short",
            help="Primary objective weight. Penalizes TVL below target "
            "(normalized: shortfall² / target²). Highest weight = top priority.",
        )
        w_mercenary = st.number_input(
            "w_mercenary",
            value=6.0,
            step=0.5,
            min_value=0.0,
            key="w_mercenary",
            help="Penalizes mercenary capital fraction. Higher = optimizer avoids "
            "configs that attract hot money seeking yield spikes.",
        )
        w_whale_proximity = st.number_input(
            "w_whale_proximity",
            value=6.0,
            step=0.5,
            min_value=0.0,
            key="w_whale_proximity",
            help="Penalizes whale exit proximity — how close APR is to whale "
            "exit thresholds. Higher = more margin of safety against whale exits.",
        )
        w_apr_floor = st.number_input(
            "w_apr_floor",
            value=7.0,
            step=0.5,
            min_value=0.0,
            key="w_apr_floor",
            help="Penalizes APR dropping below the floor APR set per venue. "
            "Only active when a venue has floor_apr > 0. "
            "High weight = optimizer strongly avoids configs that breach the floor.",
        )
        w_budget_waste = st.number_input(
            "w_budget_waste",
            value=0.0,
            step=0.5,
            min_value=0.0,
            key="w_budget_waste",
            help="Penalizes allocating budget that goes unspent (cap binds, budget refunded). "
            "Note: Merkl refunds unspent budget at 0% fee, so this is less important.",
        )
        apr_target_mult = st.number_input(
            "APR Target Multiplier (× r_threshold)",
            value=1.2,
            step=0.1,
            min_value=0.5,
            max_value=3.0,
            key="apr_target_mult",
            help="The optimizer targets a total APR of r_threshold × this multiplier. "
            "1.2 = aim for 20% above competitor rates to attract deposits.",
        )
        apr_ceiling_val = (
            st.number_input(
                "APR Hard Ceiling (%)",
                value=10.0,
                step=1.0,
                min_value=1.0,
                max_value=50.0,
                key="apr_ceiling_val",
                help="Maximum total APR allowed. APR above this triggers heavy penalty. "
                "For stablecoins, >10% typically attracts mercenary capital.",
            )
            / 100.0
        )

    # ── Retail Depositor Config ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("👥 Retail Depositor Behavior")
    with st.sidebar.expander("Retail Parameters", expanded=False):
        alpha_plus = st.number_input(
            "alpha_plus (inflow elasticity)",
            value=0.15,
            step=0.05,
            min_value=0.0,
            key="alpha_plus",
            help="How fast TVL grows when APR exceeds competitors (r_threshold). "
            "0.15 = moderate. Higher = depositors are more responsive to yield.",
        )
        alpha_minus_mult = st.number_input(
            "alpha_minus_multiplier",
            value=3.0,
            step=0.5,
            min_value=1.0,
            key="alpha_minus_mult",
            help="How much faster depositors leave vs arrive. 3.0 = outflows are 3× "
            "faster than inflows. Reflects 'sticky on the way in, fast on the way out'.",
        )
        response_lag = st.number_input(
            "response_lag_days",
            value=5.0,
            step=1.0,
            min_value=0.0,
            key="response_lag",
            help="Days before depositors react to APR changes. Models real-world delay — "
            "depositors don't instantly move capital when rates change.",
        )
        diffusion_sigma = st.number_input(
            "diffusion_sigma (TVL noise)",
            value=0.008,
            step=0.002,
            min_value=0.0,
            key="diffusion_sigma",
            format="%.4f",
            help="Random daily TVL volatility (standard deviation). 0.008 = ±0.8%/day. "
            "Models organic deposits/withdrawals unrelated to APR.",
        )

    # ── Mercenary Config ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏴‍☠️ Mercenary Capital")
    with st.sidebar.expander("Mercenary Parameters", expanded=False):
        merc_entry_thresh = (
            st.number_input(
                "Entry threshold (%)",
                value=8.0,
                step=0.5,
                min_value=0.0,
                key="merc_entry",
                help="Total APR above which mercenary (hot money) capital starts entering. "
                "For stablecoins, 8% is typical. These depositors leave as soon as APR drops.",
            )
            / 100.0
        )
        merc_exit_thresh = (
            st.number_input(
                "Exit threshold (%)",
                value=6.0,
                step=0.5,
                min_value=0.0,
                key="merc_exit",
                help="Total APR below which mercenary capital exits. The gap between entry "
                "and exit thresholds creates a hysteresis band.",
            )
            / 100.0
        )
        merc_max_frac = (
            st.number_input(
                "Max capital (% of target TVL)",
                value=10.0,
                step=1.0,
                min_value=0.0,
                key="merc_max_frac",
                help="Maximum mercenary capital as percentage of target TVL. "
                "10% = up to 10% of TVL could be hot money at peak APR.",
            )
            / 100.0
        )

    # ── Grid Ranges ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Grid Search Range")
    with st.sidebar.expander("r_max Search Range", expanded=False):
        grid_r_min = (
            st.number_input(
                "r_max min (%)",
                value=2.0,
                step=0.5,
                min_value=0.0,
                key="grid_r_min",
                help="Minimum APR cap (r_max) the optimizer will consider. "
                "Lower bound of the search grid for the Merkl rate ceiling.",
            )
            / 100.0
        )
        grid_r_max_raw = (
            st.number_input(
                "r_max max (%)",
                value=GLOBAL_R_MAX_CEILING * 100,
                step=0.5,
                min_value=1.0,
                key="grid_r_max",
                help=f"Max incentive APR cap. Global ceiling = {GLOBAL_R_MAX_CEILING:.0%}. "
                f">8% attracts mercenaries on stablecoin pools.",
            )
            / 100.0
        )
        grid_r_max = min(grid_r_max_raw, GLOBAL_R_MAX_CEILING)
        st.caption(
            "Protocol defaults: "
            + ", ".join(
                f"{k}: {lo:.0%}–{hi:.0%}" for k, (lo, hi) in PROTOCOL_R_MAX_DEFAULTS.items()
            )
        )

    # ── Top-level tabs ──
    tab_multi, tab_single = st.tabs(
        [
            "📊 Multi-Venue Program Optimization",
            "🔬 Single-Venue Optimization",
        ]
    )

    # ==================================================================
    # TAB 1: MULTI-VENUE PROGRAM OPTIMIZATION
    # ==================================================================
    with tab_multi:
        _run_multi_venue_tab(
            program_name=program_name,
            venues=venues,
            total_budget=total_budget,
            n_paths=n_paths,
            w_spend=w_spend,
            w_spend_waste=w_spend_waste,
            w_apr_var=w_apr_var,
            w_apr_ceil=w_apr_ceil,
            w_tvl_short=w_tvl_short,
            w_mercenary=w_mercenary,
            w_whale_proximity=w_whale_proximity,
            w_apr_floor=w_apr_floor,
            w_budget_waste=w_budget_waste,
            apr_target_mult=apr_target_mult,
            apr_ceiling_val=apr_ceiling_val,
            alpha_plus=alpha_plus,
            alpha_minus_mult=alpha_minus_mult,
            response_lag=response_lag,
            diffusion_sigma=diffusion_sigma,
            merc_entry_thresh=merc_entry_thresh,
            merc_exit_thresh=merc_exit_thresh,
            merc_max_frac=merc_max_frac,
            grid_r_min=grid_r_min,
            grid_r_max=grid_r_max,
        )

    # ==================================================================
    # TAB 2: SINGLE-VENUE OPTIMIZATION
    # ==================================================================
    with tab_single:
        _run_single_venue_tab(
            programs=PROGRAMS,
            n_paths=n_paths,
            w_spend=w_spend,
            w_spend_waste=w_spend_waste,
            w_apr_var=w_apr_var,
            w_apr_ceil=w_apr_ceil,
            w_tvl_short=w_tvl_short,
            w_mercenary=w_mercenary,
            w_whale_proximity=w_whale_proximity,
            w_apr_floor=w_apr_floor,
            w_budget_waste=w_budget_waste,
            apr_target_mult=apr_target_mult,
            apr_ceiling_val=apr_ceiling_val,
            alpha_plus=alpha_plus,
            alpha_minus_mult=alpha_minus_mult,
            response_lag=response_lag,
            diffusion_sigma=diffusion_sigma,
            merc_entry_thresh=merc_entry_thresh,
            merc_exit_thresh=merc_exit_thresh,
            merc_max_frac=merc_max_frac,
            grid_r_min=grid_r_min,
            grid_r_max=grid_r_max,
        )


# ============================================================================
# TAB 1: MULTI-VENUE PROGRAM OPTIMIZATION
# ============================================================================


def _run_multi_venue_tab(
    *,
    program_name,
    venues,
    total_budget,
    n_paths,
    w_spend,
    w_spend_waste,
    w_apr_var,
    w_apr_ceil,
    w_tvl_short,
    w_mercenary,
    w_whale_proximity,
    w_apr_floor,
    w_budget_waste,
    apr_target_mult,
    apr_ceiling_val,
    alpha_plus,
    alpha_minus_mult,
    response_lag,
    diffusion_sigma,
    merc_entry_thresh,
    merc_exit_thresh,
    merc_max_frac,
    grid_r_min,
    grid_r_max,
):
    """Full multi-venue program optimization workflow."""

    # ── Fetch competitor rates / r_threshold ──
    st.header("📊 Competitor Rates & r_threshold")

    assets_in_program = list(set(v["asset"] for v in venues))
    if "r_thresholds" not in st.session_state or st.button("🔄 Refresh Competitor Rates"):
        with st.spinner("Fetching competitor rates from DeFiLlama..."):
            r_thresh_data = {}
            for asset in assets_in_program:
                r_thresh_data[asset] = fetch_dynamic_r_threshold(
                    asset,
                    program_name=program_name,
                )
            st.session_state["r_thresholds"] = r_thresh_data

    r_thresholds = st.session_state.get("r_thresholds", {})

    if r_thresholds:
        for asset, data in r_thresholds.items():
            rt = data.get("r_threshold", 0.045)
            rt_lo = data.get("r_threshold_lo", 0.035)
            rt_hi = data.get("r_threshold_hi", 0.055)
            n_comp = len(data.get("competitors", []))
            source = data.get("source", "unknown")
            _rt_source = data.get("r_threshold_source", "")
            usdc_bm = data.get("usdc_benchmark")
            n_filtered = data.get("n_peers_after_outlier_filter", n_comp)
            outliers = data.get("outliers_removed", 0)

            detail_parts = [f"range: {rt_lo:.2%} – {rt_hi:.2%}"]
            if n_comp > 0:
                detail_parts.append(f"{n_comp} competitors")
                if outliers > 0:
                    detail_parts.append(f"{outliers} outliers removed")
                detail_parts.append(f"{n_filtered} used")
            if usdc_bm is not None:
                detail_parts.append(f"USDC benchmark: {usdc_bm:.2%}")
            detail_parts.append(f"source: _{source}_")

            st.markdown(f"**{asset}**: r_threshold = **{rt:.2%}** ({', '.join(detail_parts)})")
            if data.get("error"):
                st.warning(f"⚠️ {asset}: {data['error']}")
            comps = data.get("competitors", [])
            if comps:
                with st.expander(f"📋 {asset} Competitors ({len(comps)})"):
                    crows = [
                        {
                            "Venue": c.venue,
                            "Symbol": c.symbol,
                            "TVL ($M)": c.tvl_usd / 1e6,
                            "Base APY": c.apy_base,
                            "Reward APY": c.apy_reward,
                            "Total APY": c.apy_total,
                        }
                        for c in comps[:15]
                    ]
                    st.dataframe(
                        pd.DataFrame(crows).style.format(
                            {
                                "TVL ($M)": "${:,.1f}M",
                                "Base APY": "{:.2%}",
                                "Reward APY": "{:.2%}",
                                "Total APY": "{:.2%}",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ── Fetch base APYs ──
    st.header("📡 Base APY (On-Chain / API)")

    if "base_apys" not in st.session_state or st.button("🔄 Refresh Base APYs"):
        with st.spinner("Fetching base APYs..."):
            st.session_state["base_apys"] = fetch_all_base_apys(venues)

    base_apys: dict[str, BaseAPYResult] = st.session_state.get("base_apys", {})

    if base_apys:
        brows = []
        for v in venues:
            r = base_apys.get(v["name"])
            brows.append(
                {
                    "Venue": v["name"],
                    "Base APY": r.base_apy if r else 0.0,
                    "Source": r.source if r else "not fetched",
                }
            )
        st.dataframe(
            pd.DataFrame(brows).style.format({"Base APY": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Morpho sleeve detail
        for v in venues:
            r = base_apys.get(v["name"])
            if r and r.source == "morpho_graphql" and "sleeves" in r.details:
                with st.expander(f"🔍 {v['name']} — Sleeve Breakdown"):
                    srows = [
                        {
                            "Collateral": s["collateral"],
                            "Supply ($M)": s["supply_usd"] / 1e6,
                            "APY": s["apy"],
                            "Weight": s["weight"],
                            "Weighted APY": s["apy"] * s["weight"],
                        }
                        for s in r.details["sleeves"]
                    ]
                    st.dataframe(
                        pd.DataFrame(srows).style.format(
                            {
                                "Supply ($M)": "${:,.1f}M",
                                "APY": "{:.2%}",
                                "Weight": "{:.1%}",
                                "Weighted APY": "{:.3%}",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        f"Vault base APY: **{r.base_apy:.2%}** "
                        f"(active only: {r.details.get('active_only_apy', 0):.2%}, "
                        f"idle: {r.details.get('idle_fraction', 0):.1%})"
                    )

    # ── Per-venue overrides ──
    st.header(f"📋 {program_name} — Venue Targets")
    st.caption(
        "Set target TVL and utilization. Incentive rate is computed by the optimizer. "
        "Pin budget/r_max only for contractual constraints. "
        "Whale profiles fetched automatically at optimization time. "
        "**Current values are fetched live from on-chain/API data.**"
    )

    # Fetch current TVL and utilization for all venues
    # Use program_name as key to avoid caching across different programs
    cache_key = f"current_values_{program_name}"
    if cache_key not in st.session_state or st.button(
        "🔄 Refresh Live Data", key=f"refresh_{program_name}"
    ):
        with st.spinner("🔄 Fetching current TVL and utilization from live data sources..."):
            current_values = {}
            fetch_errors = []
            for v in venues:
                try:
                    current_tvl, current_util = _fetch_current_tvl_and_util(v)
                    current_values[v["name"]] = {
                        "current_tvl": current_tvl,
                        "current_util": current_util,
                    }
                except Exception as e:
                    fetch_errors.append((v["name"], str(e)))
                    # Use 0 to indicate fetch failed - will be obvious in UI
                    current_values[v["name"]] = {
                        "current_tvl": 0.0,
                        "current_util": 0.0,
                    }
            st.session_state[cache_key] = current_values
            st.session_state[f"{cache_key}_errors"] = fetch_errors

        if fetch_errors:
            st.error(f"❌ Failed to fetch live data for {len(fetch_errors)} venues:")
            for vname, err in fetch_errors:
                st.error(f"  • **{vname}**: {err}")
            st.warning(
                "⚠️ Please check RPC URLs, API keys, and venue configs. Showing $0 for failed fetches."
            )
        else:
            st.success(f"✅ Fetched current values for {len(venues)} venues")

    current_values = st.session_state.get(cache_key, {})
    fetch_errors = st.session_state.get(f"{cache_key}_errors", [])

    overrides = {}
    for i, v in enumerate(venues):
        base_r = base_apys.get(v["name"])
        fetched_base = base_r.base_apy if base_r else 0.0
        asset_thresh = r_thresholds.get(v["asset"], {})
        default_r_thresh = asset_thresh.get("r_threshold", 0.045)

        # Get live current values
        curr = current_values[v["name"]]

        with st.expander(
            f"**{v['name']}** ({v['protocol'].upper()}) — Base: {fetched_base:.2%}, "
            f"Current TVL: ${curr['current_tvl'] / 1e6:.1f}M, Util: {curr['current_util']:.1%}",
            expanded=(i < 2),
        ):
            is_curve = v["protocol"] == "curve"
            c1, c2, c3 = st.columns(3)
            with c1:
                t_tvl = st.number_input(
                    "Target TVL ($M)",
                    value=float(curr["current_tvl"] / 1e6),
                    step=10.0,
                    min_value=0.0,
                    key=f"tvl_{i}",
                    help="The TVL level you want this venue to reach or maintain. "
                    f"Current live value: ${curr['current_tvl'] / 1e6:.1f}M. "
                    "Raise to grow the venue, lower to wind down.",
                )
            with c2:
                if is_curve:
                    # Curve DEX pools have no borrow-side utilization; fixed at 50%
                    st.text_input(
                        "Target Util (%)",
                        value="50 (fixed — DEX pool)",
                        disabled=True,
                        key=f"util_{i}",
                        help="Curve DEX pools don't have borrowing — utilization is fixed at 50%.",
                    )
                    t_util = 50.0
                else:
                    t_util = st.number_input(
                        "Target Util (%)",
                        value=float(curr["current_util"] * 100),
                        step=1.0,
                        min_value=0.0,
                        max_value=100.0,
                        key=f"util_{i}",
                        help="Fraction of deposits that are borrowed. "
                        f"Current: {curr['current_util']:.1%}. "
                        "Net Supply (idle capital) = TVL × (1 − util). "
                        "High util = APY-sensitive exit risk → prefer stable rates.",
                    )
            with c3:
                r_thresh_ov = st.number_input(
                    "r_threshold Override (%)",
                    value=0.0,
                    step=0.1,
                    min_value=0.0,
                    key=f"rthresh_ov_{i}",
                    help="Override the auto-fetched competitor rate. Leave at 0 to use the "
                    f"DeFiLlama-derived value ({default_r_thresh:.2%}). "
                    "r_threshold is what depositors can earn elsewhere — when your venue's "
                    "total APR is below this, TVL drifts away.",
                )

            # ── Optional: specify target incentive APR to derive budget ──
            st.markdown("**Budget Specification** (choose one):")
            bc1, bc2 = st.columns(2)
            with bc1:
                target_inc_apr = (
                    st.number_input(
                        "Target Incentive APR at Target TVL (%)",
                        value=0.0,
                        step=0.25,
                        min_value=0.0,
                        key=f"target_inc_apr_{i}",
                        help="The incentive APR you want depositors to earn at your target TVL. "
                        "If set (> 0), the weekly budget is auto-computed: "
                        "Budget = Target TVL × APR × 7/365. "
                        "Leave at 0 to use the pinned budget or let the optimizer decide.",
                    )
                    / 100.0
                )
            with bc2:
                if target_inc_apr > 0 and t_tvl > 0:
                    implied_budget = t_tvl * 1e6 * target_inc_apr * 7.0 / 365.0
                    st.metric("Implied Weekly Budget", f"${implied_budget:,.0f}")
                    st.caption(
                        f"= ${t_tvl:.0f}M × {target_inc_apr:.2%} × 7/365 "
                        f"→ Total APR @ target: {fetched_base + target_inc_apr:.2%}"
                    )
                else:
                    implied_budget = None
                    st.caption(
                        "Set a target incentive APR to auto-derive the budget, "
                        "or use Pin Budget below."
                    )

            st.markdown("**Constraints** (optional):")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                pin_b = st.checkbox(
                    "Pin Budget?",
                    key=f"pin_b_{i}",
                    help="Lock this venue's budget to a specific amount. "
                    "Use for contractual minimums or fixed partnership amounts.",
                )
                pin_b_val = None
                if pin_b:
                    pin_b_val = st.number_input(
                        "Pinned Budget ($/wk)",
                        value=50_000,
                        step=5000,
                        key=f"pinbval_{i}",
                        help="Exact weekly budget for this venue. Optimizer won't change it.",
                    )
            with cc2:
                pin_r = st.checkbox(
                    "Pin r_max?",
                    key=f"pin_r_{i}",
                    help="Lock the APR cap (r_max) to a specific value. "
                    "Use for APY-sensitive vaults (rate MUST stay fixed) or client rate promises.",
                )
                pin_r_val = None
                if pin_r:
                    pin_r_val = (
                        st.number_input(
                            "Pinned r_max (%)",
                            value=6.0,
                            step=0.25,
                            min_value=0.0,
                            key=f"pinrval_{i}",
                            help="Maximum incentive APR cap set in Merkl. Depositors never earn more than this.",
                        )
                        / 100.0
                    )
            with cc3:
                force_rate = st.checkbox(
                    "Force incentive rate?",
                    key=f"force_r_{i}",
                    help="Override optimizer entirely — set a fixed incentive rate "
                    "regardless of TVL dynamics. Use for APY-sensitive vaults where "
                    "any rate change would trigger unwinds.",
                )
                forced_rate = None
                if force_rate:
                    forced_rate = (
                        st.number_input(
                            "Forced rate (%)",
                            value=5.0,
                            step=0.25,
                            min_value=0.0,
                            key=f"frate_{i}",
                            help="Fixed incentive APR — optimizer won't touch this. "
                            "Typically used for APY-sensitive positions.",
                        )
                        / 100.0
                    )

            effective_base = fetched_base
            effective_r_thresh = (r_thresh_ov / 100.0) if r_thresh_ov > 0 else default_r_thresh

            # ── APY Sensitivity / Floor APR ──
            st.markdown("**APY Sensitivity** (optional):")
            lc1, lc2 = st.columns(2)
            with lc1:
                floor_apr = (
                    st.number_input(
                        "Floor APR (%)",
                        value=0.0,
                        step=0.25,
                        min_value=0.0,
                        key=f"floor_apr_{i}",
                        help="Minimum acceptable total APR for APY-sensitive depositors (loopers, yield chasers). "
                        "When APR drops below this floor, sensitive depositors begin exiting — "
                        "removing headline TVL equal to leverage_multiple × their real capital. "
                        "Set to 0 to disable APY-sensitive modeling.",
                    )
                    / 100.0
                )
            with lc2:
                apr_sensitivity = st.slider(
                    "APR Sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=f"apr_sens_{i}",
                    help="How quickly APY-sensitive depositors react to APR dropping below the floor. "
                    "0.0 = slow (3-day delay, gradual unwind). "
                    "1.0 = instant (no delay, immediate mass unwind). "
                    "Higher sensitivity creates more punishing TVL cascades "
                    "for campaign configs that let APR dip below the floor.",
                )

            # ── Supply Cap ──
            venue_supply_cap = v.get("supply_cap", 0.0)
            if venue_supply_cap > 0:
                st.markdown(
                    f"**Supply Cap:** ${venue_supply_cap / 1e6:.0f}M (from registry/on-chain)"
                )
            sc_override = (
                st.number_input(
                    "Supply Cap Override ($M) — 0 = unlimited",
                    value=venue_supply_cap / 1e6 if venue_supply_cap > 0 else 0.0,
                    step=10.0,
                    min_value=0.0,
                    key=f"supply_cap_{i}",
                    help="Maximum TVL the venue can accept. Set from on-chain data or manually. "
                    "0 = unlimited. When hit, new deposits are rejected and APR calculation changes.",
                )
                * 1e6
            )

            # If target incentive APR is specified, derive pinned budget from it
            effective_pin_b = pin_b_val
            if implied_budget is not None and not pin_b:
                effective_pin_b = implied_budget

            overrides[v["name"]] = {
                "target_tvl": t_tvl * 1e6,
                "target_util": t_util / 100,
                "base_apy": effective_base,
                "r_threshold": effective_r_thresh,
                "pinned_budget": effective_pin_b,
                "pinned_r_max": pin_r_val,
                "forced_rate": forced_rate,
                "floor_apr": floor_apr,
                "apr_sensitivity": apr_sensitivity,
                "supply_cap": sc_override,
            }

    # ── Summary table ──
    st.subheader("Target Summary")
    srows = []
    for v in venues:
        ov = overrides[v["name"]]
        ns = ov["target_tvl"] * (1 - ov["target_util"])
        srows.append(
            {
                "Venue": v["name"],
                "Protocol": v["protocol"].upper(),
                "Asset": v["asset"],
                "Target TVL": ov["target_tvl"],
                "Target Util": ov["target_util"],
                "Net Supply": ns,
                "Base APY": ov["base_apy"],
                "r_threshold": ov["r_threshold"],
                "Constraints": (
                    ("📌B " if ov["pinned_budget"] else "")
                    + ("📌r " if ov["pinned_r_max"] else "")
                    + ("🎯rate " if ov["forced_rate"] else "")
                ).strip()
                or "—",
            }
        )
    st.dataframe(
        pd.DataFrame(srows).style.format(
            {
                "Target TVL": "${:,.0f}",
                "Target Util": "{:.1%}",
                "Net Supply": "${:,.0f}",
                "Base APY": "{:.2%}",
                "r_threshold": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── RUN ──
    st.markdown("---")

    # ── Dune Data Sync ──
    dune_col1, dune_col2 = st.columns([1, 3])
    with dune_col1:
        if st.button("🔄 Sync Dune Data", key="mv_dune_sync"):
            with st.spinner("Syncing whale + mercenary data from Dune Analytics..."):
                try:
                    import sys

                    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                    from dune.sync import get_sync_status, sync_all_venues

                    sync_results = sync_all_venues(days=90)
                    st.session_state["dune_sync_results"] = sync_results
                    total_flows = sum(r.whale_flows_count for r in sync_results)
                    total_merc = sum(r.mercenary_count for r in sync_results)
                    skipped = sum(1 for r in sync_results if r.skipped)
                    st.success(
                        f"✅ Dune sync complete: {total_flows} whale flows, "
                        f"{total_merc} mercenary addresses, {skipped} skipped (non-EVM)"
                    )
                except Exception as e:
                    st.error(f"❌ Dune sync failed: {e}")
    with dune_col2:
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from dune.sync import get_sync_status

            sync_status = get_sync_status()
            if sync_status:
                synced = sum(1 for s in sync_status.values() if s["has_whale_flows"])
                st.caption(
                    f"📊 Cached data: {synced} venues synced | "
                    + ", ".join(
                        f"{pid}: {s['whale_count']}flows"
                        for pid, s in list(sync_status.items())[:3]
                    )
                )
            else:
                st.caption("No cached Dune data. Click 'Sync Dune Data' to fetch.")
        except Exception:
            st.caption("Dune sync module available. Click to fetch whale + mercenary data.")

    # Per-venue calibration toggle
    _use_cal = st.checkbox(
        "📈 Use Per-Venue Calibration (DeFiLlama history)",
        value=False,
        key="mv_use_calibration",
        help="When enabled, each venue with a DeFiLlama pool ID gets its own calibrated "
        "retail depositor parameters (α+, α-, σ, lag) from 90-day historical data. "
        "Mercenary thresholds are also derived from calibrated r_threshold. "
        "Venues without history use the shared sidebar values as fallback.",
    )

    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    if run_btn:
        results = {}
        progress = st.progress(0, text="Initializing...")
        whale_fetch_errors = []

        # Build configs from sidebar
        _user_weights = LossWeights(
            w_spend=w_spend,
            w_spend_waste_penalty=w_spend_waste,
            w_apr_variance=w_apr_var,
            w_apr_ceiling=w_apr_ceil,
            w_tvl_shortfall=w_tvl_short,
            w_budget_waste=w_budget_waste,
            w_mercenary=w_mercenary,
            w_whale_proximity=w_whale_proximity,
            w_apr_floor=w_apr_floor,
            apr_target=0,  # Will be overridden per-venue
            apr_ceiling=apr_ceiling_val,
            tvl_target=0,  # Will be overridden per-venue
            apr_stability_on_total=True,
        )
        user_retail = RetailDepositorConfig(
            alpha_plus=alpha_plus,
            alpha_minus_multiplier=alpha_minus_mult,
            response_lag_days=response_lag,
            diffusion_sigma=diffusion_sigma,
        )

        # ── Per-venue calibration (auto-fetch from DeFiLlama history) ──
        venue_calibrations = {}
        if st.session_state.get("mv_use_calibration", False):
            for v in venues:
                pool_id = v.get("defillama_pool_id", "")
                if pool_id:
                    try:
                        from campaign.historical import fetch_and_calibrate

                        cal = fetch_and_calibrate(
                            pool_id,
                            days=90,
                            default_r_threshold=r_thresholds.get(v["asset"], {}).get(
                                "r_threshold", 0.045
                            ),
                        )
                        if cal.data_quality != "insufficient":
                            venue_calibrations[v["name"]] = cal
                    except Exception as e:
                        print(f"  [{v['name']}] Calibration failed: {e}")

        for idx, v in enumerate(venues):
            ov = overrides[v["name"]]
            pct = idx / len(venues)

            # Build per-venue weights with correct apr_target
            venue_floor = ov.get("floor_apr", 0.0)
            venue_sens = ov.get("apr_sensitivity", 0.0)
            venue_weights = LossWeights(
                w_spend=w_spend,
                w_spend_waste_penalty=w_spend_waste,
                w_apr_variance=w_apr_var,
                w_apr_ceiling=w_apr_ceil,
                w_tvl_shortfall=w_tvl_short,
                w_budget_waste=w_budget_waste,
                w_mercenary=w_mercenary,
                w_whale_proximity=w_whale_proximity,
                w_apr_floor=w_apr_floor,
                apr_target=ov["r_threshold"] * apr_target_mult,
                apr_ceiling=apr_ceiling_val,
                tvl_target=ov["target_tvl"],
                apr_stability_on_total=True,
                apr_floor=venue_floor,
                apr_floor_sensitivity=venue_sens,
            )

            # Build per-venue APY-sensitive config (only active when floor_apr > 0)
            venue_apy_sensitive = None
            if venue_floor > 0:
                venue_apy_sensitive = APYSensitiveConfig(
                    floor_apr=venue_floor,
                    sensitivity=venue_sens,
                    max_sensitive_tvl=ov["target_tvl"] * 0.10,  # Default: 10% of TVL
                )

            # ── Per-venue retail config (calibrated or sidebar) ──
            venue_cal = venue_calibrations.get(v["name"])
            if venue_cal:
                venue_retail = RetailDepositorConfig(
                    alpha_plus=venue_cal.alpha_plus,
                    alpha_minus_multiplier=venue_cal.alpha_minus_multiplier,
                    response_lag_days=venue_cal.response_lag_days,
                    diffusion_sigma=venue_cal.diffusion_sigma,
                )
                # Derive mercenary thresholds from calibrated r_threshold_mean
                cal_merc_entry = venue_cal.r_threshold_mean * 1.8
                cal_merc_exit = venue_cal.r_threshold_mean * 1.3
            else:
                venue_retail = user_retail
                cal_merc_entry = merc_entry_thresh
                cal_merc_exit = merc_exit_thresh

            user_merc = MercenaryConfig(
                entry_threshold=cal_merc_entry,
                exit_threshold=cal_merc_exit,
                max_capital_usd=ov["target_tvl"] * merc_max_frac,
            )

            # Step 1: Fetch whales (with optional Dune history for empirical thresholds)
            progress.progress(pct, text=f"Fetching whales for {v['name']}...")
            whale_hist = None
            try:
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from dune.sync import build_whale_history_lookup

                whale_hist = build_whale_history_lookup(v.get("pool_id", ""))
                if whale_hist:
                    print(f"  [{v['name']}] Using Dune whale history ({len(whale_hist)} addresses)")
            except Exception:
                pass  # No cached Dune data — will use synthetic thresholds
            try:
                whale_profiles = _fetch_whales_for_venue(
                    v, ov["r_threshold"], whale_history=whale_hist
                )
            except Exception as e:
                whale_fetch_errors.append((v["name"], str(e)))
                whale_profiles = []

            # Step 2: Run optimization
            # MC path guardrails: more whales = more variance = need more paths
            min_paths = max(30, 10 * max(len(whale_profiles), 1))
            n_paths_eff = max(n_paths, min_paths)
            if n_paths_eff > n_paths:
                print(
                    f"  [{v['name']}] MC paths raised {n_paths} → {n_paths_eff} "
                    f"(whale guardrail: {len(whale_profiles)} whales)"
                )

            progress.progress(pct + 0.5 / len(venues), text=f"Optimizing {v['name']}...")
            t0 = time.time()
            sr = run_venue_optimization(
                venue=v,
                base_apy=ov["base_apy"],
                target_tvl=ov["target_tvl"],
                target_util=ov["target_util"],
                r_threshold=ov["r_threshold"],
                whale_profiles=whale_profiles,
                total_budget=total_budget,
                pinned_budget=ov["pinned_budget"],
                pinned_r_max=ov["pinned_r_max"],
                forced_rate=ov.get("forced_rate"),
                n_paths=n_paths_eff,
                weights=venue_weights,
                retail_config=venue_retail,
                mercenary_config=user_merc,
                apy_sensitive_config=venue_apy_sensitive,
                r_max_range=(grid_r_min, grid_r_max),
                supply_cap=ov.get("supply_cap", 0.0),
            )
            elapsed = time.time() - t0

            # Compute forced-rate budget feasibility
            mv_forced_info = None
            if ov.get("forced_rate") and ov["forced_rate"] > 0:
                _WPY = 365.0 / 7.0
                _ref_tvl = max(v["current_tvl"], ov["target_tvl"])
                _req_B = _ref_tvl * ov["forced_rate"] / _WPY
                mv_forced_info = {
                    "forced_rate": ov["forced_rate"],
                    "required_budget": _req_B,
                    "input_budget": total_budget,
                    "overspend": _req_B > total_budget,
                    "overspend_amount": max(0, _req_B - total_budget),
                    "reference_tvl": _ref_tvl,
                }

            results[v["name"]] = {
                "surface": sr,
                "venue": v,
                "overrides": ov,
                "time": elapsed,
                "n_whales": len(whale_profiles),
                "calibration": venue_cal,  # CalibrationResult or None
                "forced_rate_info": mv_forced_info,
            }

        progress.progress(1.0, text="Allocating budget across venues (marginal equalization)...")
        _allocate_budget_across_venues(results, total_budget, venues)
        progress.progress(1.0, text="Done!")

        # Show whale fetch errors as warnings
        if whale_fetch_errors:
            for vname, err in whale_fetch_errors:
                st.warning(f"⚠️ Whale fetch failed for **{vname}**: {err}")

        # Show calibration summary
        if venue_calibrations:
            st.success(
                f"✅ Per-venue calibration applied for {len(venue_calibrations)} venues: "
                + ", ".join(venue_calibrations.keys())
            )

        st.session_state["results"] = results
        st.session_state["venue_calibrations"] = venue_calibrations

    # ── RESULTS ──
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]

    st.header("📊 Optimal Campaign Parameters")

    opt_rows = []
    for v in venues:
        if v["name"] not in results:
            continue
        r = results[v["name"]]
        sr = r["surface"]
        ov = r["overrides"]
        base = ov["base_apy"]

        # Use allocated budget and r_max from the marginal allocator
        B_star = r.get("adjusted_B", sr.optimal_B)
        r_max_star = r.get("adjusted_r_max", sr.optimal_r_max)

        # Get MC diagnostics at the allocated (B, r_max) point on the surface
        i_alloc = int(np.argmin(np.abs(sr.grid.B_values - B_star)))
        j_alloc = int(np.argmin(np.abs(sr.grid.r_max_values - r_max_star)))
        mc = sr.mc_results.get((i_alloc, j_alloc)) or sr.optimal_mc_result

        tb = t_bind(B_star, r_max_star)
        _inc_current = apr_at_tvl(B_star, v["current_tvl"], r_max_star)
        inc_target = apr_at_tvl(B_star, ov["target_tvl"], r_max_star)

        # Campaign type: compare T_bind to TARGET TVL (not current)
        # because campaign is designed for the target state.
        # Float-like: T_bind < 50% target → cap rarely binds → rate floats
        # MAX-like:   T_bind > 120% target → cap always binds → effectively constant rate
        # Hybrid:     In between
        target_tvl_for_type = ov["target_tvl"]
        if tb < target_tvl_for_type * 0.5:
            ctype = "Float-like"
        elif tb > target_tvl_for_type * 1.2:
            ctype = "MAX-like"
        else:
            ctype = "Hybrid"

        ns = ov["target_tvl"] * (1 - ov["target_util"])

        opt_rows.append(
            {
                "Venue": v["name"],
                "Asset": v["asset"],
                "B* ($/wk)": B_star,
                "r_max*": r_max_star,
                "Base APY": base,
                "Incentive @ Target TVL": inc_target,
                "Total APR @ Target": base + inc_target,
                "T_bind ($M)": tb / 1e6,
                "Mean TVL ($M)": mc.mean_tvl / 1e6 if mc else 0,
                "Budget Util": mc.mean_budget_util if mc else 0,
                "Type": ctype,
                "Loss": sr.optimal_loss,
                "Net Supply ($M)": ns / 1e6,
                "TVL/$inc": ov["target_tvl"] / B_star if B_star > 0 else 0,
                "NS/$inc": ns / B_star if B_star > 0 else 0,
                "Whales": r["n_whales"],
                "Time (s)": r["time"],
            }
        )

    if not opt_rows:
        st.warning("No optimization results available.")
        return

    odf = pd.DataFrame(opt_rows)

    # Main results table — use only columns that exist
    display_cols = [
        "Venue",
        "Asset",
        "B* ($/wk)",
        "r_max*",
        "Base APY",
        "Incentive @ Target TVL",
        "Total APR @ Target",
        "T_bind ($M)",
        "Mean TVL ($M)",
        "Budget Util",
        "Type",
        "Whales",
    ]
    # Filter to columns that actually exist in the DataFrame
    display_cols = [c for c in display_cols if c in odf.columns]
    fmt = {
        "B* ($/wk)": "${:,.0f}",
        "r_max*": "{:.2%}",
        "Base APY": "{:.2%}",
        "Incentive @ Target TVL": "{:.2%}",
        "Total APR @ Target": "{:.2%}",
        "T_bind ($M)": "${:.1f}M",
        "Mean TVL ($M)": "${:.0f}M",
        "Budget Util": "{:.1%}",
    }
    fmt = {k: v for k, v in fmt.items() if k in display_cols}
    st.dataframe(
        odf[display_cols].style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    # ── Per-Venue Calibration Table ──
    venue_cals = st.session_state.get("venue_calibrations", {})
    if venue_cals:
        with st.expander("📈 Per-Venue Calibrated Parameters", expanded=False):
            cal_rows = []
            for vname, cal in venue_cals.items():
                cal_rows.append(
                    {
                        "Venue": vname,
                        "α+ (inflow)": cal.alpha_plus,
                        "α- mult": cal.alpha_minus_multiplier,
                        "σ (noise)": cal.diffusion_sigma,
                        "Lag (days)": cal.response_lag_days,
                        "r_thresh mean": cal.r_threshold_mean,
                        "Quality": cal.data_quality,
                        "Observations": cal.n_observations,
                    }
                )
            # Add fallback venues
            for v in venues:
                if v["name"] not in venue_cals:
                    cal_rows.append(
                        {
                            "Venue": v["name"],
                            "α+ (inflow)": alpha_plus,
                            "α- mult": alpha_minus_mult,
                            "σ (noise)": diffusion_sigma,
                            "Lag (days)": response_lag,
                            "r_thresh mean": 0.0,
                            "Quality": "sidebar (fallback)",
                            "Observations": 0,
                        }
                    )
            st.dataframe(
                pd.DataFrame(cal_rows).style.format(
                    {
                        "α+ (inflow)": "{:.3f}",
                        "α- mult": "{:.1f}",
                        "σ (noise)": "{:.4f}",
                        "Lag (days)": "{:.0f}",
                        "r_thresh mean": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ── Budget allocation ──
    total_opt = sum(row["B* ($/wk)"] for row in opt_rows)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.metric("Optimizer Total", f"${total_opt:,.0f}/wk")
    with bc2:
        st.metric("Budget Envelope", f"${total_budget:,.0f}/wk")
    with bc3:
        diff = total_budget - total_opt
        st.metric("Remaining", f"${diff:+,.0f}/wk")

    # ── Actionable Merkl Campaign Instructions ──
    st.subheader("🎯 Merkl Campaign Instructions")
    st.caption(
        "Copy these exact values into Merkl when setting up each campaign. "
        "Campaign Type is derived from where T_bind sits relative to target TVL."
    )

    # ── Risk Assessment ──
    risk_alerts = []
    for row in opt_rows:
        vname = row["Venue"]
        r = results.get(vname, {})
        ov = r.get("overrides", {})
        base = row["Base APY"]
        B_star = row["B* ($/wk)"]
        r_star = row["r_max*"]
        total_apr_target = row["Total APR @ Target"]
        venue_floor = ov.get("floor_apr", 0.0)
        venue_sens = ov.get("apr_sensitivity", 0.0)
        forced_info = r.get("forced_rate_info")

        venue_risks = []

        # Risk 1: Forced rate requires overspend
        if forced_info and forced_info.get("overspend"):
            venue_risks.append(
                f"🔴 **Budget overspend required:** Forced rate {forced_info['forced_rate']:.2%} "
                f"needs ${forced_info['required_budget']:,.0f}/wk but total program budget is "
                f"${forced_info['input_budget']:,.0f}/wk "
                f"(+${forced_info['overspend_amount']:,.0f}/wk over budget)"
            )

        # Risk 2: Zero or near-zero budget → venue effectively sacrificed
        if B_star < 1_000 and not (ov.get("forced_rate") and ov["forced_rate"] == 0):
            venue_risks.append(
                f"🔴 **Venue sacrificed:** Budget allocation is only ${B_star:,.0f}/wk — "
                f"effectively zero incentives. This venue will receive no meaningful incentive APR. "
                f"Consider increasing total budget or reducing allocations to other venues."
            )

        # Risk 3: Floor APR cannot be maintained
        if venue_floor > 0 and total_apr_target < venue_floor:
            gap = venue_floor - total_apr_target
            venue_risks.append(
                f"🟠 **Floor APR at risk:** Total APR at target TVL ({total_apr_target:.2%}) "
                f"is below the floor APR ({venue_floor:.2%}) by {gap:.2%}. "
                + (
                    f"APR sensitivity is {venue_sens:.0%} — "
                    f"{'high risk of rapid TVL unwind from APY-sensitive depositors.' if venue_sens > 0.5 else 'moderate risk of gradual TVL loss.'} "
                    if venue_sens > 0
                    else ""
                )
                + "Consider increasing budget allocation or forced incentive rate for this venue."
            )

        # Risk 4: Total APR below r_threshold → venue is uncompetitive
        r_thresh = ov.get("r_threshold", 0.045)
        if total_apr_target < r_thresh:
            venue_risks.append(
                f"🟠 **Below competitor rate:** Total APR at target ({total_apr_target:.2%}) "
                f"< r_threshold ({r_thresh:.2%}). This venue will be uncompetitive and "
                f"likely lose TVL to competitors. Consider increasing budget or lowering target TVL."
            )

        # Risk 5: MC simulation shows low budget utilization
        mc = r.get("surface", None)
        if mc:
            mc_result = mc.optimal_mc_result
            if mc_result and mc_result.mean_budget_util < 0.5:
                venue_risks.append(
                    f"🟡 **Low budget utilization:** Only {mc_result.mean_budget_util:.0%} of allocated "
                    f"budget is expected to be spent on average. The r_max cap binds frequently, "
                    f"meaning TVL is too low to deploy the full budget. Consider lowering r_max or target TVL."
                )

        # Risk 6: Incentive APR at current TVL is very high (mercenary magnet)
        inc_at_current = apr_at_tvl(B_star, r.get("venue", {}).get("current_tvl", 0), r_star)
        if inc_at_current > 0.10:
            venue_risks.append(
                f"🟡 **High incentive at current TVL:** {inc_at_current:.2%} incentive APR at current TVL "
                f"may attract mercenary capital. Consider whether this spike is sustainable or if a "
                f"lower r_max would smooth the transition."
            )

        if venue_risks:
            risk_alerts.append((vname, venue_risks))

    # Show risk summary banner BEFORE individual Merkl instructions
    if risk_alerts:
        st.markdown("### ⚠️ Risk Assessment")
        for vname, risks in risk_alerts:
            with st.container(border=True):
                st.markdown(f"**{vname}**")
                for risk in risks:
                    st.markdown(risk)
        st.markdown("---")

    for row in opt_rows:
        ctype = row["Type"]
        vname = row["Venue"]
        r = results.get(vname, {})
        ov = r.get("overrides", {})
        merkl_type = "Hybrid" if ctype == "Hybrid" else ("MAX" if ctype == "MAX-like" else "Float")

        # Determine if this venue has any risks
        venue_has_risks = any(vn == vname for vn, _ in risk_alerts)

        with st.container(border=True):
            # Header with risk indicator
            if venue_has_risks:
                st.markdown(f"**{row['Venue']}** ({row['Asset']}) ⚠️")
            else:
                st.markdown(f"**{row['Venue']}** ({row['Asset']}) ✅")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"🔧 **Campaign Type:** `{merkl_type}`")
            with m2:
                st.markdown(f"💰 **Weekly Budget:** `${row['B* ($/wk)']:,.0f}`")
            with m3:
                if merkl_type == "MAX":
                    st.markdown(f"📊 **Incentive Rate:** `{row['r_max*']:.2%}`")
                else:
                    st.markdown(f"📊 **Max Incentive Rate (r_max):** `{row['r_max*']:.2%}`")
            detail_cols = st.columns(4)
            with detail_cols[0]:
                st.caption(f"Base APY: {row['Base APY']:.2%}")
            with detail_cols[1]:
                st.caption(f"Incentive @ Target: {row['Incentive @ Target TVL']:.2%}")
            with detail_cols[2]:
                st.caption(f"Total APR @ Target: {row['Total APR @ Target']:.2%}")
            with detail_cols[3]:
                st.caption(f"T_bind: ${row['T_bind ($M)']:.0f}M")

            # Inline risk notes per venue
            forced_info = r.get("forced_rate_info")
            if forced_info:
                if forced_info.get("overspend"):
                    st.error(
                        f"💸 **Budget gap:** Forced rate {forced_info['forced_rate']:.2%} requires "
                        f"${forced_info['required_budget']:,.0f}/wk — "
                        f"${forced_info['overspend_amount']:,.0f}/wk over the program budget. "
                        f"Increase budget or accept overspend."
                    )
                else:
                    st.success(
                        f"✅ Forced rate {forced_info['forced_rate']:.2%} is feasible within budget "
                        f"(requires ${forced_info['required_budget']:,.0f}/wk)"
                    )

    # ── Efficiency ──
    st.subheader("Efficiency Metrics")
    eff_cols = ["Venue", "Asset", "B* ($/wk)", "Net Supply ($M)", "TVL/$inc", "NS/$inc"]
    st.dataframe(
        odf[eff_cols]
        .style.format(
            {
                "B* ($/wk)": "${:,.0f}",
                "Net Supply ($M)": "${:.0f}M",
                "TVL/$inc": "{:,.0f}",
                "NS/$inc": "{:,.0f}",
            }
        )
        .background_gradient(subset=["TVL/$inc"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Per-venue detail ──
    st.subheader("Per-Venue Detail")
    for v in venues:
        if v["name"] not in results:
            continue
        r = results[v["name"]]
        sr = r["surface"]
        ov = r["overrides"]
        base = ov["base_apy"]
        B_star = r.get("adjusted_B", sr.optimal_B)
        r_star = r.get("adjusted_r_max", sr.optimal_r_max)

        # MC diagnostics at the allocated point
        i_alloc = int(np.argmin(np.abs(sr.grid.B_values - B_star)))
        j_alloc = int(np.argmin(np.abs(sr.grid.r_max_values - r_star)))
        mc = sr.mc_results.get((i_alloc, j_alloc)) or sr.optimal_mc_result
        alloc_loss = (
            float(sr.loss_surface[i_alloc, j_alloc])
            if sr.loss_surface.size > 0
            else sr.optimal_loss
        )
        alloc_t_bind = t_bind(B_star, r_star)

        with st.expander(f"🔬 {v['name']} ({r['n_whales']} whales)"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("B*", f"${B_star:,.0f}/wk")
            with d2:
                st.metric("r_max*", f"{r_star:.2%}")
            with d3:
                st.metric("T_bind", f"${alloc_t_bind / 1e6:.1f}M")
            with d4:
                st.metric("Loss", f"{alloc_loss:.3e}")

            if mc:
                d5, d6, d7, d8 = st.columns(4)
                with d5:
                    st.metric("Mean Total APR", f"{mc.mean_apr:.2%}")
                with d6:
                    st.metric("Mean Incentive APR", f"{mc.mean_incentive_apr:.2%}")
                with d7:
                    st.metric("APR Range (p5–p95)", f"{mc.apr_p5:.1%} – {mc.apr_p95:.1%}")
                with d8:
                    st.metric("Mean TVL", f"${mc.mean_tvl / 1e6:.0f}M")

            # APR at key TVL levels
            st.markdown("**Incentive APR at Key TVL Levels:**")
            tvl_pts = {
                "Current TVL": v["current_tvl"],
                "Target TVL": ov["target_tvl"],
                "T_bind": t_bind(B_star, r_star),
                "80% Current": v["current_tvl"] * 0.8,
                "120% Current": v["current_tvl"] * 1.2,
            }
            apr_rows = []
            for label, tvl_val in tvl_pts.items():
                inc = apr_at_tvl(B_star, tvl_val, r_star)
                regime = "🔒 Cap binds" if tvl_val < t_bind(B_star, r_star) else "📈 Float"
                apr_rows.append(
                    {
                        "Level": label,
                        "TVL ($M)": f"${tvl_val / 1e6:,.1f}M",
                        "Incentive APR": f"{inc:.2%}",
                        "Total APR": f"{(base + inc):.2%}",
                        "Regime": regime,
                    }
                )
            st.dataframe(apr_rows, use_container_width=True, hide_index=True)

            # Sensitivity
            sa = sr.sensitivity_analysis()
            st.markdown(f"**Sensitivity:** {sa['interpretation']}")

            # Duality
            dual = sr.duality_map(0.05)
            if len(dual) > 1:
                st.markdown(f"**{len(dual)} near-optimal configs** (within 5%):")
                drows = []
                for d in dual[:6]:
                    inc = apr_at_tvl(d["B"], ov["target_tvl"], d["r_max"])
                    drows.append(
                        {
                            "B": f"${d['B']:,.0f}",
                            "r_max": f"{d['r_max']:.2%}",
                            "T_bind": f"${d['t_bind'] / 1e6:.1f}M",
                            "Incentive @ Target": f"{inc:.2%}",
                            "vs Optimal": f"+{(d['loss_ratio'] - 1) * 100:.1f}%",
                        }
                    )
                st.dataframe(drows, use_container_width=True, hide_index=True)

            # Loss surface plot
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import LogNorm

                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                Bv, rv = sr.grid.B_values, sr.grid.r_max_values
                L = np.where(sr.feasibility_mask, sr.loss_surface, np.nan)
                oi, oj = sr.optimal_indices
                vals = L[~np.isnan(L)]
                norm = None
                if len(vals) > 0 and vals.max() / max(vals.min(), 1e-10) > 100:
                    norm = LogNorm(vmin=max(vals.min(), 1e-10), vmax=vals.max())
                im = ax.pcolormesh(
                    rv * 100, Bv / 1000, L, cmap="viridis_r", norm=norm, shading="nearest"
                )
                fig.colorbar(im, ax=ax, label="Loss", shrink=0.8)
                ax.plot(
                    rv[oj] * 100,
                    Bv[oi] / 1000,
                    "*",
                    color="red",
                    ms=14,
                    mec="white",
                    mew=1.5,
                    label="Unconstrained Opt",
                )
                # Mark the allocated point (after budget constraint)
                ax.plot(
                    r_star * 100,
                    B_star / 1000,
                    "D",
                    color="cyan",
                    ms=10,
                    mec="white",
                    mew=1.5,
                    label="Allocated",
                )
                ax.set_xlabel("r_max — Incentive APR Cap (%)")
                ax.set_ylabel("B — Weekly Budget ($k)")
                ax.set_title(f"Loss Surface — {v['name']}")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.info("Install matplotlib for surface plots.")

    # ── Export ──
    st.markdown("---")
    st.subheader("📤 Export")
    export = {
        "program": program_name,
        "total_budget": total_budget,
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC"),
        "simulation": {"n_paths": n_paths, "horizon_days": HORIZON_DAYS},
        "venues": [],
    }
    for row in opt_rows:
        ov = overrides[row["Venue"]]
        export["venues"].append(
            {
                "name": row["Venue"],
                "asset": row["Asset"],
                "weekly_budget": round(row["B* ($/wk)"]),
                "r_max": round(row["r_max*"], 4),
                "base_apy": round(row["Base APY"], 4),
                "incentive_at_target_tvl": round(row["Incentive @ Target TVL"], 4),
                "total_apr_at_target": round(row["Total APR @ Target"], 4),
                "t_bind": round(row["T_bind ($M)"] * 1e6),
                "campaign_type": row["Type"],
                "target_tvl": round(ov["target_tvl"]),
                "target_utilization": round(ov["target_util"], 3),
                "net_supply": round(ov["target_tvl"] * (1 - ov["target_util"])),
                "tvl_per_incentive": round(row["TVL/$inc"]),
                "ns_per_incentive": round(row["NS/$inc"]),
                "n_whale_profiles": row["Whales"],
            }
        )
    st.code(json.dumps(export, indent=2), language="json")


# ============================================================================
# TAB 2: SINGLE-VENUE OPTIMIZATION
# ============================================================================


def _run_single_venue_tab(
    *,
    programs,
    n_paths,
    w_spend,
    w_spend_waste,
    w_apr_var,
    w_apr_ceil,
    w_tvl_short,
    w_mercenary,
    w_whale_proximity,
    w_apr_floor,
    w_budget_waste,
    apr_target_mult,
    apr_ceiling_val,
    alpha_plus,
    alpha_minus_mult,
    response_lag,
    diffusion_sigma,
    merc_entry_thresh,
    merc_exit_thresh,
    merc_max_frac,
    grid_r_min,
    grid_r_max,
):
    """Fully standalone single-venue optimization workflow."""

    st.header("🔬 Single-Venue Optimization")
    st.caption(
        "Run the full MC optimization framework on a single venue with a fixed budget. "
        "This is completely independent of the multi-venue program optimization — "
        "pick any venue, set your budget, and explore the loss surface."
    )

    # ── Venue selection ──
    # Build a flat list of all venues across all programs
    all_venues = []
    venue_labels = []
    for prog_name, prog in programs.items():
        for v in prog["venues"]:
            all_venues.append(v)
            venue_labels.append(f"{v['name']}  ({prog_name})")

    sv_sel_idx = st.selectbox(
        "Select Venue",
        range(len(venue_labels)),
        format_func=lambda i: venue_labels[i],
        key="sv_venue_sel",
    )
    sv_venue = all_venues[sv_sel_idx]

    # ── Detect venue change → clear stale results ──
    # Widget values are handled by embedding venue name in each key
    # (see _svk prefix below), so no need to pop/rerun for widget state.
    _sv_venue_id = sv_venue["name"]
    if st.session_state.get("_sv_last_venue") != _sv_venue_id:
        # Clear stale results / live-fetch artefacts from previous venue
        st.session_state.pop("sv_result", None)
        st.session_state.pop("sv_live_tvl", None)
        st.session_state.pop("sv_live_util", None)
        st.session_state["_sv_last_venue"] = _sv_venue_id

    # ── Dynamic key prefix — forces fresh widgets on venue switch ──
    _svk = f"sv_{_sv_venue_id}_"

    # ── Auto-fetch live data on venue selection ──
    # Fetches TVL, base APY, and r_threshold automatically when a venue is
    # first selected (or switched to). Results are cached in session_state
    # per venue so they don't re-fetch on every rerun.
    _sv_autofetch_key = f"sv_autofetched_{_sv_venue_id}"
    if _sv_autofetch_key not in st.session_state:
        with st.spinner(f"📡 Fetching live data for {sv_venue['name']}..."):
            # 1. Fetch TVL + utilization
            try:
                _af_tvl, _af_util = _fetch_current_tvl_and_util(sv_venue)
            except Exception as _e:
                _af_tvl = sv_venue.get("current_tvl", 0.0)
                _af_util = sv_venue.get("target_util", 0.5)
                st.warning(f"⚠️ Live TVL fetch failed: {_e}")

            # Store in the same cache the multi-venue tab uses
            _sv_prog_name_af = next(
                (
                    pn
                    for pn, p in programs.items()
                    for vv in p["venues"]
                    if vv["name"] == sv_venue["name"]
                ),
                None,
            )
            if _sv_prog_name_af:
                _af_cache_key = f"current_values_{_sv_prog_name_af}"
                _af_existing = st.session_state.get(_af_cache_key, {})
                _af_existing[sv_venue["name"]] = {
                    "current_tvl": _af_tvl,
                    "current_util": _af_util,
                }
                st.session_state[_af_cache_key] = _af_existing

            # 2. Fetch base APY
            _af_base_key = f"sv_base_apy_{sv_venue['name']}"
            try:
                _af_base_result = fetch_all_base_apys([sv_venue])
                _af_r = _af_base_result.get(sv_venue["name"])
                if _af_r:
                    st.session_state[_af_base_key] = _af_r.base_apy
                    st.session_state[f"{_af_base_key}_source"] = _af_r.source
                else:
                    st.session_state[_af_base_key] = 0.0
                    st.session_state[f"{_af_base_key}_source"] = "failed"
            except Exception as _e:
                st.session_state[_af_base_key] = 0.0
                st.session_state[f"{_af_base_key}_source"] = f"error: {_e}"

            # 3. Fetch r_threshold
            _af_rthresh_key = f"sv_rthresh_{sv_venue['asset']}"
            try:
                _af_prog = next(
                    (
                        pn
                        for pn, p in programs.items()
                        for v in p["venues"]
                        if v["name"] == sv_venue["name"]
                    ),
                    None,
                )
                _af_rthresh_data = fetch_dynamic_r_threshold(
                    sv_venue["asset"],
                    program_name=_af_prog,
                )
                st.session_state[_af_rthresh_key] = _af_rthresh_data
            except Exception as _e:
                st.session_state[_af_rthresh_key] = {
                    "r_threshold": 0.045,
                    "source": "fallback",
                    "error": str(_e),
                }

            st.session_state[_sv_autofetch_key] = True

    # ── Look up program + cached live data for this venue ──
    _sv_prog_name = next(
        (pn for pn, p in programs.items() for vv in p["venues"] if vv["name"] == sv_venue["name"]),
        None,
    )
    _sv_cache_key = f"current_values_{_sv_prog_name}" if _sv_prog_name else None
    _sv_cached = (
        st.session_state.get(_sv_cache_key, {}).get(sv_venue["name"], {}) if _sv_cache_key else {}
    )
    # Prefer cached live values (matching multi-venue auto-fill behavior),
    # fall back to hardcoded PROGRAMS dict defaults.
    _sv_default_current_tvl = _sv_cached.get("current_tvl", sv_venue["current_tvl"]) / 1e6
    _sv_default_target_tvl = _sv_cached.get("current_tvl", sv_venue["target_tvl"]) / 1e6
    _sv_default_util = _sv_cached.get("current_util", sv_venue.get("target_util", 0.5)) * 100
    _sv_base_preview_key = f"sv_base_apy_{sv_venue['name']}"
    _sv_base_preview = st.session_state.get(_sv_base_preview_key, 0.0)

    # Supply cap: use live-fetched or venue config
    _sv_default_supply_cap = sv_venue.get("supply_cap", 0.0)

    # Show auto-fill summary
    _sv_rthresh_key_af = f"sv_rthresh_{sv_venue['asset']}"
    _sv_rthresh_data_af = st.session_state.get(_sv_rthresh_key_af, {})
    _sv_rthresh_val_af = _sv_rthresh_data_af.get("r_threshold", 0.045)
    _sv_base_source_af = st.session_state.get(f"{_sv_base_preview_key}_source", "not fetched")
    if _sv_cached or _sv_base_preview > 0:
        st.success(
            f"✅ Auto-filled from live data — "
            f"TVL: ${_sv_default_current_tvl:.1f}M, "
            f"Util: {_sv_default_util:.0f}%, "
            f"Base APY: {_sv_base_preview:.2%} ({_sv_base_source_af}), "
            f"r_threshold: {_sv_rthresh_val_af:.2%}"
        )

    # ── Venue parameters ──
    st.subheader("⚙️ Venue Parameters")

    # ── TVL / Util row (always visible) ──
    sv_tvl1, sv_tvl2, sv_tvl3, sv_tvl4 = st.columns(4)
    with sv_tvl1:
        sv_target_tvl = (
            st.number_input(
                "Target TVL ($M)",
                value=_sv_default_target_tvl,
                step=10.0,
                min_value=0.1,
                key=_svk + "target_tvl",
                help=(
                    "**What it is:** The TVL level you want this venue to reach or maintain. "
                    "The optimizer evaluates loss at both current and target TVL.\n\n"
                    "**How to configure:** Set to your growth goal. If TVL is already where "
                    "you want it, set equal to Current TVL (maintenance mode). If you're "
                    "bootstrapping, set higher than current. Auto-filled from live data."
                ),
            )
            * 1e6
        )
    with sv_tvl2:
        sv_current_tvl = (
            st.number_input(
                "Current TVL ($M)",
                value=_sv_default_current_tvl,
                step=10.0,
                min_value=0.1,
                key=_svk + "current_tvl",
                help=(
                    "**What it is:** The venue's current total value locked — the simulation "
                    "starting point.\n\n"
                    "**How to configure:** Auto-filled from live on-chain data on venue "
                    "selection. Override manually if you want to simulate from a different "
                    "starting TVL (e.g., stress-testing a drawdown scenario)."
                ),
            )
            * 1e6
        )
    with sv_tvl3:
        sv_target_util = (
            st.number_input(
                "Target Util (%)",
                value=_sv_default_util,
                step=1.0,
                min_value=0.0,
                max_value=100.0,
                key=_svk + "target_util",
                help=(
                    "**What it is:** Target utilization rate (borrowed / supplied). Affects "
                    "the effective incentive rate because incentives are only paid on the "
                    "utilized portion.\n\n"
                    "**How to configure:** Auto-filled from live data. For lending venues, "
                    "this is the borrow-to-supply ratio (typically 50–90%). For DEX pools "
                    "this is fixed at 50%. Higher util means each dollar of incentive "
                    "generates more effective APR."
                ),
            )
            / 100.0
        )
    with sv_tvl4:
        sv_paths = st.number_input(
            "MC Paths",
            value=n_paths,
            step=10,
            min_value=10,
            key=_svk + "mc",
            help=(
                "**What it is:** Number of Monte Carlo simulation paths. Each path is "
                "an independent random walk of TVL and whale events.\n\n"
                "**How to configure:** 50+ for quick exploration, 100+ for final "
                "decisions. More paths = more stable results but slower computation. "
                "The p5/p50/p95 percentiles in results converge around 80–100 paths."
            ),
        )

    # ── Supply Cap ──
    sv_supply_cap = (
        st.number_input(
            "Supply Cap ($M) — 0 = unlimited",
            value=_sv_default_supply_cap / 1e6 if _sv_default_supply_cap > 0 else 0.0,
            step=10.0,
            min_value=0.0,
            key=_svk + "supply_cap",
            help=(
                "Maximum TVL the venue can accept (protocol-level supply cap). 0 = unlimited.\n\n"
                "**What it does:** Limits how much capital can flow into this pool. "
                "The simulation uses this as a hard ceiling on TVL growth.\n\n"
                "**How to configure:** Set this to the protocol's on-chain supply cap. "
                "If the protocol has no cap, leave at 0. The 'Suggested TVL Cap' in the "
                "results section will advise whether this should be tightened."
            ),
        )
        * 1e6
    )

    # ==================================================================
    # MODE SELECTOR: Budget vs Set APR
    # ==================================================================
    st.markdown("---")
    sv_mode = st.radio(
        "**Optimization Mode**",
        ["💰 Set Budget", "📊 Set APR"],
        key=_svk + "mode",
        horizontal=True,
        help=(
            "**Set Budget:** You specify a weekly budget ceiling. The optimizer "
            "finds the best (B, r_max) pair within that budget.\n\n"
            "**Set APR:** You specify the total APR you want maintained at both "
            "current and target TVL. Budget and r_max are derived automatically. "
            "The optimizer ensures this APR is maintained throughout the campaign."
        ),
    )

    _is_set_apr_mode = "Set APR" in sv_mode

    WEEKS_PER_YEAR_UI = 365.0 / 7.0

    if _is_set_apr_mode:
        # ── SET APR MODE ──
        # User specifies the total APR they want maintained.
        # The optimizer searches for the MINIMUM budget that maintains this APR
        # across current TVL, target TVL, and random walk fluctuations.
        st.info(
            "📊 **Set APR Mode — Budget Minimizer:** Specify the total APR you want "
            "maintained. The optimizer searches for the **cheapest** (B, r_max) pair "
            "that keeps total APR ≥ your target throughout the simulation — including "
            "whale exits, TVL fluctuations, and competitor dynamics.\n\n"
            "💡 The derived budget shown below is the **theoretical maximum** needed. "
            "The optimizer will try to find a cheaper configuration that still meets "
            "the APR target."
        )

        sv_apr_c1, sv_apr_c2, sv_apr_c3, sv_apr_c4 = st.columns(4)
        with sv_apr_c1:
            sv_set_apr_total = (
                st.number_input(
                    "Target Total APR (%)",
                    value=5.0,
                    step=0.25,
                    min_value=0.5,
                    key=_svk + "set_apr",
                    help=(
                        "**What it is:** The total APR (base + incentive) you want depositors "
                        "to earn at both current and target TVL.\n\n"
                        "**How it works:** The optimizer finds the minimum weekly budget that "
                        "maintains this total APR throughout the simulation — including whale "
                        "exits, TVL fluctuations, and competitor dynamics.\n\n"
                        "**How to configure:** Set this to the rate you need to attract/retain "
                        "capital. Check r_threshold for competitive context — your target should "
                        "be at or above the competitor rate to attract TVL."
                    ),
                )
                / 100.0
            )
        with sv_apr_c2:
            sv_apr_floor_input = (
                st.number_input(
                    "Floor APR (%)",
                    value=sv_set_apr_total * 100,
                    step=0.25,
                    min_value=0.0,
                    key=_svk + "set_apr_floor",
                    help=(
                        "**What it is:** The minimum acceptable total APR. When APR drops below "
                        "this floor, the optimizer penalizes that configuration heavily.\n\n"
                        "**How it works:** This creates a soft or hard lower bound (controlled by "
                        "Floor Strictness). The optimizer avoids configs where the APR "
                        "frequently dips below this level across Monte Carlo paths.\n\n"
                        "**How to configure:**\n"
                        "- Set equal to Target APR for a hard minimum (APR must never drop below)\n"
                        "- Set slightly below Target for a soft buffer (brief dips tolerated)\n"
                        "- Set to 0 to disable floor enforcement entirely\n\n"
                        "**When it matters:** APY-sensitive vaults with loopers/leveraged positions "
                        "need a tight floor — these depositors will exit immediately if the rate drops. "
                        "Vaults with sticky institutional capital can tolerate a looser floor."
                    ),
                )
                / 100.0
            )
        with sv_apr_c3:
            sv_apr_ceiling_total = (
                st.number_input(
                    "Total APR Ceiling (%)",
                    value=min(10.0, sv_set_apr_total * 100 * 2),
                    step=0.5,
                    min_value=1.0,
                    key=_svk + "apr_ceiling",
                    help=(
                        "**What it is:** The maximum incentive rate cap (r_max). Total APR above "
                        "this is penalized to prevent attracting mercenary capital.\n\n"
                        "**How it works:** This controls the Float↔MAX regime spectrum:\n\n"
                        "**Closer to Target APR → MAX-like (rate stability):**\n"
                        "• Relentless TVL growth is the priority, budget constraints are loose\n"
                        "• Target TVL is near the vault cap, no reason to taper\n"
                        "• Competing directly with a known competitor rate — any wobble loses capital\n"
                        "• Bootstrapping a new vault from near zero — need a reliable rate signal\n"
                        "• APY-sensitive strategies (loopers) need rate certainty for profitability\n"
                        "• High conviction on the rate — just want to pay it, period\n"
                        "• Want MAX-like stability but finite B gives spend protection if TVL cap is raised\n\n"
                        "**Higher → Float-like (spend efficiency):**\n"
                        "• TVL already well above target, maintenance mode — only paying what's needed\n"
                        "• Want to shake out mercenary capital — let weak hands leave at lower rates\n"
                        "• Base APY is strong, only need a small incentive nudge\n"
                        "• Winding down incentives — want graceful taper as TVL stabilises\n"
                        "• Budget is tight — need spend to scale down automatically as pool grows\n"
                        "• Want price discovery on the minimum APR that holds TVL\n"
                        "• Transitioning toward less-incentive dependency over time"
                    ),
                )
                / 100.0
            )
        with sv_apr_c4:
            sv_set_apr_sensitivity = st.slider(
                "Floor Strictness",
                min_value=0.5,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key=_svk + "apr_sens",
                help=(
                    "**What it is:** How strictly the APR floor is enforced in the optimizer. "
                    "Controls the penalty weight on APR floor breaches.\n\n"
                    "**How it works:** Higher values penalize configurations where APR dips "
                    "below the floor more heavily. The optimizer avoids those configurations.\n\n"
                    "**How to configure:**\n"
                    "- **0.9–1.0 (Hard floor):** For vaults with active loopers or leveraged "
                    "strategies that WILL exit if APR drops beneath their profitability "
                    "threshold. These depositors act immediately — any dip is costly.\n"
                    "- **0.7–0.9 (Competitive floor):** For vaults competing with a known "
                    "competitor rate. You want to stay competitive but can tolerate brief "
                    "dips during TVL fluctuations. Research has flagged a competitor, and "
                    "you want to keep rates above theirs most of the time.\n"
                    "- **0.5–0.7 (Soft floor):** For vaults with sticky, rate-insensitive "
                    "capital (e.g., institutional, treasury). Depositors won't react quickly "
                    "to small APR drops. Allows the optimizer more budget flexibility."
                ),
            )

        # Derive budget ceiling and reference rate from the set APR
        _sv_set_inc_rate = max(0.005, sv_set_apr_total - _sv_base_preview)
        _sv_ref_tvl = max(sv_current_tvl, sv_target_tvl)
        _sv_derived_budget = _sv_ref_tvl * _sv_set_inc_rate / WEEKS_PER_YEAR_UI

        st.markdown(
            f"**Derived Parameters:**\n"
            f"- Base APY: **{_sv_base_preview:.2%}** → "
            f"Required incentive rate: **{_sv_set_inc_rate:.2%}**\n"
            f"- Reference TVL (max of current/target): "
            f"**${_sv_ref_tvl / 1e6:.1f}M**\n"
            f"- Budget search ceiling: **${_sv_derived_budget * 1.1:,.0f}/wk** "
            f"(derived ${_sv_derived_budget:,.0f}/wk + 10% headroom)\n"
            f"- r_max search range: **{_sv_set_inc_rate:.2%}** – "
            f"**{sv_apr_ceiling_total:.2%}** (incentive, protocol-clamped)\n"
            f"- Optimizer objective: **minimize budget** while keeping "
            f"total APR ≥ {sv_set_apr_total:.2%}"
        )

        if _sv_base_preview <= 0:
            st.warning(
                "⚠️ Base APY not fetched yet — derived budget assumes 0% base. "
                "Fetch base APY below for accurate derivation."
            )

        # Map Set APR controls to the variables consumed by the optimizer
        # KEY DESIGN: We do NOT set forced_rate. Instead, we let the optimizer
        # search a full (B, r_max) grid with:
        #   - Budget range: floor-aware minimum up to derived ceiling
        #   - r_max range: from required incentive rate to ceiling
        #   - High w_spend (×5) so optimizer prefers minimum budget
        #   - High w_apr_floor (≥12) with high sensitivity so APR target is enforced
        # The optimizer finds the CHEAPEST config that maintains the APR floor.
        sv_budget = _sv_derived_budget * 1.1  # Budget ceiling (not pinned)
        sv_floor_apr = sv_apr_floor_input
        sv_apr_sensitivity = sv_set_apr_sensitivity
        sv_forced_rate = None  # Let optimizer search — floor APR enforces target
        sv_pin_b_val = None
        sv_pin_r_val = None
        # r_max must be ≥ required incentive rate, otherwise floor is impossible.
        # Use the target APR as the minimum total r_max (→ inc_rate after base_apy subtracted).
        sv_r_lo_total = sv_set_apr_total
        sv_r_hi_total = sv_apr_ceiling_total

    else:
        # ── SET BUDGET MODE (existing behavior) ──
        st.info(
            "💰 **Set Budget Mode:** You specify the weekly budget ceiling. "
            "The optimizer searches for the best (B, r_max) pair within that budget."
        )

        sv_b1, sv_b2, sv_b3 = st.columns(3)
        with sv_b1:
            sv_budget = st.number_input(
                "Weekly Budget ($)",
                value=50_000,
                step=5_000,
                min_value=1_000,
                key=_svk + "budget",
                help=(
                    "**What it is:** Your maximum weekly incentive spend for this venue. "
                    "The optimizer finds the best (Budget, r_max) pair where Budget ≤ this amount.\n\n"
                    "**How to configure:** Start with your total program budget divided across "
                    "venues by TVL share. The optimizer will tell you if the budget is sufficient "
                    "to maintain your desired APR at target TVL."
                ),
            )
        with sv_b2:
            sv_r_lo_total = (
                st.number_input(
                    "Total APR min (%)",
                    value=grid_r_min * 100,
                    step=0.5,
                    min_value=0.0,
                    key=_svk + "rlo",
                    help=(
                        "**What it is:** The lower bound of the r_max search grid (total APR = "
                        "base + incentive).\n\n"
                        "**How to configure:** Set to a competitive rate floor — what's the "
                        "minimum total APR that would realistically attract/retain capital? "
                        "Check r_threshold (competitor rate) for guidance."
                    ),
                )
                / 100.0
            )
        with sv_b3:
            sv_r_hi_total = (
                st.number_input(
                    "Total APR max (%)",
                    value=grid_r_max * 100,
                    step=0.5,
                    min_value=1.0,
                    key=_svk + "rhi",
                    help=(
                        "**What it is:** The upper bound of the r_max search grid (total APR = "
                        "base + incentive).\n\n"
                        "**How to configure:** Set high enough to explore generous rate scenarios "
                        "but not so high that it attracts unsustainable mercenary capital. "
                        "2× your target rate is usually a good ceiling."
                    ),
                )
                / 100.0
            )
        if _sv_base_preview > 0:
            st.caption(
                f"ℹ️ Base APY: {_sv_base_preview:.2%} → "
                f"Incentive r_max range: "
                f"{max(0, sv_r_lo_total - _sv_base_preview):.2%} – "
                f"{max(0, sv_r_hi_total - _sv_base_preview):.2%}"
            )
        else:
            st.caption("⚠️ Fetch base APY first for accurate total-to-incentive conversion.")

        # ── Optional Constraints (budget mode only) ──
        with st.expander("🔧 Advanced Constraints", expanded=False):
            sv_cc1, sv_cc2, sv_cc3 = st.columns(3)
            with sv_cc1:
                sv_pin_b = st.checkbox(
                    "Pin Budget?",
                    key=_svk + "pin_b",
                    help="Lock this venue's budget to a specific amount.",
                )
                sv_pin_b_val = None
                if sv_pin_b:
                    sv_pin_b_val = st.number_input(
                        "Pinned Budget ($/wk)",
                        value=50_000,
                        step=5000,
                        key=_svk + "pinbval",
                    )
            with sv_cc2:
                sv_pin_r = st.checkbox(
                    "Pin r_max?", key=_svk + "pin_r", help="Lock the APR cap (r_max)."
                )
                sv_pin_r_val = None
                if sv_pin_r:
                    sv_pin_r_val = (
                        st.number_input(
                            "Pinned r_max (%)",
                            value=6.0,
                            step=0.25,
                            min_value=0.0,
                            key=_svk + "pinrval",
                        )
                        / 100.0
                    )
            with sv_cc3:
                sv_force_rate = st.checkbox(
                    "Force incentive rate?",
                    key=_svk + "force_r",
                    help="Override optimizer — set a fixed incentive rate.",
                )
                sv_forced_rate = None
                if sv_force_rate:
                    sv_forced_rate = (
                        st.number_input(
                            "Forced rate (%)",
                            value=5.0,
                            step=0.25,
                            min_value=0.0,
                            key=_svk + "frate",
                        )
                        / 100.0
                    )

            # ── APY Sensitivity (floor APR) ──
            st.markdown("**APY Sensitivity** (optional):")
            sv_sens_c1, sv_sens_c2 = st.columns(2)
            with sv_sens_c1:
                sv_floor_apr = (
                    st.number_input(
                        "Floor APR (%)",
                        value=0.0,
                        step=0.25,
                        min_value=0.0,
                        key=_svk + "floor_apr",
                        help=(
                            "**What it is:** The minimum acceptable total APR. Configurations "
                            "where APR drops below this are penalized.\n\n"
                            "**How to configure:**\n"
                            "- Set above 0 when your vault has APY-sensitive depositors (loopers, "
                            "leveraged strategies) who will exit if APR drops below their "
                            "profitability threshold.\n"
                            "- Set to 0 to disable — appropriate for stable, rate-insensitive capital.\n"
                            "- Use r_threshold (competitor rate) as a guide for the minimum "
                            "rate you need to stay competitive."
                        ),
                    )
                    / 100.0
                )
            with sv_sens_c2:
                sv_apr_sensitivity = st.slider(
                    "APR Sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=_svk + "apr_sens2",
                    help=(
                        "**What it is:** How strictly the APR floor is enforced in the "
                        "optimizer. Controls the penalty weight on floor breaches.\n\n"
                        "**How to configure:**\n"
                        "- **0.8–1.0:** For vaults with active loopers or leveraged strategies. "
                        "These depositors exit immediately on APR drops.\n"
                        "- **0.5–0.8:** For competitive positioning — stay above a competitor rate "
                        "most of the time, tolerate brief dips during TVL fluctuations.\n"
                        "- **0.0–0.5:** For stable capital or when floor APR is informational only."
                    ),
                )

    # ── Pre-flight sanity check ──
    # Catch the case where current_tvl is wildly wrong (e.g. leftover from
    # a different venue due to Streamlit widget caching).
    _expected_tvl = _sv_default_current_tvl * 1e6  # in USD
    if _expected_tvl > 0 and sv_current_tvl > 0:
        _tvl_ratio = sv_current_tvl / _expected_tvl
        if _tvl_ratio > 2.0 or _tvl_ratio < 0.3:
            st.error(
                f"🚨 **Current TVL (${sv_current_tvl / 1e6:.1f}M) looks wrong** — "
                f"expected ~${_expected_tvl / 1e6:.1f}M for {sv_venue['name']}. "
                f"This can happen when switching venues. "
                f"Please correct the value or click '📡 Fetch Live TVL' below."
            )

    # ── Competitive landscape: sibling venues in the same program ──
    if _sv_prog_name:
        _sibling_venues = programs[_sv_prog_name]["venues"]
        if len(_sibling_venues) > 1:
            _sib_cache = st.session_state.get(f"current_values_{_sv_prog_name}", {})
            _base_cache_all = st.session_state.get("base_apys", {})
            _sib_rows = []
            _total_prog_tvl = 0.0
            for _sv in _sibling_venues:
                _sc = _sib_cache.get(_sv["name"], {})
                _sib_tvl = _sc.get("current_tvl", _sv.get("current_tvl", 0))
                _sib_util = _sc.get("current_util", _sv.get("target_util", 0))
                _sib_base_r = _base_cache_all.get(_sv["name"])
                _sib_base = _sib_base_r.base_apy if _sib_base_r else 0.0
                _total_prog_tvl += _sib_tvl
                _is_current = _sv["name"] == sv_venue["name"]
                _sib_rows.append(
                    {
                        "Venue": ("→ " if _is_current else "") + _sv["name"],
                        "TVL ($M)": _sib_tvl / 1e6,
                        "Util": _sib_util,
                        "Base APY": _sib_base,
                        "Share": 0.0,  # filled below
                    }
                )
            for _row in _sib_rows:
                _row["Share"] = _row["TVL ($M)"] * 1e6 / max(_total_prog_tvl, 1)
            with st.expander(
                f"📊 Program Context: {_sv_prog_name} — "
                f"${_total_prog_tvl / 1e6:.0f}M total {sv_venue['asset']} TVL",
                expanded=False,
            ):
                st.dataframe(
                    pd.DataFrame(_sib_rows).style.format(
                        {
                            "TVL ($M)": "${:,.1f}M",
                            "Util": "{:.1%}",
                            "Base APY": "{:.2%}",
                            "Share": "{:.1%}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    f"Total addressable {sv_venue['asset']} in this program: "
                    f"${_total_prog_tvl / 1e6:.0f}M across {len(_sibling_venues)} venues. "
                    f"Current venue is {sv_current_tvl / _total_prog_tvl * 100:.0f}% of program TVL."
                    if _total_prog_tvl > 0
                    else ""
                )

    # ── Fetch Live TVL ──
    sv_live_col1, sv_live_col2 = st.columns([1, 3])
    with sv_live_col1:
        if st.button("📡 Fetch Live TVL", key="sv_fetch_live_tvl"):
            with st.spinner(f"Fetching live TVL for {sv_venue['name']}..."):
                try:
                    live_tvl, live_util = _fetch_current_tvl_and_util(sv_venue)
                    st.session_state["sv_live_tvl"] = live_tvl
                    st.session_state["sv_live_util"] = live_util
                    st.success(f"Live TVL: ${live_tvl / 1e6:.1f}M, Util: {live_util:.1%}")
                except Exception as e:
                    st.warning(f"Live TVL fetch failed: {e}")
    with sv_live_col2:
        if "sv_live_tvl" in st.session_state:
            st.caption(
                f"Last fetched: TVL=${st.session_state['sv_live_tvl'] / 1e6:.1f}M, "
                f"Util={st.session_state.get('sv_live_util', 0):.1%}. "
                "Update 'Current TVL' above to use this value."
            )

    # ── Fetch base APY and r_threshold for this venue ──
    sv_fetch_col1, sv_fetch_col2 = st.columns(2)

    with sv_fetch_col1:
        st.subheader("📡 Base APY")
        sv_base_key = f"sv_base_apy_{sv_venue['name']}"
        if st.button("🔄 Fetch Base APY", key="sv_fetch_base"):
            with st.spinner(f"Fetching base APY for {sv_venue['name']}..."):
                try:
                    base_result = fetch_all_base_apys([sv_venue])
                    r = base_result.get(sv_venue["name"])
                    if r:
                        st.session_state[sv_base_key] = r.base_apy
                        st.session_state[f"{sv_base_key}_source"] = r.source
                    else:
                        st.session_state[sv_base_key] = 0.0
                        st.session_state[f"{sv_base_key}_source"] = "failed"
                except Exception as e:
                    st.warning(f"Base APY fetch failed: {e}")
                    st.session_state[sv_base_key] = 0.0
                    st.session_state[f"{sv_base_key}_source"] = "error"

        sv_base_apy = st.session_state.get(sv_base_key, 0.0)
        sv_base_source = st.session_state.get(f"{sv_base_key}_source", "not fetched")
        st.metric(
            "Base APY",
            f"{sv_base_apy:.2%}",
            help="**Base APY** is the venue's native yield BEFORE incentives — "
            "interest from borrowers, protocol fees, or LP trading fees. "
            "Total APR = Base APY + Incentive APR. Auto-fetched on venue selection; "
            "click 'Fetch Base APY' to refresh.",
        )
        if sv_base_source == "not fetched":
            st.caption("⚠️ Click 'Fetch Base APY' to load — required for accurate results.")
        else:
            st.caption(f"Source: {sv_base_source}")

    with sv_fetch_col2:
        st.subheader("📊 r_threshold")
        sv_rthresh_key = f"sv_rthresh_{sv_venue['asset']}"
        if st.button("🔄 Fetch r_threshold", key="sv_fetch_rthresh"):
            with st.spinner(f"Fetching competitor rates for {sv_venue['asset']}..."):
                try:
                    prog_name_for_venue = next(
                        pn
                        for pn, p in programs.items()
                        for v in p["venues"]
                        if v["name"] == sv_venue["name"]
                    )
                    data = fetch_dynamic_r_threshold(
                        sv_venue["asset"],
                        program_name=prog_name_for_venue,
                    )
                    st.session_state[sv_rthresh_key] = data
                except Exception as e:
                    st.warning(f"r_threshold fetch failed: {e}")
                    st.session_state[sv_rthresh_key] = {
                        "r_threshold": 0.045,
                        "source": "fallback",
                        "error": str(e),
                    }

        sv_rthresh_data = st.session_state.get(sv_rthresh_key, {"r_threshold": 0.045})
        sv_r_threshold = sv_rthresh_data.get("r_threshold", 0.045)
        sv_rthresh_source = sv_rthresh_data.get(
            "r_threshold_source", sv_rthresh_data.get("source", "unknown")
        )
        n_comp = len(sv_rthresh_data.get("competitors", []))
        st.metric(
            "r_threshold",
            f"{sv_r_threshold:.2%}",
            help="**r_threshold** is the best competing rate depositors could earn "
            "elsewhere for the same asset. Computed as the median of top competitor "
            "vault rates. Whales compare your total APR to r_threshold when deciding "
            "to stay or leave. Auto-fetched on venue selection.",
        )
        if sv_rthresh_key not in st.session_state:
            st.caption("⚠️ Click 'Fetch r_threshold' to load competitor rates.")
        else:
            st.caption(f"Source: {sv_rthresh_source} ({n_comp} competitors)")

    # ── Manual r_threshold override ──
    sv_rthresh_ov = st.number_input(
        "r_threshold Override (%) — leave 0 to use fetched",
        value=0.0,
        step=0.1,
        min_value=0.0,
        key=_svk + "rthresh_ov",
        help=(
            "**What it is:** Manually override the auto-fetched competitor rate.\n\n"
            "**How to configure:** Leave at 0 to use the fetched value. Set to a "
            "specific rate if you have private intelligence about competitor dynamics "
            "(e.g., a competitor is about to launch a new campaign, or you know "
            "the fetched rate is stale)."
        ),
    )
    if sv_rthresh_ov > 0:
        sv_r_threshold = sv_rthresh_ov / 100.0

    # ── Historical Data Calibration ──
    sv_pool_id = sv_venue.get("defillama_pool_id", "")
    if sv_pool_id:
        st.markdown("---")
        st.subheader("📈 Historical Data & Calibration")
        hist_key = f"sv_hist_{sv_pool_id}"
        cal_key = f"sv_cal_{sv_pool_id}"

        if st.button("📊 Fetch 90-Day History & Calibrate", key="sv_fetch_hist"):
            with st.spinner(f"Fetching DeFiLlama history for {sv_venue['name']}..."):
                try:
                    from campaign.historical import calibrate_retail_params, fetch_pool_history

                    hist = fetch_pool_history(sv_pool_id, days=90)
                    cal = calibrate_retail_params(hist, default_r_threshold=sv_r_threshold)
                    st.session_state[hist_key] = hist
                    st.session_state[cal_key] = cal
                    st.success(
                        f"✅ Fetched {hist.days} days of history. Calibration quality: {cal.data_quality}"
                    )
                except Exception as e:
                    st.warning(f"Historical data fetch failed: {e}")

        if hist_key in st.session_state:
            hist = st.session_state[hist_key]
            cal = st.session_state[cal_key]

            # Show calibrated parameters
            st.markdown("**Calibrated Parameters (from history):**")
            hc1, hc2, hc3, hc4 = st.columns(4)
            with hc1:
                st.metric(
                    "α+ (inflow)",
                    f"{cal.alpha_plus:.3f}",
                    help="Inflow elasticity calibrated from TVL ~ APR gap regression",
                )
            with hc2:
                st.metric(
                    "α- multiplier",
                    f"{cal.alpha_minus_multiplier:.1f}",
                    help="Outflow speed relative to inflow",
                )
            with hc3:
                st.metric(
                    "σ (noise)",
                    f"{cal.diffusion_sigma:.4f}",
                    help="Daily TVL volatility from residuals",
                )
            with hc4:
                st.metric(
                    "Lag (days)",
                    f"{cal.response_lag_days:.0f}",
                    help="Estimated depositor response delay",
                )

            st.caption(
                f"Data quality: **{cal.data_quality}** | {cal.n_observations} observations | "
                f"r_threshold mean: {cal.r_threshold_mean:.2%} | "
                f"r_threshold trend: {cal.r_threshold_trend * 365:.1%}/yr"
            )

            # ── Apply Calibrated Values button ──
            if cal.data_quality != "insufficient":
                if st.button("✅ Apply Calibrated Values to Sidebar", key="sv_apply_cal"):
                    st.session_state["alpha_plus"] = cal.alpha_plus
                    st.session_state["alpha_minus_mult"] = cal.alpha_minus_multiplier
                    st.session_state["diffusion_sigma"] = cal.diffusion_sigma
                    st.session_state["response_lag"] = cal.response_lag_days
                    # Also derive mercenary thresholds from calibrated r_threshold
                    cal_merc_entry = cal.r_threshold_mean * 1.8
                    cal_merc_exit = cal.r_threshold_mean * 1.3
                    st.session_state["merc_entry"] = cal_merc_entry * 100  # stored as %
                    st.session_state["merc_exit"] = cal_merc_exit * 100
                    st.success(
                        f"✅ Applied: α+={cal.alpha_plus:.3f}, α-×={cal.alpha_minus_multiplier:.1f}, "
                        f"σ={cal.diffusion_sigma:.4f}, lag={cal.response_lag_days:.0f}d, "
                        f"merc entry={cal_merc_entry:.2%}, exit={cal_merc_exit:.2%}"
                    )
                    st.info("💡 Values updated in sidebar. Re-run optimization to use them.")
            else:
                st.warning("⚠️ Insufficient data quality — cannot apply calibrated values.")

            # Historical TVL + APY chart
            try:
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
                days_ago = np.arange(len(hist.points))

                ax1.plot(days_ago, hist.tvl_array / 1e6, color="steelblue", linewidth=1.5)
                ax1.set_ylabel("TVL ($M)")
                ax1.set_title(f"90-Day History: {sv_venue['name']}")
                ax1.axhline(
                    sv_target_tvl / 1e6, color="red", linestyle="--", alpha=0.5, label="Target TVL"
                )
                ax1.legend()
                ax1.grid(alpha=0.3)

                ax2.plot(
                    days_ago, hist.apy_array * 100, color="green", linewidth=1.5, label="Total APY"
                )
                if hist.apy_base_array.sum() > 0:
                    ax2.plot(
                        days_ago,
                        hist.apy_base_array * 100,
                        color="gray",
                        linewidth=1,
                        label="Base APY",
                        alpha=0.6,
                    )
                ax2.axhline(
                    sv_r_threshold * 100,
                    color="orange",
                    linestyle="--",
                    alpha=0.5,
                    label="r_threshold",
                )
                ax2.set_ylabel("APY (%)")
                ax2.set_xlabel("Days Ago")
                ax2.legend()
                ax2.grid(alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except ImportError:
                st.info("Install matplotlib for historical charts.")

    # ── Dune Data Sync (single venue) ──
    st.markdown("---")
    sv_dune_col1, sv_dune_col2 = st.columns([1, 3])
    with sv_dune_col1:
        if st.button("🔄 Sync Dune Data", key="sv_dune_sync"):
            with st.spinner(f"Syncing Dune data for {sv_venue['name']}..."):
                try:
                    import sys

                    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                    from campaign.venue_registry import get_venue
                    from dune.sync import sync_venue as dune_sync_venue

                    venue_rec = get_venue(sv_venue["pool_id"])
                    result = dune_sync_venue(venue_rec, days=90)
                    if result.skipped:
                        st.warning(f"⚠️ Skipped: {result.skip_reason}")
                    else:
                        st.success(
                            f"✅ Synced: {result.whale_flows_count} whale flows, "
                            f"{result.mercenary_count} mercenary addresses"
                        )
                except Exception as e:
                    st.error(f"❌ Dune sync failed: {e}")
    with sv_dune_col2:
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from dune.sync import get_sync_status

            status = get_sync_status()
            pool_status = status.get(sv_venue.get("pool_id", ""))
            if pool_status and pool_status["has_whale_flows"]:
                st.caption(
                    f"📊 Cached: {pool_status['whale_count']} whale flows, "
                    f"{pool_status['merc_count']} mercenary addresses "
                    f"(last sync: {pool_status['last_modified']})"
                )
            else:
                st.caption("No cached Dune data for this venue.")
        except Exception:
            st.caption("Dune sync available — click to fetch whale flow data.")

    # ── Run button ──
    sv_run = st.button(
        "🚀 Run Single-Venue Optimization",
        type="primary",
        use_container_width=True,
        key="sv_run_btn",
    )

    if sv_run:
        # Build the venue dict with overridden current_tvl
        sv_venue_run = {**sv_venue, "current_tvl": sv_current_tvl}

        # Fetch whales (with optional Dune history for empirical thresholds)
        with st.spinner(f"Fetching whales for {sv_venue['name']}..."):
            sv_whale_hist = None
            try:
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from dune.sync import build_whale_history_lookup

                sv_whale_hist = build_whale_history_lookup(sv_venue.get("pool_id", ""))
                if sv_whale_hist:
                    st.info(f"📊 Using Dune whale history ({len(sv_whale_hist)} addresses)")
            except Exception:
                pass
            try:
                sv_whales = _fetch_whales_for_venue(
                    sv_venue_run, sv_r_threshold, whale_history=sv_whale_hist
                )
            except Exception as e:
                st.warning(f"⚠️ Whale fetch failed: {e}")
                sv_whales = []

        # MC path guardrails
        min_paths = max(30, 10 * max(len(sv_whales), 1))
        sv_paths_eff = max(sv_paths, min_paths)
        if sv_paths_eff > sv_paths:
            st.info(
                f"MC paths raised {sv_paths} → {sv_paths_eff} "
                f"(whale guardrail: {len(sv_whales)} whales)"
            )

        sv_weights = LossWeights(
            # In Set APR mode: boost w_spend ×5 so optimizer actively minimizes
            # budget, and boost w_apr_floor to ≥15 with high sensitivity so the
            # APR target acts as a near-hard constraint. The optimizer then finds
            # the cheapest (B, r_max) that still maintains the floor.
            w_spend=w_spend if not _is_set_apr_mode else w_spend * 5.0,
            w_spend_waste_penalty=w_spend_waste,
            w_apr_variance=w_apr_var if not _is_set_apr_mode else w_apr_var * 1.5,
            w_apr_ceiling=w_apr_ceil if not _is_set_apr_mode else w_apr_ceil * 2.0,
            w_tvl_shortfall=w_tvl_short,
            w_budget_waste=w_budget_waste,
            w_mercenary=w_mercenary,
            w_whale_proximity=w_whale_proximity,
            w_apr_floor=w_apr_floor if not _is_set_apr_mode else max(w_apr_floor, 15.0),
            apr_target=sv_r_threshold * apr_target_mult,
            apr_ceiling=sv_apr_ceiling_total if _is_set_apr_mode else apr_ceiling_val,
            tvl_target=sv_target_tvl,
            apr_stability_on_total=True,
            apr_floor=sv_floor_apr,
            apr_floor_sensitivity=sv_apr_sensitivity
            if not _is_set_apr_mode
            else max(sv_apr_sensitivity, 0.85),
            # In Set APR mode, normalize spend by the budget CEILING so
            # absolute spend is compared across grid points. Without this,
            # spend_cost ≈ utilization (same for all configs when cap binds)
            # and the optimizer can't distinguish cheap from expensive configs.
            spend_reference_budget=sv_budget * 4 if _is_set_apr_mode else 0.0,
        )
        sv_retail = RetailDepositorConfig(
            alpha_plus=alpha_plus,
            alpha_minus_multiplier=alpha_minus_mult,
            response_lag_days=response_lag,
            diffusion_sigma=diffusion_sigma,
        )
        sv_merc = MercenaryConfig(
            entry_threshold=merc_entry_thresh,
            exit_threshold=merc_exit_thresh,
            max_capital_usd=sv_target_tvl * merc_max_frac,
        )

        # Build APY-sensitive config (only active when floor_apr > 0)
        sv_apy_sensitive = None
        if sv_floor_apr > 0:
            sv_apy_sensitive = APYSensitiveConfig(
                floor_apr=sv_floor_apr,
                sensitivity=sv_apr_sensitivity,
                max_sensitive_tvl=sv_target_tvl * 0.10,
            )

        # ── Convert total APR bounds to incentive r_max range ──
        # User enters total APR; subtract base_apy to get the incentive cap
        # that run_venue_optimization expects.
        _sv_r_lo_inc = max(0.005, sv_r_lo_total - sv_base_apy)
        _sv_r_hi_inc = max(_sv_r_lo_inc + 0.005, sv_r_hi_total - sv_base_apy)

        with st.spinner(f"Running optimization for {sv_venue['name']}..."):
            t0 = time.time()
            sv_sr = run_venue_optimization(
                venue=sv_venue_run,
                base_apy=sv_base_apy,
                target_tvl=sv_target_tvl,
                target_util=sv_target_util,
                r_threshold=sv_r_threshold,
                whale_profiles=sv_whales,
                total_budget=sv_budget,
                pinned_budget=sv_pin_b_val,
                pinned_r_max=sv_pin_r_val,
                forced_rate=sv_forced_rate,
                n_paths=sv_paths_eff,
                weights=sv_weights,
                retail_config=sv_retail,
                mercenary_config=sv_merc,
                apy_sensitive_config=sv_apy_sensitive,
                r_max_range=(_sv_r_lo_inc, _sv_r_hi_inc),
                supply_cap=sv_supply_cap,
            )
            sv_elapsed = time.time() - t0

        # ── Compute forced-rate budget feasibility ──
        sv_forced_rate_info = None
        if sv_forced_rate is not None and sv_forced_rate > 0:
            _WPY = 365.0 / 7.0
            _ref_tvl = max(sv_current_tvl, sv_target_tvl)
            _required_B = _ref_tvl * sv_forced_rate / _WPY
            sv_forced_rate_info = {
                "forced_rate": sv_forced_rate,
                "required_budget": _required_B,
                "input_budget": sv_budget,
                "overspend": _required_B > sv_budget,
                "overspend_amount": max(0, _required_B - sv_budget),
                "reference_tvl": _ref_tvl,
            }

        st.session_state["sv_result"] = {
            "surface": sv_sr,
            "venue": sv_venue_run,
            "base_apy": sv_base_apy,
            "r_threshold": sv_r_threshold,
            "target_tvl": sv_target_tvl,
            "target_util": sv_target_util,
            "budget": sv_budget,
            "time": sv_elapsed,
            "n_whales": len(sv_whales),
            "forced_rate_info": sv_forced_rate_info,
            "floor_apr": sv_floor_apr,
            "apr_sensitivity": sv_apr_sensitivity,
            "mode": "set_apr" if _is_set_apr_mode else "set_budget",
            "derived_budget": _sv_derived_budget if _is_set_apr_mode else None,
            "set_apr_target": sv_set_apr_total if _is_set_apr_mode else None,
        }

    # ── Display results ──
    if "sv_result" not in st.session_state:
        return

    ir = st.session_state["sv_result"]
    sv_sr = ir["surface"]
    sv_v = ir["venue"]
    sv_base_disp = ir["base_apy"]
    sv_target_disp = ir["target_tvl"]

    # ── Apply r_max floor (same logic as multi-venue allocation) ──
    # The optimizer may pick an r_max below the float rate at target TVL,
    # which wastes budget (cap binds, spend < B). Floor r_max at the float
    # rate, respecting per-protocol ceiling (not just global 8%).
    WEEKS_PER_YEAR = 365.0 / 7.0
    sv_B = sv_sr.optimal_B
    sv_r_raw = sv_sr.optimal_r_max
    float_rate = sv_B / max(sv_target_disp, 1.0) * WEEKS_PER_YEAR
    sv_proto = sv_v.get("protocol", "").lower()
    _, sv_proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(sv_proto, (0.02, GLOBAL_R_MAX_CEILING))
    sv_venue_ceiling = min(sv_proto_hi, GLOBAL_R_MAX_CEILING)
    sv_r = max(sv_r_raw, min(float_rate, sv_venue_ceiling))

    if sv_r > sv_r_raw + 1e-6:
        st.warning(
            f"⚠️ **r_max floor applied:** Optimizer picked r_max={sv_r_raw:.2%}, but "
            f"the float rate at target TVL is {float_rate:.2%}. "
            f"Raised r_max to {sv_r:.2%} (protocol ceiling: {sv_venue_ceiling:.0%}) "
            f"so the full budget is deployable."
        )

    sv_tb = t_bind(sv_B, sv_r)

    st.markdown("---")
    _result_mode = ir.get("mode", "set_budget")
    _mode_label = "📊 Set APR" if _result_mode == "set_apr" else "💰 Set Budget"
    st.header(f"📊 Results: {sv_v['name']}  ({_mode_label})")

    # Extract MC result early — needed by both Set APR status block and metrics
    mc = sv_sr.optimal_mc_result

    # In Set APR mode, show the APR constraint status and budget savings
    if _result_mode == "set_apr" and mc:
        _floor = ir.get("floor_apr", 0)
        _derived_B = ir.get("derived_budget")
        _set_apr_target = ir.get("set_apr_target", _floor)

        if _floor > 0:
            _apr_ok = mc.apr_p5 >= _floor * 0.98  # Allow 2% tolerance

            # ── Budget Savings Summary ──
            if _derived_B and _derived_B > 0:
                _savings = _derived_B - sv_B
                _savings_pct = (_savings / _derived_B) * 100
                if _savings > 0:
                    st.success(
                        f"💰 **Budget optimized:** Optimizer found **${sv_B:,.0f}/wk** "
                        f"(saved **${_savings:,.0f}/wk** = **{_savings_pct:.0f}%** vs "
                        f"derived ceiling ${_derived_B:,.0f}/wk) "
                        f"while maintaining {_floor:.2%} total APR target."
                    )
                else:
                    st.info(
                        f"📊 **Budget at ceiling:** Optimizer needs **${sv_B:,.0f}/wk** "
                        f"(≈ derived ${_derived_B:,.0f}/wk) to maintain {_floor:.2%} "
                        f"total APR — no cheaper feasible configuration found."
                    )

            # ── APR Constraint Status ──
            if _apr_ok:
                st.success(
                    f"✅ **APR constraint met:** Target {_floor:.2%} total APR is maintained. "
                    f"Simulation p5={mc.apr_p5:.2%}, mean={mc.mean_apr:.2%}."
                )
            else:
                st.error(
                    f"🚨 **APR constraint at risk:** Target {_floor:.2%} total APR, "
                    f"but simulation p5={mc.apr_p5:.2%} (below target). "
                    f"Consider increasing the target APR or reducing target TVL."
                )

            # ── APR Headroom Sensitivity Indicator ──
            # Shows how much room above the floor the chosen config has.
            # Green = lots of headroom (could lower), Yellow = moderate, Red = tight
            _headroom_abs = mc.apr_p5 - _floor  # worst-case headroom
            _headroom_mean = mc.mean_apr - _floor  # average headroom
            _floor_breach_cost = mc.loss_components.get("floor_breach", 0.0)
            _time_below = mc.mean_time_below_floor

            # Classify headroom
            if _headroom_abs >= _floor * 0.15:  # p5 is ≥15% above floor
                _hr_color = "🟢"
                _hr_label = "Large headroom"
                _hr_advice = "Budget could potentially be lowered further."
            elif _headroom_abs >= _floor * 0.03:  # p5 is 3-15% above floor
                _hr_color = "🟡"
                _hr_label = "Moderate headroom"
                _hr_advice = "Some room to lower, but limited flexibility."
            elif _headroom_abs >= 0:  # p5 is 0-3% above floor
                _hr_color = "🔴"
                _hr_label = "Tight — at the edge"
                _hr_advice = "No room to lower budget. Whale exits or TVL spikes may breach floor."
            else:  # p5 is below floor
                _hr_color = "⚫"
                _hr_label = "BREACHING"
                _hr_advice = "APR drops below target in worst-case scenarios. Increase budget."

            with st.container(border=True):
                st.markdown(f"### {_hr_color} APR Headroom: {_hr_label}")
                hr1, hr2, hr3, hr4 = st.columns(4)
                with hr1:
                    st.metric(
                        "p5 Headroom",
                        f"{_headroom_abs:+.2%}",
                        help="APR p5 minus floor — worst-case headroom across MC paths",
                    )
                with hr2:
                    st.metric(
                        "Mean Headroom",
                        f"{_headroom_mean:+.2%}",
                        help="Mean APR minus floor — average headroom",
                    )
                with hr3:
                    st.metric(
                        "Time Below Floor",
                        f"{_time_below:.1%}",
                        help="Average fraction of simulation time where APR drops below target",
                    )
                with hr4:
                    st.metric(
                        "Floor Breach Cost",
                        f"{_floor_breach_cost:.2e}",
                        help="Loss component from APR floor breaches (lower = better)",
                    )
                st.caption(f"💡 {_hr_advice}")

    # Key metrics
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    with rc1:
        st.metric(
            "B*",
            f"${sv_B:,.0f}/wk",
            help="**Optimal Weekly Budget.** The weekly incentive spend the optimizer recommends. "
            "In Set APR mode, this is the minimum budget that maintains the target APR. "
            "In Set Budget mode, this is the best allocation within your budget ceiling.",
        )
    with rc2:
        st.metric(
            "r_max*",
            f"{sv_r:.2%}",
            help="**Optimal Incentive Rate Cap.** The maximum annualized incentive rate. "
            "When TVL < T_bind, the cap binds and depositors earn exactly r_max. "
            "When TVL > T_bind, the rate floats below the cap (B/TVL × 52.14).",
        )
    with rc3:
        st.metric(
            "T_bind",
            f"${sv_tb / 1e6:.1f}M",
            help="**TVL Breakpoint.** The TVL level where incentive rate transitions from "
            "capped (rate = r_max) to floating (rate = B/TVL × 52.14). "
            "T_bind = B × 52.14 / r_max. Below T_bind: MAX-like. Above T_bind: Float-like.",
        )
    with rc4:
        st.metric(
            "Loss",
            f"{sv_sr.optimal_loss:.3e}",
            help="**Composite Loss Score.** The optimizer's objective function value — "
            "lower is better. Combines spend efficiency, APR stability, TVL shortfall, "
            "whale risk, floor breaches, and other penalty terms. Use to compare configs.",
        )
    with rc5:
        st.metric(
            "Whales",
            f"{ir['n_whales']}",
            help="**Whale Profiles Simulated.** Number of large depositor profiles included "
            "in the Monte Carlo simulation. Each whale can enter/exit based on the "
            "incentive rate vs their opportunity cost (r_threshold).",
        )
    with rc6:
        st.metric(
            "Time",
            f"{ir['time']:.1f}s",
            help="**Optimization Runtime.** Wall-clock time for the grid search + "
            "Monte Carlo simulation. Scales with MC Paths × grid resolution.",
        )

    # MC diagnostics (mc already extracted above)
    if mc:
        rc7, rc8, rc9, rc10 = st.columns(4)
        with rc7:
            st.metric(
                "Mean Total APR",
                f"{mc.mean_apr:.2%}",
                help="**Mean Total APR.** Average total APR (base + incentive) across all "
                "MC paths and timesteps. This is what a typical depositor earns on average "
                "over the campaign duration.",
            )
        with rc8:
            st.metric(
                "Mean Incentive APR",
                f"{mc.mean_incentive_apr:.2%}",
                help="**Mean Incentive APR.** Average incentive-only APR across all MC paths. "
                "This is the incremental yield from your incentive spend, excluding the "
                "venue's base APY.",
            )
        with rc9:
            st.metric(
                "APR Range (p5–p95)",
                f"{mc.apr_p5:.1%} – {mc.apr_p95:.1%}",
                help="**APR Confidence Band.** The 5th–95th percentile of total APR across "
                "MC paths. p5 = worst-case APR (only 5% of scenarios are worse). "
                "p95 = best-case APR. Narrow range = stable rate, wide = volatile.",
            )
        with rc10:
            st.metric(
                "Budget Util",
                f"{mc.mean_budget_util:.1%}",
                help="**Budget Utilization.** Average fraction of the weekly budget actually "
                "spent. <100% means the rate cap binds before the full budget is deployed "
                "(MAX-like regime). 100% = all budget spent every week (Float regime).",
            )

    # Campaign type classification
    if sv_tb < sv_target_disp * 0.5:
        ctype = "Float-like"
        ctype_desc = "Cap rarely binds — rate floats inversely with TVL"
    elif sv_tb > sv_target_disp * 1.2:
        ctype = "MAX-like"
        ctype_desc = "Cap always binds — effectively constant rate"
    else:
        ctype = "Hybrid"
        ctype_desc = "Cap binds at low TVL, floats at high TVL"
    st.info(
        f"**Campaign Type: {ctype}** — {ctype_desc}\n\n"
        f"ℹ️ *Float:* Budget is fully spent every week, rate = B/TVL×52.14 (inversely proportional to TVL). "
        f"*MAX:* Rate cap (r_max) binds, spend < budget when TVL is below T_bind. "
        f"*Hybrid:* Transitions between the two regimes depending on TVL.*"
    )

    # ── Actionable Merkl Campaign Instructions ──
    st.subheader("🎯 Merkl Campaign Instructions")
    st.caption(
        "Copy these exact values into Merkl when setting up this campaign. "
        "Campaign Type is derived from where T_bind sits relative to target TVL."
    )
    merkl_type = "Hybrid" if ctype == "Hybrid" else ("MAX" if ctype == "MAX-like" else "Float")
    inc_at_target = apr_at_tvl(sv_B, sv_target_disp, sv_r)
    total_apr_at_target = sv_base_disp + inc_at_target

    # ── Risk assessment (single venue) ──
    sv_risks = []
    sv_forced_info = ir.get("forced_rate_info")
    sv_ir_floor = ir.get("floor_apr", 0.0)
    sv_ir_sens = ir.get("apr_sensitivity", 0.0)

    # Risk 1: Forced rate overspend
    if sv_forced_info and sv_forced_info.get("overspend"):
        sv_risks.append(
            f"🔴 **Budget overspend required:** Forced rate {sv_forced_info['forced_rate']:.2%} "
            f"needs ${sv_forced_info['required_budget']:,.0f}/wk but input budget is "
            f"${sv_forced_info['input_budget']:,.0f}/wk "
            f"(+${sv_forced_info['overspend_amount']:,.0f}/wk over budget)"
        )

    # Risk 2: Floor APR breach
    if sv_ir_floor > 0 and total_apr_at_target < sv_ir_floor:
        gap = sv_ir_floor - total_apr_at_target
        sv_risks.append(
            f"🟠 **Floor APR at risk:** Total APR at target TVL ({total_apr_at_target:.2%}) "
            f"is below the floor APR ({sv_ir_floor:.2%}) by {gap:.2%}. "
            + (
                f"APR sensitivity is {sv_ir_sens:.0%} — "
                f"{'high risk of rapid TVL unwind.' if sv_ir_sens > 0.5 else 'moderate risk.'} "
                if sv_ir_sens > 0
                else ""
            )
            + "Consider increasing budget or forcing a higher incentive rate."
        )

    # Risk 3: Below competitor threshold
    sv_ir_rthresh = ir.get("r_threshold", 0.045)
    if total_apr_at_target < sv_ir_rthresh:
        sv_risks.append(
            f"🟠 **Below competitor rate:** Total APR at target ({total_apr_at_target:.2%}) "
            f"< r_threshold ({sv_ir_rthresh:.2%}). Venue will be uncompetitive. "
            f"Consider increasing budget or lowering target TVL."
        )

    # Risk 4: Low budget utilization
    if mc and mc.mean_budget_util < 0.5:
        sv_risks.append(
            f"🟡 **Low budget utilization:** Only {mc.mean_budget_util:.0%} of budget expected "
            f"to be spent. TVL is too low for the r_max cap. Consider lowering r_max."
        )

    # Risk 5: High incentive at current TVL
    inc_at_current = apr_at_tvl(sv_B, sv_v["current_tvl"], sv_r)
    if inc_at_current > 0.10:
        sv_risks.append(
            f"🟡 **High incentive at current TVL:** {inc_at_current:.2%} may attract "
            f"mercenary capital. Consider a lower r_max to smooth the transition."
        )

    if sv_risks:
        st.markdown("### ⚠️ Risk Assessment")
        with st.container(border=True):
            for risk in sv_risks:
                st.markdown(risk)
        st.markdown("---")

    with st.container(border=True):
        # Header with risk indicator
        if sv_risks:
            st.markdown(f"**{sv_v['name']}** ({sv_v['asset']}) ⚠️")
        else:
            st.markdown(f"**{sv_v['name']}** ({sv_v['asset']}) ✅")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"🔧 **Campaign Type:** `{merkl_type}`")
        with m2:
            st.markdown(f"💰 **Weekly Budget:** `${sv_B:,.0f}`")
        with m3:
            if merkl_type == "MAX":
                st.markdown(f"📊 **Incentive Rate:** `{sv_r:.2%}`")
            else:
                st.markdown(f"📊 **Max Incentive Rate (r_max):** `{sv_r:.2%}`")
        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.caption(f"Base APY: {sv_base_disp:.2%}")
        with detail_cols[1]:
            st.caption(f"Incentive @ Target: {inc_at_target:.2%}")
        with detail_cols[2]:
            st.caption(f"Total APR @ Target: {total_apr_at_target:.2%}")
        with detail_cols[3]:
            st.caption(f"T_bind: ${sv_tb / 1e6:.0f}M")

        # Inline status — mode-aware
        if _result_mode == "set_apr":
            # Set APR mode: show budget optimization result instead of forced-rate status
            _derived_B_disp = ir.get("derived_budget")
            _set_apr_disp = ir.get("set_apr_target", sv_ir_floor)
            if _derived_B_disp and _derived_B_disp > 0:
                _sav = _derived_B_disp - sv_B
                if _sav > 0:
                    st.success(
                        f"💰 **Budget minimized:** ${sv_B:,.0f}/wk "
                        f"(${_sav:,.0f}/wk saved vs derived ${_derived_B_disp:,.0f}/wk). "
                        f"Total APR target: {_set_apr_disp:.2%}."
                    )
                else:
                    st.info(
                        f"📊 **Set APR result:** ${sv_B:,.0f}/wk needed to maintain "
                        f"{_set_apr_disp:.2%} total APR. No cheaper config found."
                    )
            else:
                st.info(
                    f"📊 **Set APR result:** ${sv_B:,.0f}/wk at r_max={sv_r:.2%} "
                    f"to maintain {sv_ir_floor:.2%} total APR."
                )
        elif sv_forced_info:
            if sv_forced_info.get("overspend"):
                st.error(
                    f"💸 **Budget gap:** Forced rate {sv_forced_info['forced_rate']:.2%} requires "
                    f"${sv_forced_info['required_budget']:,.0f}/wk — "
                    f"${sv_forced_info['overspend_amount']:,.0f}/wk over input budget. "
                    f"Increase budget or accept overspend."
                )
            else:
                st.success(
                    f"✅ Forced rate {sv_forced_info['forced_rate']:.2%} is feasible "
                    f"(requires ${sv_forced_info['required_budget']:,.0f}/wk, "
                    f"within ${sv_forced_info['input_budget']:,.0f}/wk budget)"
                )

        # If floor APR set and at risk, show sub-optimal fallback info
        if sv_ir_floor > 0 and total_apr_at_target < sv_ir_floor:
            st.warning(
                f"💡 **Suggestion:** To maintain {sv_ir_floor:.2%} floor APR at target TVL "
                f"(${sv_target_disp / 1e6:.0f}M), you need a minimum incentive rate of "
                f"{max(0, sv_ir_floor - sv_base_disp):.2%}. "
                f"This implies a forced rate of at least {max(0, sv_ir_floor - sv_base_disp):.2%} "
                f"or a weekly budget of at least "
                f"${sv_target_disp * max(0, sv_ir_floor - sv_base_disp) / WEEKS_PER_YEAR:,.0f}."
            )

        # ── Suggested TVL Cap Advisory ──
        # The TVL cap (supply cap) is a safety net — it prevents uncontrolled
        # TVL growth from diluting the incentive rate below the floor.
        # It should be:
        #   - Loose enough that it rarely binds during normal growth
        #   - Tight enough that if TVL surges beyond what the budget can
        #     sustain, the pool stops accepting deposits before the rate
        #     dilutes into irrelevance or attracts bad capital at low yields
        # We derive suggested thresholds from the campaign budget, floor APR,
        # and MC simulation results.
        if mc and sv_B > 0:
            _WPY_cap = 365.0 / 7.0

            # ── Key TVL levels derived from budget + rates ──
            # TVL at which incentive rate = floor incentive rate (the critical level)
            _floor_inc = max(0, sv_ir_floor - sv_base_disp) if sv_ir_floor > 0 else 0
            _floor_tvl = (sv_B * _WPY_cap / _floor_inc) if _floor_inc > 0 else float("inf")

            # TVL at which incentive rate = r_max (T_bind — below this, cap binds)
            _tbind_tvl = t_bind(sv_B, sv_r)

            # TVL at which total APR = r_threshold (rate is competitive but not attractive)
            _sv_rthresh = ir.get("r_threshold", 0.045)
            _rthresh_inc = max(0, _sv_rthresh - sv_base_disp)
            _rthresh_tvl = (sv_B * _WPY_cap / _rthresh_inc) if _rthresh_inc > 0 else float("inf")

            # ── Suggested Safe TVL Cap ──
            # The safe cap IS the floor TVL — the exact TVL where the incentive
            # rate equals the floor.  No buffer applied: this is the hard ceiling
            # where the rate hits the minimum acceptable level.
            # Also floored at MC p95 TVL so the cap doesn't bind during normal sim paths.
            _mc_p95_tvl = getattr(mc, "tvl_max_p95", sv_target_disp * 1.5)
            if _floor_tvl < float("inf"):
                _suggested_safe_tvl = _floor_tvl
                # Ensure it's at least the MC-observed max TVL (don't bind during normal operation)
                _suggested_safe_tvl = max(
                    _suggested_safe_tvl, _mc_p95_tvl if _mc_p95_tvl > 0 else sv_target_disp
                )
            else:
                # No floor APR set — use 1.5× target as reasonable cap
                _suggested_safe_tvl = max(
                    sv_target_disp * 1.5, _mc_p95_tvl if _mc_p95_tvl > 0 else sv_target_disp * 1.5
                )

            # ── Danger TVL Threshold ──
            # Severity-scaled: how far above the floor TVL before it becomes
            # genuinely dangerous depends on how tight the floor incentive is
            # relative to the competitor rate. Tighter margins = less room.
            if _floor_inc > 0:
                # Severity ratio: how much of the competitor rate is the floor?
                # If floor ≈ r_threshold (tight), danger is close (1.3×).
                # If floor << r_threshold (loose), more room before danger (2×).
                _severity = (
                    min(_floor_inc / max(_sv_rthresh - sv_base_disp, _floor_inc), 1.0)
                    if _sv_rthresh > sv_base_disp
                    else 0.5
                )
                _danger_mult = 2.0 - 0.7 * _severity  # ranges from 1.3× (tight) to 2.0× (loose)
                _danger_tvl = _floor_tvl * _danger_mult
            else:
                _danger_tvl = (
                    _rthresh_tvl * 1.5 if _rthresh_tvl < float("inf") else sv_target_disp * 3.0
                )

            # Incentive rates at these cap levels
            _rate_at_safe = apr_at_tvl(sv_B, _suggested_safe_tvl, sv_r)
            _rate_at_danger = apr_at_tvl(sv_B, _danger_tvl, sv_r)
            _rate_at_current_cap = (
                apr_at_tvl(sv_B, sv_supply_cap, sv_r) if sv_supply_cap > 0 else None
            )

            # Current supply cap assessment
            _current_tvl_cap = sv_supply_cap  # user-set supply cap (0 = unlimited)
            if _current_tvl_cap <= 0:
                _cap_status = "⚫"
                _cap_verdict = "No supply cap is set — TVL is uncapped. Consider setting one to protect against rate dilution."
            elif _current_tvl_cap <= _suggested_safe_tvl * 1.05:
                _cap_status = "🟢"
                _cap_verdict = "Your supply cap is within the safe range — growth is possible without rate dilution."
            elif _current_tvl_cap <= _danger_tvl:
                _cap_status = "🟡"
                _cap_verdict = "Your supply cap is above the safe level. If TVL reaches this cap, the incentive rate may drop below your floor."
            else:
                _cap_status = "🔴"
                _cap_verdict = "Your supply cap is in the danger zone — at this TVL, the budget cannot sustain adequate incentive rates."

            with st.container(border=True):
                st.markdown(f"### 🛡️ Suggested TVL Cap  {_cap_status}")
                cap_c1, cap_c2, cap_c3 = st.columns(3)
                with cap_c1:
                    st.metric(
                        "Suggested Safe Cap",
                        f"${_suggested_safe_tvl / 1e6:.0f}M",
                        help=(
                            f"The TVL where the incentive rate exactly equals your floor "
                            f"({_floor_inc:.2%}). Above this, the rate drops below the "
                            f"minimum you specified.\n\n"
                            f"At this TVL, the incentive rate = {_rate_at_safe:.2%} "
                            f"(total APR: {_rate_at_safe + sv_base_disp:.2%}).\n\n"
                            + (
                                f"Derived: B × 52.14 / floor_inc = ${sv_B:,.0f} × 52.14 / "
                                f"{_floor_inc:.4f} = ${_floor_tvl / 1e6:.0f}M.\n\n"
                                if _floor_inc > 0 and _floor_tvl < float("inf")
                                else "No floor APR set — using 1.5× target TVL.\n\n"
                            )
                            + f"MC-observed p95 TVL: ${_mc_p95_tvl / 1e6:.0f}M."
                        ),
                    )
                with cap_c2:
                    st.metric(
                        "Danger Threshold",
                        f"${_danger_tvl / 1e6:.0f}M",
                        help=(
                            f"Beyond this TVL, the budget cannot sustain meaningful incentives. "
                            f"At ${_danger_tvl / 1e6:.0f}M, the incentive rate drops to "
                            f"{_rate_at_danger:.2%} "
                            f"(total APR: {_rate_at_danger + sv_base_disp:.2%})."
                            + (
                                f"\n\nDanger multiplier: {_danger_mult:.1f}× floor TVL "
                                f"(severity-scaled: floor is "
                                f"{_severity:.0%} of competitor rate → "
                                f"{'tight margins, danger close' if _severity > 0.7 else 'loose margins, more room'})."
                                if _floor_inc > 0
                                else ""
                            )
                        ),
                    )
                with cap_c3:
                    if _current_tvl_cap > 0:
                        st.metric(
                            "Current Supply Cap",
                            f"${_current_tvl_cap / 1e6:.0f}M",
                            delta=f"{(_current_tvl_cap - _suggested_safe_tvl) / 1e6:+.0f}M vs safe",
                            delta_color="inverse",
                            help=(
                                f"Your configured supply cap. "
                                f"At this TVL, the incentive rate would be "
                                f"{_rate_at_current_cap:.2%} "
                                f"(total APR: {_rate_at_current_cap + sv_base_disp:.2%})."
                            ),
                        )
                    else:
                        st.metric(
                            "Current Supply Cap",
                            "Unlimited",
                            help="No supply cap is configured. The pool accepts unlimited deposits.",
                        )

                # Verdict
                st.caption(f"💡 {_cap_verdict}")
                if _floor_inc > 0 and _floor_tvl < float("inf"):
                    st.caption(
                        f"**Derivation:** At your budget of ${sv_B:,.0f}/wk, the incentive "
                        f"rate equals your floor ({_floor_inc:.2%}) at TVL = "
                        f"${_floor_tvl / 1e6:.0f}M — this is your safe cap (no buffer). "
                        f"Danger zone ({_danger_mult:.1f}× floor TVL) starts at ${_danger_tvl / 1e6:.0f}M "
                        f"(rate drops to {_rate_at_danger:.2%})."
                    )
                    st.caption(
                        f"**From your floor APR ({sv_ir_floor:.2%}):** "
                        f"Going above ${_suggested_safe_tvl / 1e6:.0f}M means the incentive "
                        f"rate drops below your floor — deposits are under-compensated. "
                        f"Going above ${_danger_tvl / 1e6:.0f}M is dangerous — "
                        f"the budget can't sustain the rate and you're wasting spend."
                    )

    # APR at key TVL levels
    st.subheader("Incentive APR at Key TVL Levels")
    st.caption(
        "📖 **Level:** TVL scenario name. **TVL ($M):** Dollar value of deposits. "
        "**Incentive APR:** Annualized incentive-only rate at that TVL (B/TVL × 52.14, capped at r_max). "
        "**Total APR:** Base APY + Incentive APR. "
        "**Regime:** 🔒 Cap binds = rate equals r_max (MAX-like), 📈 Float = rate is below cap (Float-like)."
    )
    tvl_pts = {
        "Current TVL": sv_v["current_tvl"],
        "Target TVL": sv_target_disp,
        "T_bind": t_bind(sv_B, sv_r),
        "80% Current": sv_v["current_tvl"] * 0.8,
        "120% Current": sv_v["current_tvl"] * 1.2,
        "50% Target": sv_target_disp * 0.5,
        "150% Target": sv_target_disp * 1.5,
    }
    apr_rows = []
    for label, tvl_val in tvl_pts.items():
        inc = apr_at_tvl(sv_B, tvl_val, sv_r)
        regime = "🔒 Cap binds" if tvl_val < t_bind(sv_B, sv_r) else "📈 Float"
        apr_rows.append(
            {
                "Level": label,
                "TVL ($M)": f"${tvl_val / 1e6:,.1f}M",
                "Incentive APR": f"{inc:.2%}",
                "Total APR": f"{(sv_base_disp + inc):.2%}",
                "Regime": regime,
            }
        )
    st.dataframe(apr_rows, use_container_width=True, hide_index=True)

    # Sensitivity analysis
    sa = sv_sr.sensitivity_analysis()
    st.markdown(f"**Sensitivity:** {sa['interpretation']}")

    # Duality map
    dual = sv_sr.duality_map(0.05)
    if len(dual) > 1:
        st.subheader(f"Near-Optimal Configurations ({len(dual)} within 5%)")
        st.caption(
            "📖 Configs with loss within 5% of optimal. **B ($/wk):** weekly budget. "
            "**r_max:** incentive rate cap. **T_bind ($M):** TVL breakpoint where cap binds. "
            "**Incentive @ Target:** incentive rate at your target TVL. "
            "**Total @ Target:** base + incentive at target TVL. "
            "**vs Optimal:** how much worse than the best config (lower = closer to optimal)."
        )
        drows = []
        for d in dual[:10]:
            inc = apr_at_tvl(d["B"], sv_target_disp, d["r_max"])
            drows.append(
                {
                    "B ($/wk)": f"${d['B']:,.0f}",
                    "r_max": f"{d['r_max']:.2%}",
                    "T_bind ($M)": f"${d['t_bind'] / 1e6:.1f}M",
                    "Incentive @ Target": f"{inc:.2%}",
                    "Total @ Target": f"{(sv_base_disp + inc):.2%}",
                    "vs Optimal": f"+{(d['loss_ratio'] - 1) * 100:.1f}%",
                }
            )
        st.dataframe(drows, use_container_width=True, hide_index=True)

    # Loss surface plot
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Loss surface
        ax = axes[0]
        Bv = sv_sr.grid.B_values
        rv = sv_sr.grid.r_max_values
        L = np.where(sv_sr.feasibility_mask, sv_sr.loss_surface, np.nan)
        oi, oj = sv_sr.optimal_indices
        vals = L[~np.isnan(L)]
        norm = None
        if len(vals) > 0 and vals.max() / max(vals.min(), 1e-10) > 100:
            norm = LogNorm(vmin=max(vals.min(), 1e-10), vmax=vals.max())
        im = ax.pcolormesh(rv * 100, Bv / 1000, L, cmap="viridis_r", norm=norm, shading="nearest")
        fig.colorbar(im, ax=ax, label="Loss", shrink=0.8)
        ax.plot(
            rv[oj] * 100,
            Bv[oi] / 1000,
            "*",
            color="red",
            ms=16,
            mec="white",
            mew=2,
            label="Optimal",
        )
        ax.set_xlabel("r_max — Incentive APR Cap (%)")
        ax.set_ylabel("B — Weekly Budget ($k)")
        ax.set_title(f"Loss Surface — {sv_v['name']}")
        ax.legend()

        # Right: Component breakdown (if available)
        ax2 = axes[1]
        component_names = [
            "spend_cost",
            "tvl_shortfall_cost",
            "apr_variance_cost",
            "apr_ceiling_cost",
        ]
        component_data = {}
        for cname in component_names:
            surf = getattr(sv_sr, f"{cname}_surface", None)
            if surf is not None and surf.size > 0:
                component_data[cname] = float(surf[oi, oj])

        if component_data:
            labels = [k.replace("_cost", "").replace("_", " ").title() for k in component_data]
            values = list(component_data.values())
            total = sum(values)
            if total > 0:
                bars = ax2.barh(labels, [v / total * 100 for v in values], color="steelblue")
                ax2.set_xlabel("% of Total Loss")
                ax2.set_title("Loss Component Breakdown at Optimal")
                for bar, val in zip(bars, values):
                    ax2.text(
                        bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2e}",
                        va="center",
                        fontsize=9,
                    )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "All components zero",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "Component surfaces not available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib for surface plots.")

    # ── Efficiency Metrics ──
    st.subheader("📈 Efficiency Metrics")
    ns = sv_target_disp * (1 - ir["target_util"])
    eff_c1, eff_c2, eff_c3, eff_c4 = st.columns(4)
    with eff_c1:
        st.metric(
            "Net Supply",
            f"${ns / 1e6:.0f}M",
            help="TVL × (1 − utilization). Idle capital available for deployment.",
        )
    with eff_c2:
        tvl_per_inc = sv_target_disp / sv_B if sv_B > 0 else 0
        st.metric(
            "TVL / $incentive",
            f"{tvl_per_inc:,.0f}",
            help="Dollars of TVL attracted per dollar of weekly incentive spend.",
        )
    with eff_c3:
        ns_per_inc = ns / sv_B if sv_B > 0 else 0
        st.metric(
            "Net Supply / $incentive",
            f"{ns_per_inc:,.0f}",
            help="Dollars of net supply per dollar of weekly incentive spend.",
        )
    with eff_c4:
        annual_cost = sv_B * 52
        st.metric(
            "Annual Cost",
            f"${annual_cost / 1e6:.2f}M",
            help="Annualized incentive spend at this weekly budget.",
        )

    # Export
    st.subheader("📤 Export")
    ns = sv_target_disp * (1 - ir["target_util"])
    sv_export = {
        "venue": sv_v["name"],
        "asset": sv_v["asset"],
        "protocol": sv_v["protocol"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC"),
        "mode": ir.get("mode", "set_budget"),
        "inputs": {
            "weekly_budget": ir["budget"],
            "target_tvl": round(sv_target_disp),
            "target_util": round(ir["target_util"], 3),
            "current_tvl": round(sv_v["current_tvl"]),
            "base_apy": round(sv_base_disp, 4),
            "r_threshold": round(ir["r_threshold"], 4),
            "n_whales": ir["n_whales"],
            "mc_paths": sv_sr.grid.B_values.size,
        },
        "results": {
            "optimal_B": round(sv_B),
            "optimal_r_max": round(sv_r, 4),
            "optimal_r_max_raw": round(sv_r_raw, 4),
            "r_max_floor_applied": sv_r > sv_r_raw + 1e-6,
            "optimal_loss": round(float(sv_sr.optimal_loss), 6),
            "t_bind": round(sv_tb),
            "campaign_type": ctype,
            "incentive_at_target_tvl": round(apr_at_tvl(sv_B, sv_target_disp, sv_r), 4),
            "total_apr_at_target": round(sv_base_disp + apr_at_tvl(sv_B, sv_target_disp, sv_r), 4),
            "net_supply": round(ns),
            "tvl_per_incentive": round(sv_target_disp / sv_B) if sv_B > 0 else 0,
            "ns_per_incentive": round(ns / sv_B) if sv_B > 0 else 0,
        },
        "mc_diagnostics": {
            "mean_apr": round(mc.mean_apr, 4) if mc else None,
            "mean_incentive_apr": round(mc.mean_incentive_apr, 4) if mc else None,
            "mean_tvl": round(mc.mean_tvl) if mc else None,
            "budget_util": round(mc.mean_budget_util, 4) if mc else None,
            "apr_p5": round(mc.apr_p5, 4) if mc else None,
            "apr_p95": round(mc.apr_p95, 4) if mc else None,
            "time_below_floor": round(mc.mean_time_below_floor, 4) if mc else None,
            "floor_breach_cost": round(mc.loss_components.get("floor_breach", 0.0), 6)
            if mc
            else None,
        },
        "set_apr_info": {
            "derived_budget": round(ir.get("derived_budget") or 0),
            "set_apr_target": round(ir.get("set_apr_target") or 0, 4),
            "budget_savings": round((ir.get("derived_budget") or sv_B) - sv_B),
        }
        if ir.get("mode") == "set_apr"
        else None,
        "simulation_time_s": round(ir["time"], 1),
    }
    st.code(json.dumps(sv_export, indent=2), language="json")


if __name__ == "__main__":
    main()

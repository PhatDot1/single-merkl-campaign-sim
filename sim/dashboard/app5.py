"""
Campaign Optimizer Dashboard — Single-Venue Only

Production configuration:
- All RPC URLs loaded from .env (ALCHEMY_ETH_RPC_URL, HELIUS_SOLANA_RPC_URL)
- Kamino vault pubkeys from .env (KAMINO_PYUSD_EARN_VAULT_PUBKEY, etc.)
- Aave Horizon as separate pool (0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8)
- Whale fetching auto-triggered at optimization time via Alchemy / Helius
- Zero fallbacks to static/empty whale data — errors loudly on failure

Usage:
    streamlit run dashboard/app4.py
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
# when Streamlit is invoked as `streamlit run dashboard/app5.py` from sim/
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
from campaign.base_apy import fetch_all_base_apys  # noqa: E402
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

st.set_page_config(page_title="Single-Venue Optimizer", page_icon="🔬", layout="wide")


# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================


def _validate_env():
    """Check that required env vars are set. Warn in main area if not."""
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
        st.error(
            "**Missing required env vars:**\n\n"
            + "\n\n".join(missing_required)
            + "\n\nSet these in your `.env` file."
        )

    if missing_optional:
        st.warning(
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

    - Aave (Core/Horizon): Alchemy indexed transfer API -> aToken holders
    - Euler: Alchemy indexed transfer API -> eToken holders
    - Morpho: GraphQL V2 positions API
    - Kamino (Earn/CLMM): Helius getTokenLargestAccounts -> share holders
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

            print(f"  -> Calling fetch_aave_reserve_data(asset={asset}, market={market})")
            data = fetch_aave_reserve_data(asset_address, asset, pool_address, market)
            print(
                f"  OK Fetched: TVL=${data.total_supply_usd / 1e6:.2f}M, Util={data.utilization:.1%}"
            )
            return data.total_supply_usd, data.utilization

        elif protocol == "euler":
            from campaign.evm_data import EULER_VAULTS, fetch_euler_vault_data

            vault_address = EULER_VAULTS.get(asset)
            if not vault_address:
                raise RuntimeError(f"Unknown Euler vault address for {asset}")
            print(f"  -> Calling fetch_euler_vault_data(asset={asset})")
            data = fetch_euler_vault_data(vault_address=vault_address, asset_symbol=asset)
            print(
                f"  OK Fetched: TVL=${data.total_supply_usd / 1e6:.2f}M, Util={data.utilization:.1%}"
            )
            return data.total_supply_usd, data.utilization

        elif protocol == "morpho":
            from campaign.data import fetch_morpho_vault_snapshot

            vault_address = venue.get("vault_address")
            if not vault_address:
                raise RuntimeError(f"Morpho venue {venue['name']} has no vault_address")
            print(f"  -> Calling fetch_morpho_vault_snapshot(vault={vault_address[:10]}...)")
            snapshot = fetch_morpho_vault_snapshot(vault_address, chain_id=1)
            print(
                f"  OK Fetched: TVL=${snapshot.total_supply_usd / 1e6:.2f}M, Util={snapshot.utilization:.1%}"
            )
            return snapshot.total_supply_usd, snapshot.utilization

        elif protocol == "kamino" and venue.get("kamino_vault_pubkey"):
            from campaign.kamino_data import fetch_kamino_vault_metrics

            pubkey = venue["kamino_vault_pubkey"]
            print(f"  -> Calling fetch_kamino_vault_metrics(pubkey={pubkey[:10]}...)")
            metrics = fetch_kamino_vault_metrics(pubkey)
            print(f"  OK Fetched: TVL=${metrics.total_tvl_usd / 1e6:.2f}M, Util=N/A (vault)")
            return metrics.total_tvl_usd, 0.0

        elif protocol == "kamino" and venue.get("kamino_reserve_pubkey"):
            from campaign.kamino_data import fetch_kamino_lend_snapshot

            reserve_pubkey = venue["kamino_reserve_pubkey"]
            market = venue.get("kamino_market_name", "main")
            print(
                f"  -> Calling fetch_kamino_lend_snapshot(reserve={reserve_pubkey[:10]}..., market={market})"
            )
            snapshot = fetch_kamino_lend_snapshot(reserve_pubkey, market)
            print(
                f"  OK Fetched: TVL=${snapshot['total_supply_usd'] / 1e6:.2f}M, Util={snapshot['utilization']:.1%}"
            )
            return snapshot["total_supply_usd"], snapshot["utilization"]

        elif protocol == "kamino" and venue.get("kamino_market_name"):
            from campaign.kamino_data import fetch_kamino_reserve_for_asset

            market_name = venue["kamino_market_name"]
            print(
                f"  -> Calling fetch_kamino_reserve_for_asset(asset={asset}, market={market_name})"
            )
            reserve = fetch_kamino_reserve_for_asset(asset_symbol=asset, market_name=market_name)
            if not reserve:
                raise RuntimeError(f"No {asset} reserve found in Kamino {market_name} market")
            print(
                f"  OK Fetched: TVL=${reserve.total_supply_usd / 1e6:.2f}M, Util={reserve.utilization:.1%}"
            )
            return reserve.total_supply_usd, reserve.utilization

        elif protocol == "kamino" and venue.get("kamino_strategy_pubkey"):
            from campaign.kamino_data import fetch_kamino_strategy_metrics

            pubkey = venue["kamino_strategy_pubkey"]
            print(f"  -> Calling fetch_kamino_strategy_metrics(pubkey={pubkey[:10]}...)")
            metrics = fetch_kamino_strategy_metrics(pubkey)
            print(f"  OK Fetched: TVL=${metrics.total_value_locked / 1e6:.2f}M, Util=N/A (CLMM)")
            return metrics.total_value_locked, 0.0

        elif protocol == "curve":
            import requests

            venue_name = venue.get("name", "")

            print(f"  -> Querying DeFiLlama yields API for Curve pool '{venue_name}'...")

            resp = requests.get("https://yields.llama.fi/pools", timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"DeFiLlama API failed: HTTP {resp.status_code}")

            pools = resp.json().get("data", [])

            curve_pools = [
                p
                for p in pools
                if p.get("project") == "curve-dex"
                and p.get("chain") == "Ethereum"
                and asset.lower() in (p.get("symbol") or "").lower()
            ]

            if not curve_pools:
                raise RuntimeError(f"No Curve pool found on DeFiLlama for asset={asset}")

            best_match = None
            for p in curve_pools:
                dl_symbol = (p.get("symbol") or "").upper()
                dl_tokens = set(dl_symbol.split("-"))
                name_parts = venue_name.replace("Curve ", "").upper().split("-")
                name_tokens = set(name_parts)
                if dl_tokens == name_tokens:
                    best_match = p
                    break

            if not best_match:
                curve_pools.sort(key=lambda p: p.get("tvlUsd", 0), reverse=True)
                best_match = curve_pools[0]
                print(
                    f"  Warning: No exact pair match, using largest pool: {best_match.get('symbol')}"
                )

            tvl_usd = best_match.get("tvlUsd", 0)
            print(
                f"  OK Fetched from DeFiLlama: {best_match.get('symbol')} TVL=${tvl_usd / 1e6:.2f}M"
            )

            return tvl_usd, 0.0

        elif protocol == "kamino":
            print("  Warning: Kamino venue missing identifying info")
            raise RuntimeError(
                f"Kamino venue {venue['name']} needs vault_pubkey, strategy_pubkey, or market_name"
            )

        else:
            raise RuntimeError(f"Unknown protocol '{protocol}' for venue {venue['name']}")

    except Exception as e:
        print(f"  FAILED to fetch current TVL/util for {venue['name']}: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError(f"Live data fetch failed for {venue['name']}: {e}") from e


# ============================================================================
# VENUE DEFINITIONS
# ============================================================================

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
                "pool_address": "0x8e0D210a6B95E7a4CF3e7a94d17B7e992fA1d57f",
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
                "aave_market": "horizon",
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
                "kamino_market_name": "maple",
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
                "kamino_vault_pubkey": _PYUSD_EARN_VAULT,
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
                "kamino_strategy_pubkey": _PYUSD_CLMM_VAULT,
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
                "pool_address": "0x383E6b4437b59fff47B619CBA855CA29342A8559",
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

    # Mode B: Manual override
    if cfg.mode == "manual":
        return {
            "r_threshold": cfg.manual_value,
            "r_threshold_lo": cfg.manual_value * 0.8,
            "r_threshold_hi": cfg.manual_value * 1.2,
            "competitors": [],
            "source": f"manual ({cfg.manual_value:.2%})",
            "mode": "manual",
        }

    # Mode A: Auto (direct asset competitors)
    if cfg.mode == "auto":
        try:
            competitors = fetch_competitor_rates(
                asset_symbol=asset_symbol,
                min_tvl=1_000_000,
                exclude_vault_address=exclude_vault,
            )
            thresholds = compute_r_threshold(competitors)
            thresholds["competitors"] = competitors
            rt_source = thresholds.get("r_threshold_source", "asset_peers")
            thresholds["source"] = f"{asset_symbol} -- {rt_source}"
            thresholds["mode"] = "auto"
            return thresholds
        except RuntimeError:
            pass  # Fall through to blended

    # Mode C: Blended (stablecoin class benchmark)
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
        print(f"  Warning: Stablecoin benchmark failed: {e}")

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
    target_inc_apr: float = 0.0,
) -> SurfaceResult:
    """Run full MC surface optimization for one venue."""
    WEEKS_PER_YEAR = 365.0 / 7.0  # 52.14

    # Forced rate mode
    if forced_rate is not None and forced_rate > 0:
        reference_tvl = max(venue["current_tvl"], target_tvl)
        forced_B = reference_tvl * forced_rate / WEEKS_PER_YEAR
        pinned_r_max = forced_rate
        pinned_budget = forced_B
        if forced_B > total_budget:
            print(
                f"  [{venue['name']}] FORCED RATE {forced_rate:.2%} requires "
                f"${forced_B:,.0f}/wk but budget is ${total_budget:,.0f}/wk -- OVERSPEND"
            )
        else:
            print(
                f"  [{venue['name']}] Forced rate {forced_rate:.2%} -> "
                f"B=${forced_B:,.0f}/wk (within budget)"
            )

    r_lo_eff, r_hi_eff = r_max_range

    proto = venue.get("protocol", "").lower()
    if proto in PROTOCOL_R_MAX_DEFAULTS:
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS[proto]
        r_lo_eff = max(r_lo_eff, proto_lo)
        r_hi_eff = min(r_hi_eff, proto_hi)

    r_hi_eff = min(r_hi_eff, GLOBAL_R_MAX_CEILING)
    r_hi_eff = max(r_hi_eff, r_lo_eff + 0.005)

    if pinned_budget is not None:
        b_min, b_max, b_steps = pinned_budget * 0.95, pinned_budget * 1.05, 3
    else:
        b_min = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        b_max = min(total_budget, target_tvl * r_hi_eff / WEEKS_PER_YEAR * 1.5)
        b_max = max(b_max, b_min * 2)
        b_steps = GRID_B_STEPS

        # Floor-aware budget floor
        if apy_sensitive_config and apy_sensitive_config.floor_apr > 0:
            min_inc_for_floor = max(0, apy_sensitive_config.floor_apr - base_apy)
            floor_budget = target_tvl * min_inc_for_floor / WEEKS_PER_YEAR
            if floor_budget <= total_budget:
                b_min = max(b_min, floor_budget * 0.85)
                print(
                    f"  [{venue['name']}] Floor-aware grid: "
                    f"floor={apy_sensitive_config.floor_apr:.2%} needs "
                    f"${floor_budget:,.0f}/wk -> b_min raised to ${b_min:,.0f}"
                )
            else:
                print(
                    f"  [{venue['name']}] Floor {apy_sensitive_config.floor_apr:.2%} "
                    f"needs ${floor_budget:,.0f}/wk but budget is only "
                    f"${total_budget:,.0f}/wk -- floor is unachievable"
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
        target_inc_apr=target_inc_apr,
        target_tvl_for_feasibility=target_tvl if target_inc_apr > 0 else 0.0,
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
    st.title("🔬 Single-Venue Campaign Optimizer")
    st.caption(
        "Full MC simulation engine. Optimizer searches the (B, r_max) hybrid surface. "
        "Base APY fetched on-chain. r_threshold fetched from DeFiLlama. "
        "Whale profiles fetched live via Alchemy/Helius. "
        "Incentive rate is an **output**, not an input."
    )

    _validate_env()

    programs = PROGRAMS

    # ================================================================
    # INLINE SIMULATION SETTINGS (replaces sidebar)
    # ================================================================
    with st.expander("Settings: Simulation & Model Parameters", expanded=False):
        (
            settings_tab_sim,
            settings_tab_weights,
            settings_tab_retail,
            settings_tab_merc,
            settings_tab_grid,
        ) = st.tabs(
            [
                "Simulation",
                "Loss Weights",
                "Retail Depositors",
                "Mercenary Capital",
                "Grid Range",
            ]
        )

        with settings_tab_sim:
            st.caption(
                "Controls the Monte Carlo simulation that underpins every optimization. "
                "More paths give more statistically robust results but take longer to compute."
            )
            n_paths = st.slider(
                "MC Paths",
                10,
                200,
                MC_PATHS_DEFAULT,
                10,
                help=(
                    "**What it is:** Number of independent random simulation paths. Each path "
                    "generates a unique sequence of TVL fluctuations, whale entries/exits, "
                    "and mercenary capital events.\n\n"
                    "**Effect of changing:**\n"
                    "- **Higher (100+):** p5/p50/p95 percentiles converge, results are "
                    "reproducible across runs. Use for final decisions.\n"
                    "- **Lower (30–50):** Faster iteration for exploring parameters, but "
                    "results may shift between runs due to sampling noise.\n"
                    "- **Below 30:** Results are unreliable — don't use for production."
                ),
                key="n_paths",
            )

        with settings_tab_weights:
            st.caption(
                "These weights control the optimizer's loss function — the objective it minimizes. "
                "Higher weight = that factor matters more. The optimizer finds the (Budget, r_max) pair "
                "that minimizes the weighted sum of all these penalties across all MC paths."
            )
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                w_spend = st.number_input(
                    "w_spend",
                    value=1.0,
                    step=0.1,
                    min_value=0.0,
                    key="w_spend",
                    help=(
                        "**What it is:** Penalty on total dollar spend. The lowest priority "
                        "weight — only breaks ties between campaigns that equally hit TVL target.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer favors cheaper configs even at the cost of slightly "
                        "worse TVL or APR stability. Budget-conservative.\n"
                        "- **Lower/0:** Optimizer ignores spend entirely — picks the config with "
                        "best TVL/APR outcomes regardless of cost. Use when budget is unconstrained."
                    ),
                )
                w_spend_waste = st.number_input(
                    "w_spend_waste_penalty",
                    value=2.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_spend_waste",
                    help=(
                        "**What it is:** Extra spend penalty applied when TVL is below target — "
                        "penalizes paying incentives without achieving the TVL goal.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer strongly avoids configs that spend budget while "
                        "TVL remains below target. Favors lower-spend configs when growth is unlikely.\n"
                        "- **Lower/0:** Optimizer tolerates wasteful spend — appropriate if you "
                        "believe the spend will attract TVL over time even if initially below target."
                    ),
                )
                w_apr_var = st.number_input(
                    "w_apr_variance",
                    value=3.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_apr_var",
                    help=(
                        "**What it is:** Penalizes APR volatility across simulation timesteps. "
                        "Normalized as deviation² / target².\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer picks configs with very stable, predictable rates — "
                        "important for vaults with depositors who monitor APR daily.\n"
                        "- **Lower/0:** Allows APR to fluctuate more — appropriate for vaults with "
                        "sticky capital that doesn't react to short-term rate changes.\n\n"
                        "**In Enforce APR mode:** Automatically boosted ×1.5 to keep rates steady around target."
                    ),
                )
            with wc2:
                w_apr_ceil = st.number_input(
                    "w_apr_ceiling",
                    value=5.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_apr_ceil",
                    help=(
                        "**What it is:** Penalty when total APR exceeds the APR Hard Ceiling. "
                        "Prevents attracting unsustainable mercenary capital.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Hard cap on APR — optimizer will never pick configs that "
                        "produce APR spikes above the ceiling. Essential for stablecoin pools.\n"
                        "- **Lower/0:** Allows APR spikes (e.g., during low-TVL periods). May attract "
                        "mercenary capital that enters at high APR then leaves when it normalizes.\n\n"
                        "**In Enforce APR mode:** Automatically boosted ×2 to strictly enforce the ceiling."
                    ),
                )
                w_tvl_short = st.number_input(
                    "w_tvl_shortfall",
                    value=8.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_tvl_short",
                    help=(
                        "**What it is:** Primary objective — penalizes configurations where simulated "
                        "TVL remains below your target. Normalized as shortfall² / target².\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (8+):** TVL growth is THE priority. Optimizer will spend more "
                        "aggressively and tolerate APR volatility to hit TVL target.\n"
                        "- **Lower:** TVL becomes secondary to spend efficiency or rate stability. "
                        "Use when TVL is already near target and you're in maintenance mode."
                    ),
                )
                w_mercenary = st.number_input(
                    "w_mercenary",
                    value=6.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_mercenary",
                    help=(
                        "**What it is:** Penalizes the fraction of TVL that comes from mercenary "
                        "(hot money) capital — depositors who chase yield spikes and leave immediately.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer avoids configs with high APR spikes that attract "
                        "mercs. Favors stable, moderate rates that attract sticky capital.\n"
                        "- **Lower/0:** Tolerates merc capital — appropriate if you want to "
                        "temporarily inflate TVL numbers regardless of capital quality."
                    ),
                )
            with wc3:
                w_whale_proximity = st.number_input(
                    "w_whale_proximity",
                    value=6.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_whale_proximity",
                    help=(
                        "**What it is:** Penalizes configs where the incentive rate is close to "
                        "whale exit thresholds — creating a safety margin against large withdrawals.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer builds more headroom above whale exit points. Reduces "
                        "the risk of a cascade (whale exits → APR drops → more exits).\n"
                        "- **Lower/0:** Optimizer ignores whale exit risk. Acceptable for vaults with "
                        "no large depositors or where whale data isn't available."
                    ),
                )
                w_apr_floor = st.number_input(
                    "w_apr_floor",
                    value=7.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_apr_floor",
                    help=(
                        "**What it is:** Penalizes APR dropping below the Floor APR set per venue. "
                        "Only active when Floor APR > 0.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (7+):** Floor acts as a near-hard constraint. Optimizer won't pick "
                        "configs that breach it even briefly. Essential for APY-sensitive vaults.\n"
                        "- **Lower:** Floor is advisory — optimizer may accept occasional dips below "
                        "if other objectives (spend, TVL) improve significantly.\n\n"
                        "**In Enforce APR mode:** Automatically raised to ≥15 to make the target APR "
                        "a hard constraint."
                    ),
                )
                w_budget_waste = st.number_input(
                    "w_budget_waste",
                    value=0.0,
                    step=0.5,
                    min_value=0.0,
                    key="w_budget_waste",
                    help=(
                        "**What it is:** Penalizes allocating budget that goes unspent because the "
                        "rate cap (r_max) binds before the full budget is deployed.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher:** Optimizer avoids MAX-like configs where cap binds and budget "
                        "is refunded. Pushes toward Float-like configs where all spend is deployed.\n"
                        "- **0 (default):** Merkl refunds unspent budget at 0% fee, so waste has "
                        "no real cost. Keep at 0 unless your platform charges for unused budget."
                    ),
                )
            wc4, wc5 = st.columns(2)
            with wc4:
                apr_target_mult = st.number_input(
                    "APR Target Multiplier (× r_threshold)",
                    value=1.2,
                    step=0.1,
                    min_value=0.5,
                    max_value=3.0,
                    key="apr_target_mult",
                    help=(
                        "**What it is:** The optimizer aims for total APR = r_threshold × this "
                        "multiplier. 1.2 = aim for 20% above competitor rates.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (1.5+):** Aggressively outbid competitors — attracts TVL faster "
                        "but costs more. Good for bootstrapping new vaults.\n"
                        "- **1.0–1.2:** Match or slightly beat competitors — sustainable growth.\n"
                        "- **<1.0:** Underbid competitors — relies on other vault advantages (brand, "
                        "safety) rather than rate. Maintenance/winding-down mode."
                    ),
                )
            with wc5:
                apr_ceiling_val = (
                    st.number_input(
                        "APR Hard Ceiling (%)",
                        value=10.0,
                        step=1.0,
                        min_value=1.0,
                        max_value=50.0,
                        key="apr_ceiling_val",
                        help=(
                            "**What it is:** Maximum total APR allowed globally. APR above this "
                            "triggers heavy penalty from w_apr_ceiling.\n\n"
                            "**Effect of changing:**\n"
                            "- **Lower (5–8%):** Conservative — prevents any APR spike. Good for "
                            "mature stablecoin pools where >8% signals risk.\n"
                            "- **Higher (10–15%):** Allows temporary high APR at low TVL levels. "
                            "Acceptable for bootstrapping or volatile pools.\n"
                            "- **Very high (20%+):** Effectively disables the ceiling — use with caution."
                        ),
                    )
                    / 100.0
                )

        with settings_tab_retail:
            st.caption(
                "Controls how retail (non-whale) depositors respond to APR changes in the simulation. "
                "These parameters define TVL growth/shrink dynamics, reaction speed, and random noise. "
                "Can be auto-calibrated from historical data using the 'Fetch 90-Day History' button below."
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                alpha_plus = st.number_input(
                    "alpha_plus (inflow elasticity)",
                    value=0.15,
                    step=0.05,
                    min_value=0.0,
                    key="alpha_plus",
                    help=(
                        "**What it is:** Speed at which TVL grows when your APR exceeds "
                        "the competitor rate (r_threshold). Controls retail depositor inflows.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (0.2+):** Depositors react quickly to rate advantage — TVL grows "
                        "fast when you outbid competitors. Simulation shows rapid growth to target.\n"
                        "- **Lower (0.05–0.10):** Slow, sticky capital — depositors take weeks to "
                        "discover and move into your vault. More realistic for institutional capital.\n"
                        "- **0:** No TVL response to APR — TVL stays constant (only whale events change it)."
                    ),
                )
                alpha_minus_mult = st.number_input(
                    "alpha_minus_multiplier",
                    value=3.0,
                    step=0.5,
                    min_value=1.0,
                    key="alpha_minus_mult",
                    help=(
                        "**What it is:** Asymmetry factor — outflows are this many times faster "
                        "than inflows. Reflects 'sticky on the way in, fast on the way out'.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (4+):** Depositors flee fast when APR drops. Simulation shows "
                        "sharp TVL drops on rate decreases — more pessimistic scenarios.\n"
                        "- **Lower (1–2):** Symmetric behavior — TVL exits as slowly as it enters. "
                        "More optimistic. Appropriate for locked/vesting positions.\n"
                        "- **1.0:** Inflows = outflows speed. Unrealistic for most DeFi pools."
                    ),
                )
            with rc2:
                response_lag = st.number_input(
                    "response_lag_days",
                    value=5.0,
                    step=1.0,
                    min_value=0.0,
                    key="response_lag",
                    help=(
                        "**What it is:** Days before retail depositors react to APR changes. "
                        "Models real-world delay — depositors don't instantly move capital.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (7–14 days):** Sluggish market — TVL doesn't react to short APR "
                        "spikes/dips. Reduces APR volatility impact. More conservative.\n"
                        "- **Lower (1–3 days):** Fast-reacting depositors — TVL swings quickly "
                        "with APR. More volatile simulations, harder to maintain stable TVL.\n"
                        "- **0:** Instant reaction — unrealistic but useful for stress-testing."
                    ),
                )
                diffusion_sigma = st.number_input(
                    "diffusion_sigma (TVL noise)",
                    value=0.008,
                    step=0.002,
                    min_value=0.0,
                    key="diffusion_sigma",
                    format="%.4f",
                    help=(
                        "**What it is:** Random daily TVL volatility (standard deviation). "
                        "Models organic deposits/withdrawals unrelated to APR.\n\n"
                        "**Effect of changing:**\n"
                        "- **Higher (0.015+):** Very noisy TVL — large random swings each day. "
                        "Widens the APR confidence band (p5–p95). Stress-tests robustness.\n"
                        "- **Lower (0.003–0.005):** Smooth, predictable TVL. Narrow APR band. "
                        "Appropriate for stable institutional pools.\n"
                        "- **0:** No noise — TVL only changes from retail response + whale events. "
                        "Deterministic baseline."
                    ),
                )

        with settings_tab_merc:
            st.caption(
                "Mercenary capital is hot money that enters vaults when APR spikes and exits when it "
                "normalizes. These thresholds control when mercs appear and disappear in the simulation, "
                "and how much capital they can bring."
            )
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                merc_entry_thresh = (
                    st.number_input(
                        "Entry threshold (%)",
                        value=8.0,
                        step=0.5,
                        min_value=0.0,
                        key="merc_entry",
                        help=(
                            "**What it is:** Total APR above which mercenary (hot money) capital "
                            "starts entering the vault. These depositors chase yield spikes.\n\n"
                            "**Effect of changing:**\n"
                            "- **Higher (10%+):** Mercs only appear at extreme APR — less likely to "
                            "be triggered. Simulation has cleaner TVL dynamics.\n"
                            "- **Lower (5–6%):** Mercs enter at moderate APR — more realistic for "
                            "competitive stablecoin pools where yield-farmers are active.\n"
                            "- **Gap between entry & exit:** Creates hysteresis — mercs enter at 8%, "
                            "stay until 6%. Wider gap = more persistent merc capital."
                        ),
                    )
                    / 100.0
                )
            with mc2:
                merc_exit_thresh = (
                    st.number_input(
                        "Exit threshold (%)",
                        value=6.0,
                        step=0.5,
                        min_value=0.0,
                        key="merc_exit",
                        help=(
                            "**What it is:** Total APR below which mercenary capital exits. The gap "
                            "between entry and exit creates a hysteresis band.\n\n"
                            "**Effect of changing:**\n"
                            "- **Higher (close to entry):** Mercs exit quickly when APR dips — "
                            "volatile TVL. More pessimistic simulation.\n"
                            "- **Lower (wide gap):** Mercs stay longer even as APR normalizes — "
                            "they've entered and have inertia. More optimistic.\n"
                            "- **Equal to entry:** No hysteresis — mercs enter and exit at the same "
                            "threshold. Very twitchy behavior."
                        ),
                    )
                    / 100.0
                )
            with mc3:
                merc_max_frac = (
                    st.number_input(
                        "Max capital (% of target TVL)",
                        value=10.0,
                        step=1.0,
                        min_value=0.0,
                        key="merc_max_frac",
                        help=(
                            "**What it is:** Maximum mercenary capital as percentage of target TVL. "
                            "Caps how much hot money can enter the vault at peak APR.\n\n"
                            "**Effect of changing:**\n"
                            "- **Higher (20%+):** Large merc influx possible — TVL can spike "
                            "significantly when APR is high, then crash when it normalizes. "
                            "Stress-tests robustness.\n"
                            "- **Lower (5%):** Small merc impact — even at peak APR, TVL won't "
                            "swing much from mercs. More stable simulations.\n"
                            "- **0:** No mercenary model — only retail + whale dynamics."
                        ),
                    )
                    / 100.0
                )

        with settings_tab_grid:
            st.caption(
                "The optimizer searches a 2D grid of (Budget, r_max) combinations. These bounds "
                "control the r_max dimension of the search grid. Budget bounds are automatically "
                "derived from the r_max range and target TVL. Narrower ranges = faster but may miss "
                "the optimal config."
            )
            gc1, gc2 = st.columns(2)
            with gc1:
                grid_r_min = (
                    st.number_input(
                        "r_max min (%)",
                        value=2.0,
                        step=0.5,
                        min_value=0.0,
                        key="grid_r_min",
                        help=(
                            "**What it is:** Minimum incentive APR cap (r_max) the optimizer will consider. "
                            "Lower bound of the search grid.\n\n"
                            "**Effect of changing:**\n"
                            "- **Higher:** Narrows the search space — optimizer only considers generous "
                            "rate caps. Faster computation but may miss efficient low-rate configs.\n"
                            "- **Lower (1–2%):** Allows the optimizer to explore minimal incentive scenarios. "
                            "Good for discovering if a small nudge is sufficient."
                        ),
                    )
                    / 100.0
                )
            with gc2:
                grid_r_max_raw = (
                    st.number_input(
                        "r_max max (%)",
                        value=GLOBAL_R_MAX_CEILING * 100,
                        step=0.5,
                        min_value=1.0,
                        key="grid_r_max",
                        help=(
                            f"**What it is:** Maximum incentive APR cap the optimizer will consider. "
                            f"Clamped at global ceiling = {GLOBAL_R_MAX_CEILING:.0%}.\n\n"
                            f"**Effect of changing:**\n"
                            f"- **Higher:** Wider search — optimizer can explore aggressive rate configs. "
                            f"But >8% on stablecoin pools typically attracts mercenary capital.\n"
                            f"- **Lower (4–6%):** Conservative — caps the maximum possible APR. Good for "
                            f"mature, stable pools where you want to avoid rate spikes."
                        ),
                    )
                    / 100.0
                )
                grid_r_max = min(grid_r_max_raw, GLOBAL_R_MAX_CEILING)
            st.caption(
                "Protocol defaults: "
                + ", ".join(
                    f"{k}: {lo:.0%}-{hi:.0%}" for k, (lo, hi) in PROTOCOL_R_MAX_DEFAULTS.items()
                )
            )

    # ================================================================
    # SINGLE-VENUE OPTIMIZATION
    # ================================================================

    st.header("Single-Venue Optimization")
    st.caption(
        "Run the full MC optimization framework on a single venue with a fixed budget. "
        "This is completely independent of the multi-venue program optimization -- "
        "pick any venue, set your budget, and explore the loss surface."
    )

    # Venue selection
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

    # Detect venue change -> clear stale results
    _sv_venue_id = sv_venue["name"]
    if st.session_state.get("_sv_last_venue") != _sv_venue_id:
        st.session_state.pop("sv_result", None)
        st.session_state.pop("sv_live_tvl", None)
        st.session_state.pop("sv_live_util", None)
        st.session_state["_sv_last_venue"] = _sv_venue_id

    # Dynamic key prefix -- forces fresh widgets on venue switch
    _svk = f"sv_{_sv_venue_id}_"

    # Auto-fetch live data on venue selection
    _sv_autofetch_key = f"sv_autofetched_{_sv_venue_id}"
    if _sv_autofetch_key not in st.session_state:
        with st.spinner(f"Fetching live data for {sv_venue['name']}..."):
            # 1. Fetch TVL + utilization
            try:
                _af_tvl, _af_util = _fetch_current_tvl_and_util(sv_venue)
            except Exception as _e:
                _af_tvl = sv_venue.get("current_tvl", 0.0)
                _af_util = sv_venue.get("target_util", 0.5)
                st.warning(f"Live TVL fetch failed: {_e}")

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

    # Look up program + cached live data for this venue
    _sv_prog_name = next(
        (pn for pn, p in programs.items() for vv in p["venues"] if vv["name"] == sv_venue["name"]),
        None,
    )
    _sv_cache_key = f"current_values_{_sv_prog_name}" if _sv_prog_name else None
    _sv_cached = (
        st.session_state.get(_sv_cache_key, {}).get(sv_venue["name"], {}) if _sv_cache_key else {}
    )
    _sv_default_current_tvl = _sv_cached.get("current_tvl", sv_venue["current_tvl"]) / 1e6
    _sv_default_target_tvl = _sv_cached.get("current_tvl", sv_venue["target_tvl"]) / 1e6
    _sv_default_util = _sv_cached.get("current_util", sv_venue.get("target_util", 0.5)) * 100
    _sv_base_preview_key = f"sv_base_apy_{sv_venue['name']}"
    _sv_base_preview = st.session_state.get(_sv_base_preview_key, 0.0)

    _sv_default_supply_cap = sv_venue.get("supply_cap", 0.0)

    # Show auto-fill summary
    _sv_rthresh_key_af = f"sv_rthresh_{sv_venue['asset']}"
    _sv_rthresh_data_af = st.session_state.get(_sv_rthresh_key_af, {})
    _sv_rthresh_val_af = _sv_rthresh_data_af.get("r_threshold", 0.045)
    _sv_base_source_af = st.session_state.get(f"{_sv_base_preview_key}_source", "not fetched")
    if _sv_cached or _sv_base_preview > 0:
        st.success(
            f"Auto-filled from live data -- "
            f"TVL: ${_sv_default_current_tvl:.1f}M, "
            f"Util: {_sv_default_util:.0f}%, "
            f"Base APY: {_sv_base_preview:.2%} ({_sv_base_source_af}), "
            f"r_threshold: {_sv_rthresh_val_af:.2%}"
        )

    # Venue parameters
    st.subheader("Venue Parameters")

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

    # Supply Cap
    sv_supply_cap = (
        st.number_input(
            "Supply Cap ($M) -- 0 = unlimited",
            value=_sv_default_supply_cap / 1e6 if _sv_default_supply_cap > 0 else 0.0,
            step=10.0,
            min_value=0.0,
            key=_svk + "supply_cap",
            help=(
                "**What it is:** Maximum TVL the venue can accept (supply cap / deposit limit). "
                "0 = unlimited. The simulation stops accepting deposits when TVL hits this cap.\n\n"
                "**How to configure:** Set to the protocol's on-chain supply cap if one exists. "
                "After running the optimization, check the 'Suggested TVL Cap' advisory in the "
                "results section for data-driven cap recommendations based on your budget and floor APR.\n\n"
                "**Effect on simulation:** Caps TVL growth — prevents the incentive rate from "
                "being diluted below your floor by unlimited TVL growth."
            ),
        )
        * 1e6
    )

    # ==================================================================
    # MODE SELECTOR: Enforce Budget vs Enforce APR
    # ==================================================================
    st.markdown("---")
    sv_mode = st.radio(
        "**Optimization Mode**",
        ["Enforce Budget", "Enforce APR"],
        index=1,
        key=_svk + "mode",
        horizontal=True,
        help=(
            "**Enforce Budget:** You specify a weekly budget ceiling. The optimizer "
            "finds the best (B, r_max) pair within that budget.\n\n"
            "**Enforce APR:** You specify the incentive APR you want maintained at "
            "target TVL. Budget and r_max are derived automatically to meet that target."
        ),
    )

    _is_set_apr_mode = "Enforce APR" in sv_mode

    WEEKS_PER_YEAR_UI = 365.0 / 7.0

    if _is_set_apr_mode:
        # ENFORCE APR MODE
        st.info(
            "**Enforce APR Mode -- Budget Minimizer:** Specify the incentive APR you want "
            "delivered at target TVL. The optimizer searches for the **cheapest** (B, r_max) pair "
            "that keeps incentive APR >= your target throughout the simulation -- including "
            "whale exits, TVL fluctuations, and competitor dynamics.\n\n"
            "The derived budget shown below is the **theoretical maximum** needed. "
            "The optimizer will try to find a cheaper configuration that still meets "
            "the incentive APR target."
        )

        sv_apr_c1, sv_apr_c2, sv_apr_c3, sv_apr_c4 = st.columns(4)
        with sv_apr_c1:
            sv_target_inc_apr = (
                st.number_input(
                    "Target Incentive APR (%)",
                    value=5.0,
                    step=0.25,
                    min_value=0.5,
                    key=_svk + "set_apr",
                    help=(
                        "**What it is:** The incentive-only APR (Merkl rewards, excluding base yield) "
                        "you want depositors to earn at target TVL.\n\n"
                        "**How it works:** The optimizer finds the minimum weekly budget that "
                        "delivers at least this incentive APR at target TVL throughout the "
                        "simulation including whale exits, TVL fluctuations, and competitor dynamics.\n\n"
                        "**How to configure:** Set this to the incentive rate needed to attract/retain "
                        "capital on top of the venue's base APY. Total APR = base APY + this value."
                    ),
                )
                / 100.0
            )
        with sv_apr_c2:
            sv_floor_inc_apr = (
                st.number_input(
                    "Incentive APR Floor (%)",
                    value=sv_target_inc_apr * 100,
                    step=0.25,
                    min_value=0.0,
                    key=_svk + "set_apr_floor",
                    help=(
                        "**What it is:** The minimum acceptable incentive APR. When the incentive "
                        "rate drops below this floor, the optimizer penalizes that configuration heavily.\n\n"
                        "**How it works:** This creates a soft or hard lower bound (controlled by "
                        "Floor Strictness). The optimizer avoids configs where the incentive APR "
                        "frequently dips below this level across Monte Carlo paths.\n\n"
                        "**How to configure:**\n"
                        "- Set equal to Target Incentive APR for a hard minimum\n"
                        "- Set slightly below Target for a soft buffer (brief dips tolerated)\n"
                        "- Set to 0 to disable floor enforcement entirely\n\n"
                        "**When it matters:** APY-sensitive vaults with loopers/leveraged positions "
                        "need a tight floor. Vaults with sticky institutional capital can tolerate a looser floor."
                    ),
                )
                / 100.0
            )
        with sv_apr_c3:
            sv_ceiling_inc_apr = (
                st.number_input(
                    "Incentive APR Ceiling (%)",
                    value=min(10.0, sv_target_inc_apr * 100 * 2),
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

        _sv_set_inc_rate = max(0.005, sv_target_inc_apr)
        _sv_ref_tvl = max(sv_current_tvl, sv_target_tvl)
        _sv_derived_budget = _sv_ref_tvl * _sv_set_inc_rate / WEEKS_PER_YEAR_UI

        st.markdown(
            f"**Derived Parameters:**\n"
            f"- Base APY (informational): **{_sv_base_preview:.2%}** "
            f"(total APR at target will be ~{sv_target_inc_apr + _sv_base_preview:.2%})\n"
            f"- Reference TVL (max of current/target): "
            f"**${_sv_ref_tvl / 1e6:.1f}M**\n"
            f"- Budget search ceiling: **${_sv_derived_budget * 1.1:,.0f}/wk** "
            f"(derived ${_sv_derived_budget:,.0f}/wk + 10% headroom)\n"
            f"- r_max search range: **{sv_target_inc_apr:.2%}** -- "
            f"**{sv_ceiling_inc_apr:.2%}** (incentive, protocol-clamped)\n"
            f"- Optimizer objective: **minimize budget** while keeping "
            f"incentive APR >= {sv_target_inc_apr:.2%} at target TVL"
        )

        if _sv_base_preview <= 0:
            st.warning(
                "Base APY not fetched yet -- base APY shown as 0%. "
                "Fetch base APY below for accurate total APR estimate."
            )

        sv_budget = _sv_derived_budget * 1.1
        # Convert incentive APR inputs to total APR for engine compatibility
        sv_floor_apr = sv_floor_inc_apr + _sv_base_preview
        sv_apr_sensitivity = sv_set_apr_sensitivity
        sv_forced_rate = None
        sv_pin_b_val = None
        sv_pin_r_val = None
        sv_r_lo_total = sv_target_inc_apr + _sv_base_preview
        sv_r_hi_total = sv_ceiling_inc_apr + _sv_base_preview

    else:
        # ENFORCE BUDGET MODE (existing behavior)
        st.info(
            "**Enforce Budget Mode:** You specify the weekly budget ceiling. "
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
                f"Base APY: {_sv_base_preview:.2%} -> "
                f"Incentive r_max range: "
                f"{max(0, sv_r_lo_total - _sv_base_preview):.2%} -- "
                f"{max(0, sv_r_hi_total - _sv_base_preview):.2%}"
            )
        else:
            st.caption("Fetch base APY first for accurate total-to-incentive conversion.")

        # Optional Constraints (budget mode only)
        with st.expander("Advanced Constraints", expanded=False):
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
                    help="Override optimizer -- set a fixed incentive rate.",
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

            # APY Sensitivity (floor APR)
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

    # Pre-flight sanity check
    _expected_tvl = _sv_default_current_tvl * 1e6
    if _expected_tvl > 0 and sv_current_tvl > 0:
        _tvl_ratio = sv_current_tvl / _expected_tvl
        if _tvl_ratio > 2.0 or _tvl_ratio < 0.3:
            st.error(
                f"**Current TVL (${sv_current_tvl / 1e6:.1f}M) looks wrong** -- "
                f"expected ~${_expected_tvl / 1e6:.1f}M for {sv_venue['name']}. "
                f"This can happen when switching venues. "
                f"Please correct the value or click 'Fetch Live TVL' below."
            )

    # Competitive landscape: sibling venues in the same program
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
                        "Venue": ("-> " if _is_current else "") + _sv["name"],
                        "TVL ($M)": _sib_tvl / 1e6,
                        "Util": _sib_util,
                        "Base APY": _sib_base,
                        "Share": 0.0,
                    }
                )
            for _row in _sib_rows:
                _row["Share"] = _row["TVL ($M)"] * 1e6 / max(_total_prog_tvl, 1)
            with st.expander(
                f"Program Context: {_sv_prog_name} -- "
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

    # Fetch Live TVL
    sv_live_col1, sv_live_col2 = st.columns([1, 3])
    with sv_live_col1:
        if st.button("Fetch Live TVL", key="sv_fetch_live_tvl"):
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

    # Fetch base APY and r_threshold for this venue
    sv_fetch_col1, sv_fetch_col2 = st.columns(2)

    with sv_fetch_col1:
        st.subheader("Base APY")
        sv_base_key = f"sv_base_apy_{sv_venue['name']}"
        if st.button("Fetch Base APY", key="sv_fetch_base"):
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
            st.caption("Click 'Fetch Base APY' to load -- required for accurate results.")
        else:
            st.caption(f"Source: {sv_base_source}")

    with sv_fetch_col2:
        st.subheader("r_threshold")
        sv_rthresh_key = f"sv_rthresh_{sv_venue['asset']}"
        if st.button("Fetch r_threshold", key="sv_fetch_rthresh"):
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
            st.caption("Click 'Fetch r_threshold' to load competitor rates.")
        else:
            st.caption(f"Source: {sv_rthresh_source} ({n_comp} competitors)")

    # Manual r_threshold override
    sv_rthresh_ov = st.number_input(
        "r_threshold Override (%) -- leave 0 to use fetched",
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

    # Historical Data Calibration
    sv_pool_id = sv_venue.get("defillama_pool_id", "")
    if sv_pool_id:
        st.markdown("---")
        st.subheader("Historical Data & Calibration")
        hist_key = f"sv_hist_{sv_pool_id}"
        cal_key = f"sv_cal_{sv_pool_id}"

        if st.button("Fetch 90-Day History & Calibrate", key="sv_fetch_hist"):
            with st.spinner(f"Fetching DeFiLlama history for {sv_venue['name']}..."):
                try:
                    from campaign.historical import calibrate_retail_params, fetch_pool_history

                    hist = fetch_pool_history(sv_pool_id, days=90)
                    cal = calibrate_retail_params(hist, default_r_threshold=sv_r_threshold)
                    st.session_state[hist_key] = hist
                    st.session_state[cal_key] = cal
                    st.success(
                        f"Fetched {hist.days} days of history. Calibration quality: {cal.data_quality}"
                    )
                except Exception as e:
                    st.warning(f"Historical data fetch failed: {e}")

        if hist_key in st.session_state:
            hist = st.session_state[hist_key]
            cal = st.session_state[cal_key]

            st.markdown("**Calibrated Parameters (from history):**")
            hc1, hc2, hc3, hc4 = st.columns(4)
            with hc1:
                st.metric("a+ (inflow)", f"{cal.alpha_plus:.3f}")
            with hc2:
                st.metric("a- multiplier", f"{cal.alpha_minus_multiplier:.1f}")
            with hc3:
                st.metric("sigma (noise)", f"{cal.diffusion_sigma:.4f}")
            with hc4:
                st.metric("Lag (days)", f"{cal.response_lag_days:.0f}")

            st.caption(
                f"Data quality: **{cal.data_quality}** | {cal.n_observations} observations | "
                f"r_threshold mean: {cal.r_threshold_mean:.2%} | "
                f"r_threshold trend: {cal.r_threshold_trend * 365:.1%}/yr"
            )

            if cal.data_quality != "insufficient":
                if st.button("Apply Calibrated Values to Settings", key="sv_apply_cal"):
                    st.session_state["alpha_plus"] = cal.alpha_plus
                    st.session_state["alpha_minus_mult"] = cal.alpha_minus_multiplier
                    st.session_state["diffusion_sigma"] = cal.diffusion_sigma
                    st.session_state["response_lag"] = cal.response_lag_days
                    cal_merc_entry = cal.r_threshold_mean * 1.8
                    cal_merc_exit = cal.r_threshold_mean * 1.3
                    st.session_state["merc_entry"] = cal_merc_entry * 100
                    st.session_state["merc_exit"] = cal_merc_exit * 100
                    st.success(
                        f"Applied: a+={cal.alpha_plus:.3f}, a-x={cal.alpha_minus_multiplier:.1f}, "
                        f"sigma={cal.diffusion_sigma:.4f}, lag={cal.response_lag_days:.0f}d, "
                        f"merc entry={cal_merc_entry:.2%}, exit={cal_merc_exit:.2%}"
                    )
                    st.info("Values updated in settings. Re-run optimization to use them.")
            else:
                st.warning("Insufficient data quality -- cannot apply calibrated values.")

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

    # Dune Data Sync (single venue)
    st.markdown("---")
    sv_dune_col1, sv_dune_col2 = st.columns([1, 3])
    with sv_dune_col1:
        if st.button("Sync Dune Data", key="sv_dune_sync"):
            with st.spinner(f"Syncing Dune data for {sv_venue['name']}..."):
                try:
                    import sys

                    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                    from campaign.venue_registry import get_venue
                    from dune.sync import sync_venue as dune_sync_venue

                    venue_rec = get_venue(sv_venue["pool_id"])
                    result = dune_sync_venue(venue_rec, days=90)
                    if result.skipped:
                        st.warning(f"Skipped: {result.skip_reason}")
                    else:
                        st.success(
                            f"Synced: {result.whale_flows_count} whale flows, "
                            f"{result.mercenary_count} mercenary addresses"
                        )
                except Exception as e:
                    st.error(f"Dune sync failed: {e}")
    with sv_dune_col2:
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from dune.sync import get_sync_status

            status = get_sync_status()
            pool_status = status.get(sv_venue.get("pool_id", ""))
            if pool_status and pool_status["has_whale_flows"]:
                st.caption(
                    f"Cached: {pool_status['whale_count']} whale flows, "
                    f"{pool_status['merc_count']} mercenary addresses "
                    f"(last sync: {pool_status['last_modified']})"
                )
            else:
                st.caption("No cached Dune data for this venue.")
        except Exception:
            st.caption("Dune sync available -- click to fetch whale flow data.")

    # Run button
    sv_run = st.button(
        "Run Single-Venue Optimization",
        type="primary",
        use_container_width=True,
        key="sv_run_btn",
    )

    if sv_run:
        sv_venue_run = {**sv_venue, "current_tvl": sv_current_tvl}

        # Fetch whales
        with st.spinner(f"Fetching whales for {sv_venue['name']}..."):
            sv_whale_hist = None
            try:
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from dune.sync import build_whale_history_lookup

                sv_whale_hist = build_whale_history_lookup(sv_venue.get("pool_id", ""))
                if sv_whale_hist:
                    st.info(f"Using Dune whale history ({len(sv_whale_hist)} addresses)")
            except Exception:
                pass
            try:
                sv_whales = _fetch_whales_for_venue(
                    sv_venue_run, sv_r_threshold, whale_history=sv_whale_hist
                )
            except Exception as e:
                st.warning(f"Whale fetch failed: {e}")
                sv_whales = []

        # MC path guardrails
        min_paths = max(30, 10 * max(len(sv_whales), 1))
        sv_paths_eff = max(sv_paths, min_paths)
        if sv_paths_eff > sv_paths:
            st.info(
                f"MC paths raised {sv_paths} -> {sv_paths_eff} "
                f"(whale guardrail: {len(sv_whales)} whales)"
            )

        sv_weights = LossWeights(
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
            apr_ceiling=sv_r_hi_total if _is_set_apr_mode else apr_ceiling_val,
            tvl_target=sv_target_tvl,
            apr_stability_on_total=True,
            apr_floor=sv_floor_apr,
            apr_floor_sensitivity=sv_apr_sensitivity
            if not _is_set_apr_mode
            else max(sv_apr_sensitivity, 0.85),
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

        sv_apy_sensitive = None
        if sv_floor_apr > 0:
            sv_apy_sensitive = APYSensitiveConfig(
                floor_apr=sv_floor_apr,
                sensitivity=sv_apr_sensitivity,
                max_sensitive_tvl=sv_target_tvl * 0.10,
            )

        # Convert total APR bounds to incentive r_max range
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
                target_inc_apr=sv_target_inc_apr if _is_set_apr_mode else 0.0,
            )
            sv_elapsed = time.time() - t0

        # Compute forced-rate budget feasibility
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
            "set_apr_target": sv_target_inc_apr if _is_set_apr_mode else None,
            "floor_inc_apr": sv_floor_inc_apr if _is_set_apr_mode else None,
        }

    # ── Display results ──
    if "sv_result" not in st.session_state:
        return

    ir = st.session_state["sv_result"]
    sv_sr = ir["surface"]
    sv_v = ir["venue"]
    sv_base_disp = ir["base_apy"]
    sv_target_disp = ir["target_tvl"]

    # Apply r_max floor
    WEEKS_PER_YEAR = 365.0 / 7.0
    sv_B = sv_sr.optimal_B
    sv_r_raw = sv_sr.optimal_r_max
    float_rate = sv_B / max(sv_target_disp, 1.0) * WEEKS_PER_YEAR
    sv_proto = sv_v.get("protocol", "").lower()
    _, sv_proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(sv_proto, (0.02, GLOBAL_R_MAX_CEILING))
    sv_venue_ceiling = min(sv_proto_hi, GLOBAL_R_MAX_CEILING)
    sv_r = max(sv_r_raw, min(float_rate, sv_venue_ceiling))

    if sv_r > sv_r_raw + 1e-6:
        _mode_for_msg = ir.get("mode", "set_budget")
        _target_for_msg = ir.get("set_apr_target") if _mode_for_msg == "set_apr" else None
        st.info(
            f"**r_max adjusted to float rate:** The optimizer selected r_max={sv_r_raw:.2%}, "
            f"but at target TVL the float rate (B ÷ TVL × 52.14) = {float_rate:.2%}. "
            f"Since the float rate already exceeds the cap, the cap would never bind — "
            f"the budget delivers {float_rate:.2%} incentive at target TVL without needing "
            f"to be limited. r_max has been raised to match the float rate ({sv_r:.2%}) "
            f"so the full budget is always deployable."
            + (
                f" This is slightly above your {_target_for_msg:.2%} target — that's fine."
                if _target_for_msg
                else ""
            )
        )

    sv_tb = t_bind(sv_B, sv_r)

    st.markdown("---")
    _result_mode = ir.get("mode", "set_budget")
    _mode_label = "Enforce APR" if _result_mode == "set_apr" else "Enforce Budget"
    st.header(f"Results: {sv_v['name']}  ({_mode_label})")

    mc = sv_sr.optimal_mc_result

    # ── Compute campaign type + Merkl fields early (shown first) ──
    if sv_tb < sv_target_disp * 0.5:
        ctype = "Float-like"
        ctype_desc = "Cap rarely binds -- rate floats inversely with TVL"
    elif sv_tb > sv_target_disp * 1.2:
        ctype = "MAX-like"
        ctype_desc = "Cap always binds -- effectively constant rate"
    else:
        ctype = "Hybrid"
        ctype_desc = "Cap binds at low TVL, floats at high TVL"
    merkl_type = "Hybrid" if ctype == "Hybrid" else ("MAX" if ctype == "MAX-like" else "Float")
    inc_at_target = apr_at_tvl(sv_B, sv_target_disp, sv_r)
    total_apr_at_target = sv_base_disp + inc_at_target

    # ── Risk data (needed by inline status inside the Merkl card) ──
    sv_risks = []
    sv_forced_info = ir.get("forced_rate_info")
    sv_ir_floor = ir.get("floor_apr", 0.0)
    sv_ir_sens = ir.get("apr_sensitivity", 0.0)

    if sv_forced_info and sv_forced_info.get("overspend"):
        sv_risks.append(
            f"**Budget overspend required:** Forced rate {sv_forced_info['forced_rate']:.2%} "
            f"needs ${sv_forced_info['required_budget']:,.0f}/wk but input budget is "
            f"${sv_forced_info['input_budget']:,.0f}/wk "
            f"(+${sv_forced_info['overspend_amount']:,.0f}/wk over budget)"
        )

    if sv_ir_floor > 0 and total_apr_at_target < sv_ir_floor:
        gap = sv_ir_floor - total_apr_at_target
        sv_risks.append(
            f"**Floor APR at risk:** Total APR at target TVL ({total_apr_at_target:.2%}) "
            f"is below the floor APR ({sv_ir_floor:.2%}) by {gap:.2%}. "
            + (
                f"APR sensitivity is {sv_ir_sens:.0%} -- "
                f"{'high risk of rapid TVL unwind.' if sv_ir_sens > 0.5 else 'moderate risk.'} "
                if sv_ir_sens > 0
                else ""
            )
            + "Consider increasing budget or forcing a higher incentive rate."
        )

    sv_ir_rthresh = ir.get("r_threshold", 0.045)
    if total_apr_at_target < sv_ir_rthresh:
        sv_risks.append(
            f"**Below competitor rate:** Total APR at target ({total_apr_at_target:.2%}) "
            f"< r_threshold ({sv_ir_rthresh:.2%}). Venue will be uncompetitive. "
            f"Consider increasing budget or lowering target TVL."
        )

    if mc and mc.mean_budget_util < 0.5:
        sv_risks.append(
            f"**Low budget utilization:** Only {mc.mean_budget_util:.0%} of budget expected "
            f"to be spent. TVL is too low for the r_max cap. Consider lowering r_max."
        )

    inc_at_current = apr_at_tvl(sv_B, sv_v["current_tvl"], sv_r)
    if inc_at_current > 0.10:
        sv_risks.append(
            f"**High incentive at current TVL:** {inc_at_current:.2%} may attract "
            f"mercenary capital. Consider a lower r_max to smooth the transition."
        )

    # ── Merkl Campaign Instructions (shown first, top of results) ──
    st.subheader("Merkl Campaign Instructions")
    st.caption(
        "Copy these exact values into Merkl when setting up this campaign. "
        "Campaign Type is derived from where T_bind sits relative to target TVL."
    )
    with st.container(border=True):
        st.markdown(f"**{sv_v['name']}** ({sv_v['asset']})")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"**Campaign Type:** `{merkl_type}`")
        with m2:
            st.markdown(f"**Weekly Budget:** `${sv_B:,.0f}`")
        with m3:
            if merkl_type == "MAX":
                st.markdown(f"**Incentive Rate:** `{sv_r:.2%}`")
            else:
                st.markdown(f"**Max Incentive Rate (r_max):** `{sv_r:.2%}`")
        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.caption(f"Base APY: {sv_base_disp:.2%}")
        with detail_cols[1]:
            st.caption(f"Incentive @ Target: {inc_at_target:.2%}")
        with detail_cols[2]:
            st.caption(f"Total APR @ Target: {total_apr_at_target:.2%}")
        with detail_cols[3]:
            st.caption(f"T_bind: ${sv_tb / 1e6:.0f}M")

        # Inline status -- mode-aware
        if _result_mode == "set_apr":
            _derived_B_disp = ir.get("derived_budget")
            _set_apr_disp = ir.get("set_apr_target") or sv_ir_floor
            _floor_inc_disp = ir.get("floor_inc_apr") or max(0, sv_ir_floor - sv_base_disp)
            if _derived_B_disp and _derived_B_disp > 0:
                _sav = _derived_B_disp - sv_B
                if _sav > 0:
                    st.success(
                        f"**Budget minimized:** ${sv_B:,.0f}/wk "
                        f"(${_sav:,.0f}/wk saved vs derived ${_derived_B_disp:,.0f}/wk). "
                        f"Target incentive APR: {_set_apr_disp:.2%} at target TVL."
                    )
                else:
                    st.info(
                        f"**Enforce APR result:** ${sv_B:,.0f}/wk needed to deliver "
                        f"{_set_apr_disp:.2%} incentive APR. No cheaper config found."
                    )
            else:
                st.info(
                    f"**Enforce APR result:** ${sv_B:,.0f}/wk at r_max={sv_r:.2%} "
                    f"targeting {_set_apr_disp:.2%} incentive APR (floor: {_floor_inc_disp:.2%})."
                )
        elif sv_forced_info:
            if sv_forced_info.get("overspend"):
                st.error(
                    f"**Budget gap:** Forced rate {sv_forced_info['forced_rate']:.2%} requires "
                    f"${sv_forced_info['required_budget']:,.0f}/wk -- "
                    f"${sv_forced_info['overspend_amount']:,.0f}/wk over input budget. "
                    f"Increase budget or accept overspend."
                )
            else:
                st.success(
                    f"Forced rate {sv_forced_info['forced_rate']:.2%} is feasible "
                    f"(requires ${sv_forced_info['required_budget']:,.0f}/wk, "
                    f"within ${sv_forced_info['input_budget']:,.0f}/wk budget)"
                )

        if sv_ir_floor > 0 and total_apr_at_target < sv_ir_floor:
            _floor_inc_sug = ir.get("floor_inc_apr") or max(0, sv_ir_floor - sv_base_disp)
            st.warning(
                f"**Floor gap:** Incentive APR at target TVL ({inc_at_target:.2%}) is below "
                f"the floor ({_floor_inc_sug:.2%}). "
                f"Need a weekly budget of at least "
                f"${sv_target_disp * _floor_inc_sug / WEEKS_PER_YEAR:,.0f} "
                f"to deliver {_floor_inc_sug:.2%} incentive at ${sv_target_disp / 1e6:.0f}M TVL."
            )

        # Suggested TVL Cap Advisory
        if mc and sv_B > 0:
            _WPY_cap = 365.0 / 7.0

            _floor_inc = max(0, sv_ir_floor - sv_base_disp) if sv_ir_floor > 0 else 0
            _floor_tvl = (sv_B * _WPY_cap / _floor_inc) if _floor_inc > 0 else float("inf")

            _tbind_tvl = t_bind(sv_B, sv_r)

            _sv_rthresh = ir.get("r_threshold", 0.045)
            _rthresh_inc = max(0, _sv_rthresh - sv_base_disp)
            _rthresh_tvl = (sv_B * _WPY_cap / _rthresh_inc) if _rthresh_inc > 0 else float("inf")

            _mc_p95_tvl = getattr(mc, "tvl_max_p95", sv_target_disp * 1.5)
            if _floor_tvl < float("inf"):
                _suggested_safe_tvl = _floor_tvl
                _suggested_safe_tvl = max(
                    _suggested_safe_tvl, _mc_p95_tvl if _mc_p95_tvl > 0 else sv_target_disp
                )
            else:
                _suggested_safe_tvl = max(
                    sv_target_disp * 1.5, _mc_p95_tvl if _mc_p95_tvl > 0 else sv_target_disp * 1.5
                )

            if _floor_inc > 0:
                _severity = (
                    min(_floor_inc / max(_sv_rthresh - sv_base_disp, _floor_inc), 1.0)
                    if _sv_rthresh > sv_base_disp
                    else 0.5
                )
                _danger_mult = 2.0 - 0.7 * _severity
                _danger_tvl = _floor_tvl * _danger_mult
            else:
                _danger_tvl = (
                    _rthresh_tvl * 1.5 if _rthresh_tvl < float("inf") else sv_target_disp * 3.0
                )

            _rate_at_safe = apr_at_tvl(sv_B, _suggested_safe_tvl, sv_r)
            _rate_at_danger = apr_at_tvl(sv_B, _danger_tvl, sv_r)
            _rate_at_current_cap = (
                apr_at_tvl(sv_B, sv_supply_cap, sv_r) if sv_supply_cap > 0 else None
            )

            _current_tvl_cap = sv_supply_cap
            if _current_tvl_cap <= 0:
                _cap_status = "No Cap"
                _cap_verdict = "No supply cap is set -- TVL is uncapped. Consider setting one to protect against rate dilution."
            elif _current_tvl_cap <= _suggested_safe_tvl * 1.05:
                _cap_status = "Safe"
                _cap_verdict = "Your supply cap is within the safe range."
            elif _current_tvl_cap <= _danger_tvl:
                _cap_status = "Caution"
                _cap_verdict = "Your supply cap is above the safe level. If TVL reaches this cap, the incentive rate may drop below your floor."
            else:
                _cap_status = "Danger"
                _cap_verdict = "Your supply cap is in the danger zone -- at this TVL, the budget cannot sustain adequate incentive rates."

            with st.container(border=True):
                st.markdown(f"### Suggested TVL Cap ({_cap_status})")
                cap_c1, cap_c2, cap_c3 = st.columns(3)
                with cap_c1:
                    st.metric(
                        "Suggested Safe Cap",
                        f"${_suggested_safe_tvl / 1e6:.0f}M",
                        help=(
                            f"TVL where incentive rate = floor ({_floor_inc:.2%}). "
                            f"At this TVL, rate = {_rate_at_safe:.2%} "
                            f"(total APR: {_rate_at_safe + sv_base_disp:.2%})."
                        ),
                    )
                with cap_c2:
                    st.metric(
                        "Danger Threshold",
                        f"${_danger_tvl / 1e6:.0f}M",
                        help=(
                            f"Beyond this TVL, budget cannot sustain meaningful incentives. "
                            f"Rate drops to {_rate_at_danger:.2%}."
                        ),
                    )
                with cap_c3:
                    if _current_tvl_cap > 0:
                        st.metric(
                            "Current Supply Cap",
                            f"${_current_tvl_cap / 1e6:.0f}M",
                            delta=f"{(_current_tvl_cap - _suggested_safe_tvl) / 1e6:+.0f}M vs safe",
                            delta_color="inverse",
                        )
                    else:
                        st.metric("Current Supply Cap", "Unlimited")

                st.caption(f"{_cap_verdict}")
                if _floor_inc > 0 and _floor_tvl < float("inf"):
                    st.caption(
                        f"**Derivation:** At budget ${sv_B:,.0f}/wk, incentive rate = floor "
                        f"({_floor_inc:.2%}) at TVL = ${_floor_tvl / 1e6:.0f}M. "
                        f"Danger zone starts at ${_danger_tvl / 1e6:.0f}M (rate = {_rate_at_danger:.2%})."
                    )

    # ── Risk Assessment ──
    if sv_risks:
        st.markdown("### Risk Assessment")
        with st.container(border=True):
            for risk in sv_risks:
                st.warning(risk)
        st.markdown("---")

    # ── APR Constraint Status (Enforce APR mode) ──
    if _result_mode == "set_apr" and mc:
        _floor = ir.get("floor_apr", 0)
        _derived_B = ir.get("derived_budget")

        if _floor > 0:
            # Strictness-based APR check:
            # Static gate: can the budget physically deliver target incentive APR at target TVL?
            # Dynamic gate: does mean_time_below_floor respect the strictness threshold?
            _WEEKS_PER_YEAR_CHK = 365.0 / 7.0
            _target_inc = ir.get("set_apr_target") or 0.0
            _floor_inc = ir.get("floor_inc_apr") or 0.0
            _strictness = ir.get("apr_sensitivity") or 0.9

            _inc_at_tgt_static = min(sv_r, sv_B / max(sv_target_disp, 1.0) * _WEEKS_PER_YEAR_CHK)
            _static_ok = (_target_inc <= 0) or (_inc_at_tgt_static >= _target_inc * 0.99)
            _allowed_violation = 1.0 - _strictness
            _dynamic_ok = mc.mean_time_below_floor <= _allowed_violation + 1e-6
            _apr_ok = _static_ok and _dynamic_ok

            if _derived_B and _derived_B > 0:
                _savings = _derived_B - sv_B
                _savings_pct = (_savings / _derived_B) * 100
                if _savings > 0:
                    st.success(
                        f"**Budget optimized:** Optimizer found **${sv_B:,.0f}/wk** "
                        f"(saved **${_savings:,.0f}/wk** = **{_savings_pct:.0f}%** vs "
                        f"derived ceiling ${_derived_B:,.0f}/wk) "
                        f"while delivering target incentive APR >= {_target_inc:.2%}."
                    )
                else:
                    st.info(
                        f"**Budget at ceiling:** Optimizer needs **${sv_B:,.0f}/wk** "
                        f"(approx derived ${_derived_B:,.0f}/wk) to deliver {_target_inc:.2%} "
                        f"incentive APR -- no cheaper feasible configuration found."
                    )

            if _apr_ok:
                st.success(
                    f"**APR constraint met:** Target {_target_inc:.2%} incentive APR is achievable "
                    f"at target TVL (static: {_inc_at_tgt_static:.2%}). "
                    f"Time below floor: {mc.mean_time_below_floor:.1%} "
                    f"(allowed: {_allowed_violation:.1%})."
                )
            else:
                _reason_parts = []
                if not _static_ok:
                    _reason_parts.append(
                        f"budget cannot deliver {_target_inc:.2%} incentive at target TVL "
                        f"(achieves {_inc_at_tgt_static:.2%})"
                    )
                if not _dynamic_ok:
                    _reason_parts.append(
                        f"floor breached {mc.mean_time_below_floor:.1%} of time "
                        f"(strictness allows {_allowed_violation:.1%})"
                    )
                st.error(
                    "**APR constraint not met:** " + "; ".join(_reason_parts) + ". "
                    "Increase budget, reduce target TVL, or loosen strictness."
                )

            _headroom_abs = mc.apr_p5 - _floor
            _headroom_mean = mc.mean_apr - _floor
            _floor_breach_cost = mc.loss_components.get("floor_breach", 0.0)
            _time_below = mc.mean_time_below_floor

            if _headroom_abs >= _floor * 0.15:
                _hr_label = "Large headroom"
                _hr_advice = "Budget could potentially be lowered further."
            elif _headroom_abs >= _floor * 0.03:
                _hr_label = "Moderate headroom"
                _hr_advice = "Some room to lower, but limited flexibility."
            elif _headroom_abs >= 0:
                _hr_label = "Tight -- at the edge"
                _hr_advice = "No room to lower budget. Whale exits or TVL spikes may breach floor."
            else:
                _hr_label = "BREACHING"
                _hr_advice = "APR drops below target in worst-case scenarios. Increase budget."

            with st.container(border=True):
                st.markdown(f"### APR Headroom: {_hr_label}")
                hr1, hr2, hr3, hr4 = st.columns(4)
                with hr1:
                    st.metric(
                        "p5 Headroom",
                        f"{_headroom_abs:+.2%}",
                        help="APR p5 minus floor — worst-case headroom across MC paths.",
                    )
                with hr2:
                    st.metric(
                        "Mean Headroom",
                        f"{_headroom_mean:+.2%}",
                        help="Mean APR minus floor — average headroom.",
                    )
                with hr3:
                    st.metric(
                        "Time Below Floor",
                        f"{_time_below:.1%}",
                        help="Average fraction of simulation time where APR drops below target.",
                    )
                with hr4:
                    st.metric(
                        "Floor Breach Cost",
                        f"{_floor_breach_cost:.2e}",
                        help="Loss component from APR floor breaches (lower = better).",
                    )
                st.caption(f"{_hr_advice}")

    # ── Key Metrics ──
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    with rc1:
        st.metric(
            "B*",
            f"${sv_B:,.0f}/wk",
            help="**Optimal Weekly Budget.** The weekly incentive spend the optimizer recommends. "
            "In Enforce APR mode, this is the minimum budget that delivers the target incentive APR. "
            "In Enforce Budget mode, this is the best allocation within your budget ceiling.",
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
                "APR Range (p5-p95)",
                f"{mc.apr_p5:.1%} -- {mc.apr_p95:.1%}",
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

    st.info(f"**Campaign Type: {ctype}** -- {ctype_desc}")

    # APR at key TVL levels
    st.subheader("Incentive APR at Key TVL Levels")
    st.caption(
        "**Level:** TVL scenario name. **TVL ($M):** Dollar value of deposits. "
        "**Incentive APR:** Annualized incentive-only rate at that TVL (B/TVL × 52.14, capped at r_max). "
        "**Total APR:** Base APY + Incentive APR. "
        "**Regime:** Cap binds = rate equals r_max (MAX-like), Float = rate is below cap (Float-like)."
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
        regime = "Cap binds" if tvl_val < t_bind(sv_B, sv_r) else "Float"
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
            "Configs with loss within 5% of optimal. **B ($/wk):** weekly budget. "
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
        ax.set_xlabel("r_max -- Incentive APR Cap (%)")
        ax.set_ylabel("B -- Weekly Budget ($k)")
        ax.set_title(f"Loss Surface -- {sv_v['name']}")
        ax.legend()

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

    # Efficiency Metrics
    st.subheader("Efficiency Metrics")
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
    st.subheader("Export")
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

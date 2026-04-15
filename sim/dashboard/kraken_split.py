"""
Kraken Earn Split Incentive Dashboard
======================================
Two-campaign structure for the 4 Kraken-targeted Euler/Morpho vaults.

Campaign A  — all depositors, pro-rata hybrid distribution
Campaign B  — Kraken Earn whitelisted address(es) only, hybrid distribution

Rate math (exact)
-----------------
  rate_A  = min(RmaxA, BudgetA × 52 / TotalTVL_M)   [% / year]
  rate_B  = min(RmaxB, BudgetB × 52 / KrakenTVL_M)  [% / year]
  TotalTVL_M  = (NonKrakenTVL + KrakenTVL) / 1_000_000

  non_kraken_merkl = rate_A
  kraken_merkl     = rate_A + rate_B

  spend_A = min(BudgetA, RmaxA/100 × TotalTVL × 1000 / 52)   [$K/week]
  spend_B = min(BudgetB, RmaxB/100 × KrakenTVL × 1000 / 52)  [$K/week]

  breakeven_B_tvl = BudgetB × 52 / (RmaxB / 100) / 1_000_000  [$M]

Kraken TVL dilutes BOTH campaigns:
  - Dilutes A because KrakenTVL is part of TotalTVL (denominator of rate_A)
  - Dilutes B directly (denominator of rate_B)

Live data sources
-----------------
  - Kraken strategy TVLs : on-chain totalAssets() on strategy contracts
  - Vault total TVLs     : Morpho GraphQL + Euler on-chain + DeFiLlama fallback
  - Base APYs            : campaign.base_apy.fetch_all_base_apys (Morpho GQL / Euler on-chain)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
import streamlit as st

# ── path setup ──────────────────────────────────────────────────────────────
_DASH_DIR = os.path.dirname(__file__)
_SIM_DIR = os.path.abspath(os.path.join(_DASH_DIR, ".."))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(_SIM_DIR, "..", ".env"), override=False)

from campaign.base_apy import (
    BaseAPYResult,
    fetch_all_base_apys,
    fetch_euler_base_apy,
    fetch_morpho_base_apy,
)
from campaign.evm_data import _eth_call, _decode_uint256, get_eth_rpc_url, fetch_euler_vault_data

# ============================================================================
# CONSTANTS & VENUE CONFIGURATION
# ============================================================================

# Kraken Earn strategy addresses for the four lending vaults
KRAKEN_STRATEGY_ADDRS: dict[str, list[str]] = {
    "morpho_rlusd": ["0x8ad9b1cb3128c871dd958c22ec485da32000536b"],
    "morpho_pyusd": ["0xc5e0e2bd8b8663c621b5051d863d072295da9720"],
    "euler_rlusd":  ["0x4d6376dddd67af6f9ad40225ec566212f85b5a16"],
    # Both PYUSD loop strategies deposit into the Euler PYUSD vault
    "euler_pyusd":  [
        "0xba4970b839678168340f823ef8f255832ab18c12",  # Euler PYUSD→USDC loop
        "0xb134641b80982bed7cdbb307e56e55abbc8b3197",  # Euler USDC→PYUSD loop
    ],
}

# Morpho MetaMorpho vault addresses
MORPHO_VAULT_ADDRS = {
    "morpho_pyusd": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
    # RLUSD vault discovered dynamically via GQL; fallback below
    "morpho_rlusd": None,  # will be resolved at runtime
}

# Euler vault addresses (from evm_data.py)
EULER_VAULT_ADDRS = {
    "euler_rlusd": "0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2",
    "euler_pyusd": "0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5",
}

# Curve pool addresses
CURVE_POOL_ADDRS = {
    "curve_rlusd": "0xD001aE433f254283FeCE51d4ACcE8c53263aa186",  # RLUSD-USDC
    "curve_pyusd": "0x383E6b4437b59fff47B619CBA855CA29342A8559",  # PYUSD-USDC
}

# Kraken Earn LP addresses for Curve pools
# (separate from KRAKEN_STRATEGY_ADDRS because Curve uses LP-token balance, not ERC4626)
CURVE_KRAKEN_LP_ADDRS: dict[str, str | None] = {
    "curve_pyusd": "0xb11ed12e302815c8c5f12a3a1a93ebd7bd730a21",  # Curve PayPool LP
    "curve_rlusd": None,  # no Kraken address yet — will be added in future
}

# DeFiLlama pool IDs for Curve TVL lookup
CURVE_DEFILLAMA_IDS = {
    "curve_rlusd": "e91e23af-9099-45d9-8ba5-ea5b4638e453",
    "curve_pyusd": "14681aee-05c9-4733-acd0-7b2c84616209",
}

RLUSD_ADDRESS = "0x8292Bb45bf1Ee4d140127049757C0C38e47a8A75"
PYUSD_ADDRESS = "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8"
STABLECOIN_DECIMALS = 6   # PYUSD decimals
RLUSD_DECIMALS = 18        # RLUSD is a standard 18-decimal ERC20

MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"

@dataclass
class VenueConfig:
    key: str
    label: str
    asset: str
    protocol: str
    # Current live numbers (defaults; overridden by live fetch)
    default_total_tvl_m: float       # $M total vault TVL
    default_kraken_tvl_m: float      # $M Kraken lending TVL
    default_budget_a: float          # $K / week
    default_budget_b: float          # $K / week
    default_rmax_a: float            # % / year cap on campaign A
    default_rmax_b: float            # % / year cap on campaign B
    default_base_apy: float          # % / year base organic APY
    floor_apy: Optional[float]       # % / year all-in floor (None = no constraint)
    floor_label: Optional[str]       # human label for the floor constraint
    # For Morpho GQL / Euler on-chain fetching
    vault_address: Optional[str] = None
    vault_decimals: int = 6
    # base_apy fetch config re-used from app5.py
    base_apy_config: dict = field(default_factory=dict)


VENUES: list[VenueConfig] = [
    VenueConfig(
        key="morpho_rlusd",
        label="Morpho RLUSD",
        asset="RLUSD",
        protocol="morpho",
        default_total_tvl_m=198.6,
        default_kraken_tvl_m=31.4,
        default_budget_a=101.1,
        default_budget_b=0.0,
        default_rmax_a=5.00,
        default_rmax_b=1.00,
        default_base_apy=1.0,
        floor_apy=None,
        floor_label=None,
        vault_address=None,  # discovered at runtime
        base_apy_config={
            "name": "Morpho RLUSD",
            "protocol": "morpho",
            "asset": "RLUSD",
            "chain": "Ethereum",
        },
    ),
    VenueConfig(
        key="euler_rlusd",
        label="Euler RLUSD",
        asset="RLUSD",
        protocol="euler",
        default_total_tvl_m=230.0,
        default_kraken_tvl_m=23.0,
        default_budget_a=190.6,
        default_budget_b=25.0,
        default_rmax_a=5.70,
        default_rmax_b=0.80,
        default_base_apy=1.5,
        floor_apy=None,
        floor_label=None,
        vault_address=EULER_VAULT_ADDRS["euler_rlusd"],
        base_apy_config={
            "name": "Euler Sentora RLUSD",
            "protocol": "euler",
            "asset": "RLUSD",
            "chain": "Ethereum",
            "defillama_project": "euler-v2",
        },
    ),
    VenueConfig(
        key="morpho_pyusd",
        label="Morpho PYUSD",
        asset="PYUSD",
        protocol="morpho",
        default_total_tvl_m=414.5,
        default_kraken_tvl_m=28.8,
        default_budget_a=280.0,
        default_budget_b=10.0,
        default_rmax_a=5.40,
        default_rmax_b=1.00,
        default_base_apy=1.26,
        floor_apy=4.7,
        floor_label="Looper floor (all-in ≥ 4.7%)",
        vault_address=MORPHO_VAULT_ADDRS["morpho_pyusd"],
        base_apy_config={
            "name": "Morpho PYUSD",
            "protocol": "morpho",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "vault_address": MORPHO_VAULT_ADDRS["morpho_pyusd"],
        },
    ),
    VenueConfig(
        key="euler_pyusd",
        label="Euler PYUSD",
        asset="PYUSD",
        protocol="euler",
        default_total_tvl_m=280.0,
        default_kraken_tvl_m=14.0,
        default_budget_a=212.1,
        default_budget_b=33.0,
        default_rmax_a=6.20,
        default_rmax_b=1.00,
        default_base_apy=1.5,
        floor_apy=None,
        floor_label=None,
        vault_address=EULER_VAULT_ADDRS["euler_pyusd"],
        base_apy_config={
            "name": "Euler Sentora PYUSD",
            "protocol": "euler",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "defillama_project": "euler-v2",
        },
    ),
    VenueConfig(
        key="curve_rlusd",
        label="Curve RLUSD-USDC",
        asset="RLUSD",
        protocol="curve",
        default_total_tvl_m=75.0,
        default_kraken_tvl_m=0.000001,  # $1 placeholder — no Kraken address yet
        default_budget_a=80.0,
        default_budget_b=5.0,
        default_rmax_a=50.00,  # pure float, no cap on Campaign A
        default_rmax_b=2.00,
        default_base_apy=0.5,
        floor_apy=None,
        floor_label=None,
        vault_address=CURVE_POOL_ADDRS["curve_rlusd"],
        base_apy_config={
            "name": "Curve RLUSD-USDC",
            "protocol": "curve",
            "asset": "RLUSD",
            "chain": "Ethereum",
            "pool_id_contains": "e91e23af",
        },
    ),
    VenueConfig(
        key="curve_pyusd",
        label="Curve PYUSD-USDC",
        asset="PYUSD",
        protocol="curve",
        default_total_tvl_m=30.0,
        default_kraken_tvl_m=0.000001,  # no Kraken LP balance yet
        default_budget_a=40.0,
        default_budget_b=5.0,
        default_rmax_a=50.00,  # pure float, no cap on Campaign A
        default_rmax_b=1.00,
        default_base_apy=0.3,
        floor_apy=None,
        floor_label=None,
        vault_address=CURVE_POOL_ADDRS["curve_pyusd"],
        base_apy_config={
            "name": "Curve PYUSD-USDC",
            "protocol": "curve",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "pool_id_contains": "14681aee",
        },
    ),
]

KRAKEN_MIN_MERKL = 3.5   # % / year floor on Kraken's combined Merkl rate

# ============================================================================
# PURE-MATH HELPERS
# ============================================================================

def compute_rates(
    non_kraken_tvl_m: float,
    kraken_tvl_m: float,
    budget_a: float,       # $K / week
    budget_b: float,       # $K / week
    rmax_a: float,         # % / year
    rmax_b: float,         # % / year
) -> dict:
    """
    Compute all campaign rates/spend for a given state.

    All TVL inputs are in $M.  Budgets are in $K/week.
    Rates are in % / year.

    Returns dict with:
      rate_a, rate_b, kraken_merkl, non_kraken_merkl,
      spend_a, spend_b, total_spend, unspent,
      total_tvl_m, at_cap_a, at_cap_b, breakeven_b_m
    """
    total_tvl_m = non_kraken_tvl_m + kraken_tvl_m

    annual_a = budget_a * 52  # $K / year
    annual_b = budget_b * 52

    # ── Campaign A ──
    if total_tvl_m > 0:
        raw_rate_a = annual_a / (total_tvl_m * 1_000) * 100   # %
        rate_a = min(rmax_a, raw_rate_a)
        at_cap_a = rate_a >= rmax_a - 1e-9
        # actual spend per week
        if at_cap_a:
            spend_a = (rmax_a / 100) * total_tvl_m * 1_000 / 52
        else:
            spend_a = budget_a
    else:
        rate_a, spend_a, at_cap_a = 0.0, 0.0, False

    # ── Campaign B ──
    if kraken_tvl_m > 0 and budget_b > 0:
        raw_rate_b = annual_b / (kraken_tvl_m * 1_000) * 100  # %
        rate_b = min(rmax_b, raw_rate_b)
        at_cap_b = rate_b >= rmax_b - 1e-9
        spend_b = (rmax_b / 100) * kraken_tvl_m * 1_000 / 52 if at_cap_b else budget_b
    else:
        rate_b, spend_b, at_cap_b = 0.0, 0.0, False

    kraken_merkl = rate_a + rate_b
    non_kraken_merkl = rate_a

    total_spend = spend_a + spend_b
    total_budget = budget_a + budget_b
    unspent = total_budget - total_spend

    # Breakeven B: Kraken TVL at which Campaign B rate starts compressing
    breakeven_b_m = (annual_b / (rmax_b / 100) / 1_000) if (budget_b > 0 and rmax_b > 0) else 0.0

    return {
        "total_tvl_m": total_tvl_m,
        "rate_a": rate_a,
        "rate_b": rate_b,
        "kraken_merkl": kraken_merkl,
        "non_kraken_merkl": non_kraken_merkl,
        "spend_a": spend_a,
        "spend_b": spend_b,
        "total_spend": total_spend,
        "total_budget": total_budget,
        "unspent": unspent,
        "at_cap_a": at_cap_a,
        "at_cap_b": at_cap_b,
        "breakeven_b_m": breakeven_b_m,
    }


def sweep_kraken_tvl(
    non_kraken_tvl_m: float,
    budget_a: float,
    budget_b: float,
    rmax_a: float,
    rmax_b: float,
    max_kraken_m: float = 500.0,
    n: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep Kraken TVL from 0 to max_kraken_m.

    Returns (k_arr, rate_a_arr, rate_b_arr, kraken_arr, non_kraken_arr).
    """
    k_arr = np.linspace(0.01, max_kraken_m, n)
    rate_a_arr = np.zeros(n)
    rate_b_arr = np.zeros(n)
    kraken_arr = np.zeros(n)
    non_kraken_arr = np.zeros(n)
    spend_arr = np.zeros(n)

    for i, k in enumerate(k_arr):
        c = compute_rates(non_kraken_tvl_m, k, budget_a, budget_b, rmax_a, rmax_b)
        rate_a_arr[i] = c["rate_a"]
        rate_b_arr[i] = c["rate_b"]
        kraken_arr[i] = c["kraken_merkl"]
        non_kraken_arr[i] = c["non_kraken_merkl"]
        spend_arr[i] = c["total_spend"]

    return k_arr, rate_a_arr, rate_b_arr, kraken_arr, non_kraken_arr, spend_arr


def find_max_kraken_for_floor(
    non_kraken_tvl_m: float,
    budget_a: float,
    budget_b: float,
    rmax_a: float,
    rmax_b: float,
    floor_rate: float = KRAKEN_MIN_MERKL,
    hi: float = 5_000.0,
) -> float:
    """
    Binary search: maximum Kraken TVL such that Kraken Merkl rate >= floor_rate.
    Returns $M.  Returns 0 if current rate is already below floor.
    """
    lo, hi_s = 0.001, hi
    for _ in range(80):
        mid = (lo + hi_s) / 2
        r = compute_rates(non_kraken_tvl_m, mid, budget_a, budget_b, rmax_a, rmax_b)
        if r["kraken_merkl"] >= floor_rate:
            lo = mid
        else:
            hi_s = mid
    return lo


def budget_for_target_rate(
    target_rate: float,
    tvl_m: float,
    rmax: float,
) -> float:
    """
    Compute weekly budget ($K) needed to achieve `target_rate` (% / yr)
    over `tvl_m` ($M TVL), capped at `rmax`.

    Budget = effective_rate × TVL_M × 1000 / 52
    If target_rate > rmax, the rate is capped regardless of budget.
    """
    effective = min(target_rate, rmax)
    return (effective / 100) * tvl_m * 1_000 / 52


# ============================================================================
# LIVE DATA FETCHING
# ============================================================================

def _total_assets_onchain(address: str, decimals: int = 6) -> float:
    """
    Call ERC4626 totalAssets() on a contract.  Returns USD value (1:1 stablecoin).
    Selector: 0x01e1d114 (keccak256("totalAssets()")[:4])
    """
    rpc_url = get_eth_rpc_url()
    try:
        raw = _eth_call(rpc_url, address, "0x01e1d114")
        raw_val = _decode_uint256(raw)
        return raw_val / (10 ** decimals)
    except Exception as e:
        st.warning(f"On-chain query failed for {address[:10]}…: {e}")
        return 0.0


def _balance_of_onchain(vault: str, holder: str, decimals: int = 6) -> float:
    """
    Call ERC20/ERC4626 balanceOf(holder) on vault, returns units (not shares).
    Then calls convertToAssets(shares) to get underlying amount.
    """
    rpc_url = get_eth_rpc_url()
    # selector balanceOf(address): 0x70a08231 + 32-byte padded address
    encoded = "0x70a08231" + holder[2:].lower().zfill(64)
    try:
        shares_raw = _decode_uint256(_eth_call(rpc_url, vault, encoded))
        if shares_raw == 0:
            return 0.0
        # convertToAssets(uint256 shares): 0x07a2d13a
        assets_raw = _decode_uint256(
            _eth_call(rpc_url, vault, "0x07a2d13a" + hex(shares_raw)[2:].zfill(64))
        )
        return assets_raw / (10 ** decimals)
    except Exception as e:
        st.warning(f"balanceOf/convertToAssets failed ({vault[:10]}/{holder[:10]}): {e}")
        return 0.0


def _curve_lp_value_onchain(pool: str, holder: str) -> float:
    """
    Get USD value of Curve LP tokens held by `holder` in `pool`.
    Uses balanceOf(holder) * get_virtual_price() / 1e36.
    LP tokens are 18 decimals; virtual_price is 18 decimals.
    """
    rpc_url = get_eth_rpc_url()
    try:
        bal = _decode_uint256(
            _eth_call(rpc_url, pool, "0x70a08231" + holder[2:].lower().zfill(64))
        )
        if bal == 0:
            return 0.0
        vp = _decode_uint256(_eth_call(rpc_url, pool, "0xbb7b8b80"))  # get_virtual_price()
        return bal * vp / 1e36
    except Exception:
        return 0.0


def _get_token_decimals(token_address: str) -> int:
    """Call ERC20 decimals() on-chain.  Falls back to 6 on error."""
    rpc_url = get_eth_rpc_url()
    try:
        return _decode_uint256(_eth_call(rpc_url, token_address, "0x313ce567"))
    except Exception:
        return 6


def _morpho_gql_vault_by_asset(asset_address: str) -> list[dict]:
    """Search Morpho GQL for MetaMorpho vaults by asset address, ordered by TVL desc."""
    query = """
    query($asset: String!) {
      vaults(
        where: { assetAddress_in: [$asset] }
        orderBy: totalAssets
        orderDirection: desc
        first: 5
      ) {
        items {
          address
          name
          asset { decimals }
          state { totalAssets }
        }
      }
    }
    """
    resp = requests.post(
        MORPHO_GRAPHQL_URL,
        json={"query": query, "variables": {"asset": asset_address.lower()}},
        headers={"Content-Type": "application/json"},
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()
    return body.get("data", {}).get("vaults", {}).get("items", [])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_kraken_tvls() -> dict[str, float]:
    """
    For each Kraken strategy address, call balanceOf(strategy) on the underlying
    vault then convertToAssets(shares).  Strategy contracts are Boring Vault
    strategies that do NOT implement ERC4626 totalAssets() themselves.
    Returns {venue_key: tvl_usd}.  Sums multi-address venues.
    """
    # Underlying vault address and asset decimals per venue key
    rlusd_decimals = RLUSD_DECIMALS

    # Resolve Morpho RLUSD vault address dynamically if needed
    morpho_rlusd_vault = MORPHO_VAULT_ADDRS.get("morpho_rlusd")
    if not morpho_rlusd_vault:
        try:
            vaults = _morpho_gql_vault_by_asset(RLUSD_ADDRESS)
            if vaults:
                morpho_rlusd_vault = vaults[0]["address"]
        except Exception:
            pass

    vault_lookup: dict[str, tuple[str | None, int]] = {
        "morpho_rlusd": (morpho_rlusd_vault, rlusd_decimals),
        "morpho_pyusd": (MORPHO_VAULT_ADDRS["morpho_pyusd"], STABLECOIN_DECIMALS),
        "euler_rlusd":  (EULER_VAULT_ADDRS["euler_rlusd"],  rlusd_decimals),
        "euler_pyusd":  (EULER_VAULT_ADDRS["euler_pyusd"],  STABLECOIN_DECIMALS),
    }

    results: dict[str, float] = {}
    for key, addrs in KRAKEN_STRATEGY_ADDRS.items():
        underlying_vault, decimals = vault_lookup.get(key, (None, STABLECOIN_DECIMALS))
        if not underlying_vault:
            results[key] = 0.0
            continue
        results[key] = sum(
            _balance_of_onchain(underlying_vault, strategy, decimals)
            for strategy in addrs
        )

    # ── Curve Kraken LP positions — LP balance × virtual_price ──
    for key, holder in CURVE_KRAKEN_LP_ADDRS.items():
        if not holder:
            continue
        pool = CURVE_POOL_ADDRS.get(key)
        if pool:
            results[key] = _curve_lp_value_onchain(pool, holder)

    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_vault_tvls() -> dict[str, float]:
    """
    Fetch total vault TVLs ($USD) for each venue.
    Morpho: GraphQL totalAssets on vault contract (asset decimals from GQL).
    Euler: fetch_euler_vault_data which fetches asset decimals on-chain.
    Returns raw USD values (not $M); callers divide by 1e6 for $M.
    """
    results: dict[str, float] = {}

    # ── Morpho PYUSD ──
    # total_assets from fetch_morpho_base_apy is already in USD — do NOT multiply by 1e6
    try:
        data = fetch_morpho_base_apy(
            MORPHO_VAULT_ADDRS["morpho_pyusd"], chain_id=1, decimals=6
        )
        results["morpho_pyusd"] = data.details.get("total_assets", 0.0)
    except Exception:
        results["morpho_pyusd"] = 0.0

    # ── Morpho RLUSD — discover vault dynamically; use GQL asset decimals ──
    try:
        vaults = _morpho_gql_vault_by_asset(RLUSD_ADDRESS)
        if vaults:
            v = vaults[0]
            raw_assets = int(v["state"]["totalAssets"] or 0)
            asset_decimals = (v.get("asset") or {}).get("decimals") or RLUSD_DECIMALS
            results["morpho_rlusd"] = raw_assets / (10 ** asset_decimals)
            MORPHO_VAULT_ADDRS["morpho_rlusd"] = v["address"]
        else:
            results["morpho_rlusd"] = 0.0
    except Exception:
        results["morpho_rlusd"] = 0.0

    # ── Euler RLUSD / PYUSD — fetch_euler_vault_data fetches asset decimals on-chain ──
    for key, addr in EULER_VAULT_ADDRS.items():
        try:
            asset_sym = key.split("_")[1].upper()
            vdata = fetch_euler_vault_data(addr, asset_sym)
            results[key] = vdata.total_supply_usd
        except Exception:
            results[key] = 0.0

    # ── Curve RLUSD-USDC / PYUSD-USDC — DeFiLlama yields API ──
    try:
        dl_resp = requests.get("https://yields.llama.fi/pools", timeout=30)
        dl_resp.raise_for_status()
        dl_pools = {p["pool"]: p for p in dl_resp.json().get("data", [])}
    except Exception:
        dl_pools = {}

    for key, pool_id in CURVE_DEFILLAMA_IDS.items():
        pool = dl_pools.get(pool_id)
        if pool:
            results[key] = pool.get("tvlUsd", 0.0)
        else:
            results[key] = 0.0

    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_base_apys() -> dict[str, float]:
    """Returns {venue_key: base_apy_pct}."""
    configs = [v.base_apy_config for v in VENUES]
    try:
        raw: dict[str, BaseAPYResult] = fetch_all_base_apys(configs)
        return {
            v.key: raw[v.base_apy_config["name"]].base_apy_pct
            for v in VENUES
            if v.base_apy_config["name"] in raw
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_euler_caps() -> dict[str, dict]:
    """
    Fetch Euler vault supply cap status via ERC4626 maxDeposit(address).

    These Sentora/Euler Earn vaults don't expose caps() directly.
    Instead we use:
      - totalAssets()  → current vault TVL (USD)
      - maxDeposit(addr) → remaining deposit capacity (0 = at cap)
      - Inferred supply cap = totalAssets + maxDeposit

    Returns {venue_key: {
        "total_supply_usd": float,   # current TVL
        "remaining_usd": float,      # room before cap (0 = at cap)
        "supply_cap_usd": float,     # inferred cap (current + remaining)
        "at_cap": bool,              # True if no more deposits possible
    }}.
    """
    ASSET_DECIMALS = {"euler_rlusd": RLUSD_DECIMALS, "euler_pyusd": STABLECOIN_DECIMALS}
    MAX_DEPOSIT_SEL = "0x402d267d"  # maxDeposit(address)
    ADDR_ONE_PADDED = "0" * 63 + "1"  # address(1)
    TOTAL_ASSETS_SEL = "0x01e1d114"

    rpc_url = get_eth_rpc_url()
    results: dict[str, dict] = {}
    for key, addr in EULER_VAULT_ADDRS.items():
        dec = ASSET_DECIMALS.get(key, 6)
        try:
            total_raw = _decode_uint256(_eth_call(rpc_url, addr, TOTAL_ASSETS_SEL))
            total_usd = total_raw / (10 ** dec)

            remaining_raw = _decode_uint256(
                _eth_call(rpc_url, addr, MAX_DEPOSIT_SEL + ADDR_ONE_PADDED)
            )
            # type(uint256).max means unlimited
            if remaining_raw > 10 ** 30:
                remaining_usd = 0.0
                cap_usd = 0.0  # 0 = unlimited
                at_cap = False
            else:
                remaining_usd = remaining_raw / (10 ** dec)
                cap_usd = total_usd + remaining_usd
                at_cap = remaining_usd == 0.0

            results[key] = {
                "total_supply_usd": total_usd,
                "remaining_usd": remaining_usd,
                "supply_cap_usd": cap_usd,
                "at_cap": at_cap,
            }
        except Exception:
            results[key] = {
                "total_supply_usd": 0.0,
                "remaining_usd": 0.0,
                "supply_cap_usd": 0.0,
                "at_cap": False,
            }
    return results


# ============================================================================
# CHART BUILDER
# ============================================================================

def make_rate_chart(
    venue: VenueConfig,
    non_kraken_tvl_m: float,
    budget_a: float,
    budget_b: float,
    rmax_a: float,
    rmax_b: float,
    base_apy: float,
    current_kraken_m: float,
) -> "go.Figure":
    import plotly.graph_objects as go

    max_k = max(current_kraken_m * 4, 200.0)
    k, ra, rb, kraken_r, nk_r, spend = sweep_kraken_tvl(
        non_kraken_tvl_m, budget_a, budget_b, rmax_a, rmax_b, max_kraken_m=max_k
    )

    fig = go.Figure()

    # Band — Kraken all-in
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([k, k[::-1]]),
            y=np.concatenate([kraken_r + base_apy, (nk_r + base_apy)[::-1]]),
            fill="toself",
            fillcolor="rgba(249,115,22,0.10)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=True,
            name="Kraken all-in band",
        )
    )

    # Non-Kraken all-in line
    fig.add_trace(
        go.Scatter(
            x=k,
            y=nk_r + base_apy,
            mode="lines",
            line=dict(color="#6366f1", width=2),
            name="Non-Kraken all-in APY",
        )
    )

    # Non-Kraken Merkl line
    fig.add_trace(
        go.Scatter(
            x=k,
            y=nk_r,
            mode="lines",
            line=dict(color="#6366f1", width=1.5, dash="dot"),
            name="Non-Kraken Merkl only",
        )
    )

    # Kraken Merkl line
    fig.add_trace(
        go.Scatter(
            x=k,
            y=kraken_r,
            mode="lines",
            line=dict(color="#f97316", width=2),
            name="Kraken Merkl (A+B)",
        )
    )

    # Kraken all-in line
    fig.add_trace(
        go.Scatter(
            x=k,
            y=kraken_r + base_apy,
            mode="lines",
            line=dict(color="#f97316", width=2.5),
            name="Kraken all-in APY",
        )
    )

    # 3.5% Kraken floor
    fig.add_hline(
        y=KRAKEN_MIN_MERKL,
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        annotation_text=f"Kraken floor {KRAKEN_MIN_MERKL:.1f}%",
        annotation_position="top right",
    )

    # Looper floor (Morpho PYUSD only)
    if venue.floor_apy is not None:
        fig.add_hline(
            y=venue.floor_apy,
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            annotation_text=f"Looper floor {venue.floor_apy:.1f}%",
            annotation_position="bottom right",
        )

    # Current Kraken TVL marker
    fig.add_vline(
        x=current_kraken_m,
        line=dict(color="white", width=1.5, dash="dot"),
        annotation_text="Current",
        annotation_position="top left",
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"{venue.label} — Rate vs Kraken TVL",
        xaxis_title="Kraken TVL ($M)",
        yaxis_title="Rate (% / year)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=80, b=40, l=60, r=20),
        height=400,
        hovermode="x unified",
    )
    return fig


def make_spend_chart(
    venue: VenueConfig,
    non_kraken_tvl_m: float,
    budget_a: float,
    budget_b: float,
    rmax_a: float,
    rmax_b: float,
    current_kraken_m: float,
) -> "go.Figure":
    import plotly.graph_objects as go

    max_k = max(current_kraken_m * 4, 200.0)
    k, ra, rb, kraken_r, nk_r, spend = sweep_kraken_tvl(
        non_kraken_tvl_m, budget_a, budget_b, rmax_a, rmax_b, max_kraken_m=max_k
    )

    spend_a_arr = np.zeros(len(k))
    spend_b_arr = np.zeros(len(k))
    for i, ki in enumerate(k):
        c = compute_rates(non_kraken_tvl_m, ki, budget_a, budget_b, rmax_a, rmax_b)
        spend_a_arr[i] = c["spend_a"]
        spend_b_arr[i] = c["spend_b"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=spend_a_arr + spend_b_arr, name="Total spend", mode="lines", line=dict(color="white", width=2)))
    fig.add_trace(go.Scatter(x=k, y=spend_a_arr, name="Spend A", fill="tozeroy", mode="lines", line=dict(color="#6366f1", width=1.5), fillcolor="rgba(99,102,241,0.25)"))
    fig.add_trace(go.Scatter(x=k, y=spend_a_arr + spend_b_arr, name="Spend B (stack)", fill="tonexty", mode="lines", line=dict(color="#f97316", width=1.5), fillcolor="rgba(249,115,22,0.25)"))
    fig.add_hline(y=budget_a + budget_b, line=dict(color="#ef4444", dash="dash", width=1.5), annotation_text="Budget cap")
    fig.add_vline(x=current_kraken_m, line=dict(color="white", width=1.5, dash="dot"), annotation_text="Now")

    fig.update_layout(
        template="plotly_dark",
        title=f"{venue.label} — Weekly Spend vs Kraken TVL",
        xaxis_title="Kraken TVL ($M)",
        yaxis_title="Weekly spend ($K/week)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=80, b=40, l=60, r=20),
        height=320,
    )
    return fig


# ============================================================================
# STREAMLIT PAGE
# ============================================================================

st.set_page_config(
    page_title="Kraken Earn Split Incentive",
    page_icon="⚡",
    layout="wide",
)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("⚡ Kraken Earn Split Incentive Dashboard")
st.caption(
    "**Campaign A** — open to all depositors (Euler/Morpho vault) | "
    "**Campaign B** — whitelisted to Kraken Earn address(es) only. "
    "Kraken earns A + B. Non-Kraken earns A only. "
    "Kraken TVL dilutes _both_ campaigns."
)

# ── Live-data sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Live Data")
    refresh = st.button("🔄 Refresh on-chain data", use_container_width=True)
    if refresh:
        st.cache_data.clear()

    st.caption("Fetches Kraken strategy TVLs, vault TVLs, and base APYs. Cached 5 min.")
    st.divider()
    st.subheader("Kraken Earn Addresses")
    for key, addrs in KRAKEN_STRATEGY_ADDRS.items():
        label = next((v.label for v in VENUES if v.key == key), key)
        for a in addrs:
            st.caption(f"**{label}**: `{a[:10]}…`")
    for key, addr in CURVE_KRAKEN_LP_ADDRS.items():
        if addr:
            label = next((v.label for v in VENUES if v.key == key), key)
            st.caption(f"**{label}**: `{addr[:10]}…`")



min_kraken_merkl = KRAKEN_MIN_MERKL

# ── Fetch live data (silent, with fallbacks) ─────────────────────────────────
with st.spinner("Fetching live data…"):
    try:
        live_kraken = fetch_live_kraken_tvls()
    except Exception:
        live_kraken = {}
    try:
        live_vault = fetch_live_vault_tvls()
    except Exception:
        live_vault = {}
    try:
        live_base_apy = fetch_live_base_apys()
    except Exception:
        live_base_apy = {}
    try:
        euler_caps = fetch_euler_caps()
    except Exception:
        euler_caps = {}

def _resolve_default(venue: VenueConfig, live_dict: dict, default_val: float) -> float:
    val = live_dict.get(venue.key, 0.0)
    return val / 1e6 if val > 0 else default_val  # live_vault comes in raw USD


def _format_cap(cap_info: dict | None) -> str:
    """Format Euler supply cap for the summary table."""
    if not cap_info:
        return "—"
    if cap_info.get("supply_cap_usd", 0) == 0:
        return "∞ (unlimited)" if cap_info.get("total_supply_usd", 0) > 0 else "—"
    cap_m = cap_info["supply_cap_usd"] / 1e6
    cur_m = cap_info["total_supply_usd"] / 1e6
    pct = cur_m / cap_m * 100 if cap_m > 0 else 0
    if cap_info.get("at_cap"):
        return f"⚠ AT CAP ${cap_m:,.0f}M"
    if pct >= 90:
        return f"⚠ ${cap_m:,.0f}M ({pct:.0f}%)"
    return f"${cap_m:,.0f}M ({pct:.0f}%)"


# ── Cross-venue summary table ─────────────────────────────────────────────────
with st.expander("📊 Cross-venue summary", expanded=True):
    summary_rows = []
    for v in VENUES:
        k_tvl = live_kraken.get(v.key, 0.0) / 1e6 if live_kraken.get(v.key, 0.0) > 0 else v.default_kraken_tvl_m
        t_tvl = live_vault.get(v.key, 0.0) / 1e6 if live_vault.get(v.key, 0.0) > 0 else v.default_total_tvl_m
        nk_tvl = max(t_tvl - k_tvl, 0.0)
        base = live_base_apy.get(v.key, v.default_base_apy)
        c = compute_rates(nk_tvl, k_tvl, v.default_budget_a, v.default_budget_b, v.default_rmax_a, v.default_rmax_b)
        max_k = find_max_kraken_for_floor(nk_tvl, v.default_budget_a, v.default_budget_b, v.default_rmax_a, v.default_rmax_b, min_kraken_merkl)
        hdroom = max(0.0, max_k - k_tvl)
        summary_rows.append({
            "Venue": v.label,
            "Total TVL ($M)": f"${t_tvl:.1f}M",
            "Kraken TVL ($M)": f"${k_tvl:.1f}M",
            "Non-Kraken Merkl": f"{c['non_kraken_merkl']:.2f}%",
            "Non-Kraken all-in": f"{c['non_kraken_merkl'] + base:.2f}%",
            "Kraken Merkl (A+B)": f"{c['kraken_merkl']:.2f}%",
            "Kraken all-in": f"{c['kraken_merkl'] + base:.2f}%",
            "Total spend ($K/wk)": f"${c['total_spend']:.1f}K",
            "Unspent ($K/wk)": f"${c['unspent']:.1f}K",
            f"Kraken headroom (>{min_kraken_merkl:.1f}% Merkl)": f"+${hdroom:.1f}M",
            "Supply Cap": _format_cap(euler_caps.get(v.key)),
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# PER-VENUE PANELS
# ============================================================================

for venue in VENUES:
    with st.expander(f"**{venue.label}**", expanded=True):

        # ── Resolve live defaults ─────────────────────────────────────────
        live_k_usd = live_kraken.get(venue.key, 0.0)
        live_t_usd = live_vault.get(venue.key, 0.0)
        init_kraken_m = live_k_usd / 1e6 if live_k_usd > 0 else venue.default_kraken_tvl_m
        init_total_m  = live_t_usd / 1e6 if live_t_usd > 0 else venue.default_total_tvl_m
        init_nk_m = max(init_total_m - init_kraken_m, 0.0)
        init_base_apy = live_base_apy.get(venue.key, venue.default_base_apy)

        k = venue.key

        # ── Build state keys ──────────────────────────────────────────────
        if f"{k}_init" not in st.session_state:
            st.session_state[f"{k}_nk_tvl"]   = float(init_nk_m)
            st.session_state[f"{k}_k_tvl"]    = float(init_kraken_m)
            st.session_state[f"{k}_budg_a"]   = float(venue.default_budget_a)
            st.session_state[f"{k}_budg_b"]   = float(venue.default_budget_b)
            st.session_state[f"{k}_rmax_a"]   = float(venue.default_rmax_a)
            st.session_state[f"{k}_rmax_b"]   = float(venue.default_rmax_b)
            st.session_state[f"{k}_base_apy"] = float(init_base_apy)
            st.session_state[f"{k}_mode_a"]   = "Set budget ($K/wk)"
            st.session_state[f"{k}_mode_b"]   = "Set budget ($K/wk)"
            st.session_state[f"{k}_init"]     = True

        # ── Row 0: live data status bar ───────────────────────────────────
        data_cols = st.columns([3, 3, 3, 1])
        if live_k_usd > 0:
            data_cols[0].success(f"Live Kraken TVL: ${live_k_usd/1e6:.2f}M", icon="🟢")
        else:
            data_cols[0].info(f"Default Kraken TVL: ${venue.default_kraken_tvl_m:.1f}M", icon="📌")
        if live_t_usd > 0:
            data_cols[1].success(f"Live vault TVL: ${live_t_usd/1e6:.2f}M", icon="🟢")
        else:
            data_cols[1].info(f"Default vault TVL: ${venue.default_total_tvl_m:.1f}M", icon="📌")
        if venue.key in live_base_apy:
            data_cols[2].success(f"Live base APY: {live_base_apy[venue.key]:.2f}%", icon="🟢")
        else:
            data_cols[2].info(f"Default base APY: {venue.default_base_apy:.2f}%", icon="📌")

        # ── Euler supply cap banner ───────────────────────────────────────
        cap_info = euler_caps.get(venue.key)
        if cap_info and cap_info["supply_cap_usd"] > 0:
            cap_m = cap_info["supply_cap_usd"] / 1e6
            current_m = (cap_info["total_supply_usd"] / 1e6) if cap_info["total_supply_usd"] > 0 else init_total_m
            headroom_cap = cap_m - current_m
            utilisation_pct = (current_m / cap_m * 100) if cap_m > 0 else 0
            if headroom_cap <= 0:
                st.error(
                    f"🚫 **Euler supply cap reached!** Cap: ${cap_m:,.1f}M — "
                    f"Current: ${current_m:,.1f}M ({utilisation_pct:.0f}% utilised). "
                    f"To facilitate growth, increase Euler cap by at least **${abs(headroom_cap) + 10:,.0f}M** "
                    f"to **${cap_m + abs(headroom_cap) + 10:,.0f}M**.",
                    icon="🔴",
                )
            elif headroom_cap < 20:
                st.warning(
                    f"⚠️ **Euler supply cap is tight.** Cap: ${cap_m:,.1f}M — "
                    f"Current: ${current_m:,.1f}M ({utilisation_pct:.0f}% utilised, "
                    f"${headroom_cap:,.1f}M remaining). "
                    f"To accommodate further growth, increase Euler cap to at least **${current_m + 50:,.0f}M**.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"Euler supply cap: ${cap_m:,.1f}M — "
                    f"Current: ${current_m:,.1f}M ({utilisation_pct:.0f}% utilised, "
                    f"${headroom_cap:,.1f}M headroom)",
                    icon="✅",
                )

        st.divider()

        # ── Row 1: Input columns ──────────────────────────────────────────
        col_tvl, col_a, col_b = st.columns([1.2, 1.4, 1.4])

        with col_tvl:
            st.markdown("##### TVL inputs ($M)")
            nk_tvl = st.slider(
                "Non-Kraken TVL ($M)",
                min_value=0.0, max_value=1_500.0, step=1.0,
                value=float(st.session_state[f"{k}_nk_tvl"]),
                key=f"{k}_nk_tvl_slider",
                help="TVL from non-Kraken depositors. This is Total TVL minus Kraken TVL.",
            )
            k_tvl = st.slider(
                "Kraken Earn TVL ($M)",
                min_value=0.0, max_value=1_000.0, step=1.0,
                value=float(st.session_state[f"{k}_k_tvl"]),
                key=f"{k}_k_tvl_slider",
                help="Kraken Earn strategy deposits. Dilutes both Campaign A (part of total TVL) and Campaign B (direct denominator).",
            )
            base_apy = st.number_input(
                "Base organic APY (%)",
                min_value=0.0, max_value=20.0, step=0.05, format="%.2f",
                value=float(st.session_state[f"{k}_base_apy"]),
                key=f"{k}_base_apy_input",
                help="Organic yield from borrowers, before any Merkl incentives.",
            )

        # ── Campaign A ────────────────────────────────────────────────────
        with col_a:
            st.markdown("##### Campaign A — all depositors")
            mode_a = st.radio(
                "Input mode",
                ["Set budget ($K/wk)", "Set target rate (%)"],
                horizontal=True,
                key=f"{k}_mode_a_radio",
                index=["Set budget ($K/wk)", "Set target rate (%)"].index(
                    st.session_state[f"{k}_mode_a"]
                ),
                help="Budget mode: set weekly spend directly. Target mode: set desired APR and compute required budget.",
            )
            st.session_state[f"{k}_mode_a"] = mode_a

            _rmax_a_max = max(15.0, float(st.session_state[f"{k}_rmax_a"]) * 1.5)
            rmax_a = st.slider(
                "Rmax A (cap, %/yr)",
                min_value=0.1, max_value=_rmax_a_max, step=0.1, format="%.1f",
                value=float(st.session_state[f"{k}_rmax_a"]),
                key=f"{k}_rmax_a_slider",
                help="Maximum incentive APR Campaign A will pay. Set very high for pure-float (no cap). Budget underspends when rate hits this cap.",
            )

            total_tvl = nk_tvl + k_tvl  # $M
            if mode_a == "Set budget ($K/wk)":
                budget_a = st.slider(
                    "Budget A ($K/wk)",
                    min_value=0.0, max_value=1_000.0, step=0.5, format="%.1f",
                    value=float(st.session_state[f"{k}_budg_a"]),
                    key=f"{k}_budg_a_slider",
                    help="Weekly budget for Campaign A distributed pro-rata to all depositors (Kraken + non-Kraken).",
                )
                # Show implied rate
                if total_tvl > 0:
                    impl_rate_a = min(rmax_a, budget_a * 52 / (total_tvl * 1_000) * 100)
                    st.caption(f"→ Implied rate A: **{impl_rate_a:.3f}%**")
            else:
                # Rate → budget
                target_rate_a = st.slider(
                    "Target rate A (%/yr)",
                    min_value=0.01, max_value=rmax_a, step=0.05, format="%.2f",
                    value=min(float(st.session_state[f"{k}_budg_a"] * 52 / (total_tvl * 1_000) * 100) if total_tvl > 0 else 1.0, rmax_a),
                    key=f"{k}_rate_a_target",
                    help="Desired Campaign A annual incentive rate. Budget is computed to achieve this rate at current TVL.",
                )
                budget_a = budget_for_target_rate(target_rate_a, total_tvl, rmax_a)
                st.caption(f"→ Required budget: **${budget_a:.1f}K/wk**")

        # ── Campaign B ────────────────────────────────────────────────────
        with col_b:
            st.markdown("##### Campaign B — Kraken Earn only")
            mode_b = st.radio(
                "Input mode",
                ["Set budget ($K/wk)", "Set target rate (%)"],
                horizontal=True,
                key=f"{k}_mode_b_radio",
                index=["Set budget ($K/wk)", "Set target rate (%)"].index(
                    st.session_state[f"{k}_mode_b"]
                ),
                help="Budget mode: set weekly spend directly. Target mode: set desired APR and compute required budget.",
            )
            st.session_state[f"{k}_mode_b"] = mode_b

            rmax_b = st.slider(
                "Rmax B (cap, %/yr)",
                min_value=0.0, max_value=10.0, step=0.1, format="%.1f",
                value=float(st.session_state[f"{k}_rmax_b"]),
                key=f"{k}_rmax_b_slider",
                help="Keep low (0.8–1.0%) so rate is stable over a wide range of Kraken TVL.",
            )

            if mode_b == "Set budget ($K/wk)":
                budget_b = st.slider(
                    "Budget B ($K/wk)",
                    min_value=0.0, max_value=200.0, step=0.5, format="%.1f",
                    value=float(st.session_state[f"{k}_budg_b"]),
                    key=f"{k}_budg_b_slider",
                    help="Weekly budget for Campaign B distributed only to whitelisted Kraken Earn address(es).",
                )
                if k_tvl > 0 and rmax_b > 0:
                    impl_rate_b = min(rmax_b, budget_b * 52 / (k_tvl * 1_000) * 100)
                    st.caption(f"→ Implied rate B: **{impl_rate_b:.3f}%**")
            else:
                target_rate_b = st.slider(
                    "Target rate B (%/yr)",
                    min_value=0.0, max_value=max(rmax_b, 0.01), step=0.05, format="%.2f",
                    value=min(
                        float(st.session_state[f"{k}_budg_b"] * 52 / (k_tvl * 1_000) * 100) if k_tvl > 0 else 0.5,
                        rmax_b if rmax_b > 0 else 1.0,
                    ),
                    key=f"{k}_rate_b_target",
                    help="Desired Campaign B annual incentive rate. Budget is computed to achieve this rate at current Kraken TVL.",
                )
                budget_b = budget_for_target_rate(target_rate_b, k_tvl, rmax_b)
                st.caption(f"→ Required budget: **${budget_b:.1f}K/wk**")

        # ── Compute current state ──────────────────────────────────────────
        c = compute_rates(nk_tvl, k_tvl, budget_a, budget_b, rmax_a, rmax_b)
        non_kraken_allin = c["non_kraken_merkl"] + base_apy
        kraken_allin = c["kraken_merkl"] + base_apy
        floor_breached = (venue.floor_apy is not None) and (non_kraken_allin < venue.floor_apy)

        breakeven_b_m = c["breakeven_b_m"]
        headroom_b = max(0.0, breakeven_b_m - k_tvl)
        max_kraken_for_floor = find_max_kraken_for_floor(
            nk_tvl, budget_a, budget_b, rmax_a, rmax_b, min_kraken_merkl
        )
        additional_before_3p5 = max(0.0, max_kraken_for_floor - k_tvl)

        st.divider()

        # ── Row 2: Key rate metrics ────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            color_nk = "inverse" if floor_breached else "normal"
            st.metric(
                "Non-Kraken Merkl",
                f"{c['non_kraken_merkl']:.3f}%",
                help="Campaign A effective rate for external depositors.",
            )
            st.metric(
                "Non-Kraken all-in APY",
                f"{non_kraken_allin:.2f}%",
                delta=f"{'⚠ BELOW FLOOR' if floor_breached else ''}",
                delta_color="inverse" if floor_breached else "off",
                help="Base APY + Merkl incentive rate for non-Kraken depositors.",
            )
            if floor_breached:
                st.error(f"Below {venue.floor_label or 'floor'} ({venue.floor_apy:.1f}%)!", icon="🔴")

        with m2:
            st.metric(
                "Kraken Merkl (A+B)",
                f"{c['kraken_merkl']:.3f}%",
                help="Combined effective incentive rate for Kraken Earn.",
            )
            st.metric(
                "Kraken all-in APY",
                f"{kraken_allin:.2f}%",
                help="Base APY + Campaign A + Campaign B combined incentive rate for Kraken Earn.",
            )

        with m3:
            st.metric("Campaign A rate", f"{c['rate_a']:.3f}%",
                      delta="at cap" if c["at_cap_a"] else "below cap",
                      delta_color="off",
                      help="Effective Campaign A incentive rate. 'At cap' means Rmax A has been reached and budget is underspent.")
            st.metric("Campaign B rate", f"{c['rate_b']:.3f}%",
                      delta="at cap" if c["at_cap_b"] else "below cap",
                      delta_color="off",
                      help="Effective Campaign B incentive rate (Kraken only). 'At cap' means Rmax B has been reached.")

        with m4:
            st.metric(
                "Total spend (A+B)",
                f"${c['total_spend']:.1f}K/wk",
                delta=f"-${c['unspent']:.1f}K unspent" if c["unspent"] > 0.5 else "fully deployed",
                delta_color="normal" if c["unspent"] > 0.5 else "off",
                help="Combined weekly spend across both campaigns. Unspent means rate cap was hit before budget was exhausted.",
            )
            st.metric(
                "Annualised cost",
                f"${(c['total_spend'] * 52 / 1_000):.2f}M/yr",
                help="Total spend extrapolated to annual cost (spend × 52 weeks).",
            )

        st.divider()

        # ── Row 3: Capacity metrics ────────────────────────────────────────
        cap1, cap2, cap3, cap4 = st.columns(4)

        cap1.metric(
            "Campaign B breakeven TVL",
            f"${breakeven_b_m:.1f}M",
            help="Kraken TVL at which Campaign B rate starts compressing below Rmax B.",
        )
        cap2.metric(
            "Headroom before B compresses",
            f"+${headroom_b:.1f}M",
            delta="⚠ tightening" if 5 < headroom_b <= 20 else ("⚠ at cap" if headroom_b <= 5 else ""),
            delta_color="off" if headroom_b > 20 else ("off" if headroom_b > 5 else "inverse"),
            help="How much more Kraken TVL can grow before Campaign B rate compresses below Rmax B.",
        )
        cap3.metric(
            f"Max Kraken TVL >{min_kraken_merkl:.1f}% Merkl",
            f"${max_kraken_for_floor:.1f}M",
            help=f"Maximum Kraken TVL before combined Merkl rate drops below {min_kraken_merkl:.1f}%.",
        )
        cap4.metric(
            "Additional Kraken capacity",
            f"+${additional_before_3p5:.1f}M",
            delta="⚠ moderate" if 20 < additional_before_3p5 <= 50 else ("⚠ limited" if additional_before_3p5 <= 20 else ""),
            delta_color="off" if additional_before_3p5 > 50 else ("off" if additional_before_3p5 > 20 else "inverse"),
            help=f"How much more Kraken TVL can be added before combined Merkl rate drops below {min_kraken_merkl:.1f}%.",
        )

        # ── Euler cap vs scenario TVL check ───────────────────────────────
        if cap_info and cap_info["supply_cap_usd"] > 0:
            cap_m = cap_info["supply_cap_usd"] / 1e6
            scenario_tvl = nk_tvl + k_tvl  # slider TVL ($M)
            if scenario_tvl > cap_m:
                needed = scenario_tvl - cap_m
                st.error(
                    f"🚫 Scenario TVL (${scenario_tvl:.0f}M) exceeds Euler supply cap (${cap_m:,.0f}M). "
                    f"**Increase Euler cap by ${needed:,.0f}M to ${scenario_tvl:,.0f}M** to support this scenario.",
                    icon="🔴",
                )
            elif (cap_m - scenario_tvl) < 20:
                st.warning(
                    f"⚠️ Scenario TVL (${scenario_tvl:.0f}M) is within ${cap_m - scenario_tvl:,.1f}M of "
                    f"Euler supply cap (${cap_m:,.0f}M). Consider increasing cap to **${scenario_tvl + 50:,.0f}M**.",
                    icon="⚠️",
                )

        # ── Row 4: Detailed breakdown ─────────────────────────────────────
        with st.expander("Detailed breakdown", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Campaign A**")
                st.markdown(
                    f"- Weekly budget: **${budget_a:.1f}K**\n"
                    f"- Annual budget: **${budget_a * 52 / 1_000:.2f}M**\n"
                    f"- Total TVL (A denominator): **${c['total_tvl_m']:.1f}M**\n"
                    f"- Raw rate (no cap): **{budget_a * 52 / (c['total_tvl_m'] * 1_000) * 100:.3f}%**\n"
                    f"- Rmax A cap: **{rmax_a:.2f}%**\n"
                    f"- Effective rate A: **{c['rate_a']:.3f}%** {'(capped)' if c['at_cap_a'] else '(uncapped)'}\n"
                    f"- Actual spend A: **${c['spend_a']:.2f}K/wk** (of ${budget_a:.1f}K budget)"
                )
            with d2:
                st.markdown("**Campaign B**")
                st.markdown(
                    f"- Weekly budget: **${budget_b:.1f}K**\n"
                    f"- Annual budget: **${budget_b * 52 / 1_000:.2f}M**\n"
                    f"- Kraken TVL (B denominator): **${k_tvl:.1f}M**\n"
                    f"- Raw rate (no cap): **{budget_b * 52 / (k_tvl * 1_000) * 100 if k_tvl > 0 else 0:.3f}%**\n"
                    f"- Rmax B cap: **{rmax_b:.2f}%**\n"
                    f"- Effective rate B: **{c['rate_b']:.3f}%** {'(capped)' if c['at_cap_b'] else '(uncapped)'}\n"
                    f"- Actual spend B: **${c['spend_b']:.2f}K/wk** (of ${budget_b:.1f}K budget)\n"
                    f"- Breakeven Kraken TVL: **${breakeven_b_m:.1f}M** "
                    f"(BudgetB × 52 / Rmax B = ${budget_b:.1f}K × 52 / {rmax_b:.2f}%)"
                )

        # ── Row 5: Scenario explorer ──────────────────────────────────────
        with st.expander("📈 Rate curve — Kraken TVL sweep", expanded=True):
            chart_tabs = st.tabs(["Rate vs Kraken TVL", "Spend vs Kraken TVL"])
            with chart_tabs[0]:
                try:
                    fig = make_rate_chart(
                        venue, nk_tvl, budget_a, budget_b, rmax_a, rmax_b, base_apy, k_tvl
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart error: {e}")
            with chart_tabs[1]:
                try:
                    fig2 = make_spend_chart(
                        venue, nk_tvl, budget_a, budget_b, rmax_a, rmax_b, k_tvl
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart error: {e}")

        # ── Row 6: What-if Kraken TVL growth table ────────────────────────
        with st.expander("📋 What-if: Kraken TVL growth scenarios", expanded=False):
            steps = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500]
            rows = []
            for delta_k in steps:
                k_test = k_tvl + delta_k
                ct = compute_rates(nk_tvl, k_test, budget_a, budget_b, rmax_a, rmax_b)
                rows.append({
                    "+Δ Kraken ($M)": f"+${delta_k}M",
                    "Kraken TVL ($M)": f"${k_test:.0f}M",
                    "Rate A (%)": f"{ct['rate_a']:.3f}%",
                    "Rate B (%)": f"{ct['rate_b']:.3f}%",
                    "Kraken Merkl": f"{ct['kraken_merkl']:.3f}%",
                    "Kraken all-in": f"{ct['kraken_merkl'] + base_apy:.2f}%",
                    "Non-Kraken Merkl": f"{ct['non_kraken_merkl']:.3f}%",
                    "Non-Kraken all-in": f"{ct['non_kraken_merkl'] + base_apy:.2f}%",
                    "Spend A ($K/wk)": f"${ct['spend_a']:.1f}",
                    "Spend B ($K/wk)": f"${ct['spend_b']:.1f}",
                    "Total Spend ($K/wk)": f"${ct['total_spend']:.1f}",
                    "B at cap?": "✓" if ct["at_cap_b"] else "",
                    f"Kraken >{min_kraken_merkl:.1f}%?": "✓" if ct["kraken_merkl"] >= min_kraken_merkl else "⚠",
                })
            import pandas as pd
            df = pd.DataFrame(rows)
            # Highlight rows where Kraken drops below floor
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Row 7: Budget re-allocation helper ─────────────────────────────
        with st.expander("💡 Budget re-allocation calculator", expanded=False):
            st.markdown(
                "Redirect budget from Campaign A to Campaign B while keeping "
                "total spend flat. Shows the impact of each shift."
            )
            total_budget_fixed = budget_a + budget_b
            shift_cols = st.columns([2, 3])
            with shift_cols[0]:
                b_new = st.slider(
                    "Budget B (after shift) ($K/wk)",
                    min_value=0.0, max_value=float(total_budget_fixed),
                    step=0.5, value=float(budget_b),
                    key=f"{k}_realloc_b",
                    help="Increase B, decrease A proportionally. Total held constant.",
                )
            a_new = total_budget_fixed - b_new
            with shift_cols[1]:
                c_new = compute_rates(nk_tvl, k_tvl, a_new, b_new, rmax_a, rmax_b)
                delta_kraken = c_new["kraken_merkl"] - c["kraken_merkl"]
                delta_nk = c_new["non_kraken_merkl"] - c["non_kraken_merkl"]
                ra1, ra2, ra3, ra4 = st.columns(4)
                ra1.metric("New Budget A", f"${a_new:.1f}K/wk", delta=f"{a_new - budget_a:+.1f}K", delta_color="inverse", help="Adjusted Campaign A weekly budget after reallocation.")
                ra2.metric("New Budget B", f"${b_new:.1f}K/wk", delta=f"{b_new - budget_b:+.1f}K", delta_color="normal", help="Adjusted Campaign B weekly budget after reallocation.")
                ra3.metric("Kraken Merkl change", f"{c_new['kraken_merkl']:.3f}%", delta=f"{delta_kraken:+.3f}%", delta_color="normal", help="Change in Kraken's combined incentive rate (A+B) from reallocation.")
                ra4.metric("Non-Kraken Merkl change", f"{c_new['non_kraken_merkl']:.3f}%", delta=f"{delta_nk:+.3f}%", delta_color="inverse" if delta_nk < 0 else "normal", help="Change in non-Kraken incentive rate (A only) from reallocation.")

        st.divider()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "All rates are % APY.  TVL inputs in $M.  Budget inputs in $K/week.  "
    "Base APY is organic yield from borrowers before incentives.  "
    "Kraken TVL dilutes **both** Campaign A (via total TVL denominator) "
    "and Campaign B (directly, as its sole denominator).  "
    "On-chain data cached 5 minutes.  Updated: " + time.strftime("%Y-%m-%d %H:%M UTC")
)

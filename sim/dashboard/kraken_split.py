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
from campaign.evm_data import _eth_call, _decode_uint256, get_eth_rpc_url

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

RLUSD_ADDRESS = "0x8292Bb45bf1Ee4d140127049757C0C38e47a8A75"
PYUSD_ADDRESS = "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8"
STABLECOIN_DECIMALS = 6

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
    On-chain totalAssets() for each strategy contract.
    Returns {venue_key: tvl_usd}.  Sums multi-address venues.
    """
    results: dict[str, float] = {}
    for key, addrs in KRAKEN_STRATEGY_ADDRS.items():
        total = sum(_total_assets_onchain(a, STABLECOIN_DECIMALS) for a in addrs)
        results[key] = total
    return results


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_vault_tvls() -> dict[str, float]:
    """
    Fetch total vault TVLs ($USD) for each venue.
    Morpho: GraphQL totalAssets on vault contract.
    Euler: ERC4626 totalAssets() on vault address.
    """
    results: dict[str, float] = {}

    # ── Morpho PYUSD ──
    try:
        data = fetch_morpho_base_apy(
            MORPHO_VAULT_ADDRS["morpho_pyusd"], chain_id=1, decimals=6
        )
        results["morpho_pyusd"] = data.details.get("total_assets", 0.0) * 1e6
    except Exception:
        results["morpho_pyusd"] = 0.0

    # ── Morpho RLUSD — discover vault dynamically ──
    try:
        vaults = _morpho_gql_vault_by_asset(RLUSD_ADDRESS)
        if vaults:
            # Largest TVL vault
            v = vaults[0]
            raw_assets = int(v["state"]["totalAssets"] or 0)
            results["morpho_rlusd"] = raw_assets / (10 ** STABLECOIN_DECIMALS)
            MORPHO_VAULT_ADDRS["morpho_rlusd"] = v["address"]
        else:
            results["morpho_rlusd"] = 0.0
    except Exception:
        results["morpho_rlusd"] = 0.0

    # ── Euler RLUSD / PYUSD ──
    for key, addr in EULER_VAULT_ADDRS.items():
        try:
            results[key] = _total_assets_onchain(addr, STABLECOIN_DECIMALS)
        except Exception:
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

    st.divider()
    st.subheader("Global settings")
    min_kraken_merkl = st.number_input(
        "Min Kraken Merkl floor (%)", value=KRAKEN_MIN_MERKL, min_value=0.0, max_value=10.0, step=0.1,
        help="Alert when Kraken combined Merkl rate drops below this.",
    )

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

def _resolve_default(venue: VenueConfig, live_dict: dict, default_val: float) -> float:
    val = live_dict.get(venue.key, 0.0)
    return val / 1e6 if val > 0 else default_val  # live_vault comes in raw USD


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
            )
            k_tvl = st.slider(
                "Kraken Earn TVL ($M)",
                min_value=0.0, max_value=1_000.0, step=1.0,
                value=float(st.session_state[f"{k}_k_tvl"]),
                key=f"{k}_k_tvl_slider",
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
            )
            st.session_state[f"{k}_mode_a"] = mode_a

            rmax_a = st.slider(
                "Rmax A (cap, %/yr)",
                min_value=0.1, max_value=15.0, step=0.1, format="%.1f",
                value=float(st.session_state[f"{k}_rmax_a"]),
                key=f"{k}_rmax_a_slider",
                help="Maximum incentive APR Campaign A will pay. Budget may underspend below this TVL.",
            )

            total_tvl = nk_tvl + k_tvl  # $M
            if mode_a == "Set budget ($K/wk)":
                budget_a = st.slider(
                    "Budget A ($K/wk)",
                    min_value=0.0, max_value=1_000.0, step=0.5, format="%.1f",
                    value=float(st.session_state[f"{k}_budg_a"]),
                    key=f"{k}_budg_a_slider",
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
                )
                budget_b = budget_for_target_rate(target_rate_b, k_tvl, rmax_b)
                st.caption(f"→ Required budget: **${budget_b:.1f}K/wk**")

        # ── Compute current state ──────────────────────────────────────────
        c = compute_rates(nk_tvl, k_tvl, budget_a, budget_b, rmax_a, rmax_b)
        non_kraken_allin = c["non_kraken_merkl"] + base_apy
        kraken_allin = c["kraken_merkl"] + base_apy
        floor_breached = (venue.floor_apy is not None) and (non_kraken_allin < venue.floor_apy)
        kraken_below_floor = c["kraken_merkl"] < min_kraken_merkl

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
                delta=f"{'⚠ BELOW {:.1f}%'.format(min_kraken_merkl) if kraken_below_floor else '✓ Above floor'}",
                delta_color="inverse" if kraken_below_floor else "normal",
            )
            if kraken_below_floor:
                st.error(f"Kraken Merkl below {min_kraken_merkl:.1f}% minimum!", icon="🔴")

        with m3:
            st.metric("Campaign A rate", f"{c['rate_a']:.3f}%",
                      delta="at cap" if c["at_cap_a"] else "below cap",
                      delta_color="off")
            st.metric("Campaign B rate", f"{c['rate_b']:.3f}%",
                      delta="at cap" if c["at_cap_b"] else "below cap",
                      delta_color="off")

        with m4:
            st.metric(
                "Total spend (A+B)",
                f"${c['total_spend']:.1f}K/wk",
                delta=f"-${c['unspent']:.1f}K unspent" if c["unspent"] > 0.5 else "fully deployed",
                delta_color="normal" if c["unspent"] > 0.5 else "off",
            )
            st.metric(
                "Annualised cost",
                f"${(c['total_spend'] * 52 / 1_000):.2f}M/yr",
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
            delta="✓ comfortable" if headroom_b > 20 else ("⚠ tightening" if headroom_b > 5 else "⚠ at cap"),
            delta_color="normal" if headroom_b > 20 else ("off" if headroom_b > 5 else "inverse"),
        )
        cap3.metric(
            f"Max Kraken TVL >{min_kraken_merkl:.1f}% Merkl",
            f"${max_kraken_for_floor:.1f}M",
            help=f"Maximum Kraken TVL before combined Merkl rate drops below {min_kraken_merkl:.1f}%.",
        )
        cap4.metric(
            "Additional Kraken capacity",
            f"+${additional_before_3p5:.1f}M",
            delta="✓ ample" if additional_before_3p5 > 50 else ("⚠ moderate" if additional_before_3p5 > 20 else "⚠ limited"),
            delta_color="normal" if additional_before_3p5 > 50 else ("off" if additional_before_3p5 > 20 else "inverse"),
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
                ra1.metric("New Budget A", f"${a_new:.1f}K/wk", delta=f"{a_new - budget_a:+.1f}K", delta_color="inverse")
                ra2.metric("New Budget B", f"${b_new:.1f}K/wk", delta=f"{b_new - budget_b:+.1f}K", delta_color="normal")
                ra3.metric("Kraken Merkl change", f"{c_new['kraken_merkl']:.3f}%", delta=f"{delta_kraken:+.3f}%", delta_color="normal")
                ra4.metric("Non-Kraken Merkl change", f"{c_new['non_kraken_merkl']:.3f}%", delta=f"{delta_nk:+.3f}%", delta_color="inverse" if delta_nk < 0 else "normal")

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

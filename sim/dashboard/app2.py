"""
Campaign Optimizer Dashboard

Runs the full Monte Carlo simulation engine per-venue.

Inputs that require human judgment:
  - Target TVL, Target Utilization per venue
  - Total budget per program
  - Optional: pin budget or pin r_max for contractual constraints
  - Optional: pin incentive rate (research team override only)

Inputs fetched automatically:
  - Base APY (on-chain: Morpho sleeve-weighted avg, DeFiLlama for others)
  - Current TVL (from venue config — can be enhanced with live fetch)

The optimizer always searches the (B, r_max) hybrid surface.
Float and MAX are boundary cases it may land on.

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from campaign.agents import MercenaryConfig, RetailDepositorConfig, WhaleProfile
from campaign.base_apy import BaseAPYResult, fetch_all_base_apys
from campaign.engine import LossWeights
from campaign.optimizer import SurfaceGrid, SurfaceResult, optimize_surface
from campaign.state import CampaignEnvironment

st.set_page_config(page_title="Campaign Optimizer", page_icon="🎯", layout="wide")


# ============================================================================
# VENUE DEFINITIONS — Your real-world numbers
# ============================================================================
# Only things that require human judgment or can't be fetched:
#   - current_tvl, target_tvl, target_util
#   - budget_min/max constraints, r_max search range
#   - whale profiles (from on-chain analysis)
#   - r_threshold (competitor analysis)
# Base APY is fetched dynamically. Incentive rate is OUTPUT, not input.

PROGRAMS = {
    "RLUSD Core": {
        "total_budget": 869_700,
        "venues": [
            {
                "name": "AAVE Core Market",
                "asset": "RLUSD",
                "protocol": "aave",
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 600_000_000,
                "target_tvl": 600_000_000,
                "target_util": 0.40,
                "r_threshold": 0.045,
                "budget_min": 300_000,
                "budget_max": 700_000,
                "r_max_range": (0.03, 0.08),
                "whales": [],
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
                "r_threshold": 0.05,
                "budget_min": 100_000,
                "budget_max": 350_000,
                "r_max_range": (0.04, 0.10),
                "whales": [
                    WhaleProfile(
                        "euler_rlusd_w1",
                        38_000_000,
                        alt_rate=0.05,
                        risk_premium=0.005,
                        switching_cost_usd=3000,
                        exit_delay_days=3.0,
                        whale_type="institutional",
                    ),
                ],
            },
            {
                "name": "Curve RLUSD-USDC",
                "asset": "RLUSD",
                "protocol": "curve",
                "defillama_project": "curve-dex",
                "chain": "Ethereum",
                "current_tvl": 75_000_000,
                "target_tvl": 75_000_000,
                "target_util": 0.50,
                "r_threshold": 0.04,
                "budget_min": 40_000,
                "budget_max": 150_000,
                "r_max_range": (0.04, 0.10),
                "whales": [],
            },
        ],
    },
    "RLUSD Horizon": {
        "total_budget": 180_500,
        "venues": [
            {
                "name": "AAVE Horizon",
                "asset": "RLUSD",
                "protocol": "aave",
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 221_500_000,
                "target_tvl": 221_500_000,
                "target_util": 0.60,
                "r_threshold": 0.05,
                "budget_min": 100_000,
                "budget_max": 250_000,
                "r_max_range": (0.03, 0.08),
                "whales": [],
            },
        ],
    },
    "PYUSD": {
        "total_budget": 1_300_000,
        "venues": [
            {
                "name": "AAVE Core Market",
                "asset": "PYUSD",
                "protocol": "aave",
                "defillama_project": "aave-v3",
                "chain": "Ethereum",
                "current_tvl": 400_000_000,
                "target_tvl": 400_000_000,
                "target_util": 0.60,
                "r_threshold": 0.05,
                "budget_min": 200_000,
                "budget_max": 500_000,
                "r_max_range": (0.03, 0.08),
                "whales": [],
            },
            {
                "name": "Kamino Main Market",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-lend",
                "chain": "Solana",
                "current_tvl": 461_500_000,
                "target_tvl": 461_500_000,
                "target_util": 0.315,
                "r_threshold": 0.03,
                "budget_min": 100_000,
                "budget_max": 300_000,
                "r_max_range": (0.02, 0.06),
                "whales": [],
            },
            {
                "name": "Kamino Earn Vault",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino",
                "chain": "Solana",
                "current_tvl": 352_000_000,
                "target_tvl": 352_000_000,
                "target_util": 0.42,
                "r_threshold": 0.04,
                "budget_min": 150_000,
                "budget_max": 450_000,
                "r_max_range": (0.03, 0.08),
                "whales": [],
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
                "r_threshold": 0.05,
                "budget_min": 200_000,
                "budget_max": 400_000,
                "r_max_range": (0.04, 0.10),
                "whales": [
                    WhaleProfile(
                        "euler_pyusd_w1",
                        50_000_000,
                        alt_rate=0.05,
                        risk_premium=0.005,
                        switching_cost_usd=3000,
                        exit_delay_days=3.0,
                        whale_type="institutional",
                    ),
                ],
            },
            {
                "name": "Morpho PYUSD",
                "asset": "PYUSD",
                "protocol": "morpho",
                "vault_address": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
                "chain": "Ethereum",
                "current_tvl": 195_000_000,
                "target_tvl": 100_000_000,
                "target_util": 0.90,
                "r_threshold": 0.045,
                "budget_min": 50_000,
                "budget_max": 200_000,
                "r_max_range": (0.04, 0.12),
                "whales": [
                    WhaleProfile(
                        "morpho_w1",
                        29_600_000,
                        alt_rate=0.05,
                        risk_premium=0.003,
                        switching_cost_usd=2000,
                        exit_delay_days=3.0,
                        whale_type="institutional",
                    ),
                    WhaleProfile(
                        "morpho_w2",
                        25_000_000,
                        alt_rate=0.048,
                        risk_premium=0.004,
                        switching_cost_usd=1800,
                        exit_delay_days=2.0,
                        whale_type="quant_desk",
                    ),
                ],
            },
            {
                "name": "Kamino CLMM",
                "asset": "PYUSD",
                "protocol": "kamino",
                "defillama_project": "kamino-liquidity",
                "chain": "Solana",
                "current_tvl": 30_000_000,
                "target_tvl": 30_000_000,
                "target_util": 0.50,
                "r_threshold": 0.04,
                "budget_min": 20_000,
                "budget_max": 60_000,
                "r_max_range": (0.04, 0.10),
                "whales": [],
            },
            {
                "name": "Curve PYUSD-USDC",
                "asset": "PYUSD",
                "protocol": "curve",
                "defillama_project": "curve-dex",
                "chain": "Ethereum",
                "current_tvl": 30_000_000,
                "target_tvl": 30_000_000,
                "target_util": 0.50,
                "r_threshold": 0.04,
                "budget_min": 15_000,
                "budget_max": 60_000,
                "r_max_range": (0.03, 0.10),
                "whales": [],
            },
        ],
    },
}


# ============================================================================
# SIMULATION DEFAULTS
# ============================================================================

GRID_B_STEPS = 8
GRID_R_STEPS = 10
MC_PATHS_DEFAULT = 10
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


def implied_budget(tvl: float, rate: float) -> float:
    return tvl * rate * 7.0 / 365.0


def run_venue_optimization(
    venue: dict,
    base_apy: float,
    target_tvl: float,
    target_util: float,
    pinned_budget: float | None = None,
    pinned_r_max: float | None = None,
    n_paths: int = MC_PATHS_DEFAULT,
) -> SurfaceResult:
    """
    Run full MC surface optimization for one venue.

    The optimizer finds (B*, r_max*). The incentive rate is an OUTPUT:
      incentive_rate_at_target = B* / target_tvl * 365/7 (capped at r_max*)
    """
    # Budget grid
    if pinned_budget is not None:
        b_min = pinned_budget * 0.95
        b_max = pinned_budget * 1.05
        b_steps = 3
    else:
        b_min = venue["budget_min"]
        b_max = venue["budget_max"]
        b_steps = GRID_B_STEPS

    # r_max grid
    if pinned_r_max is not None:
        r_lo = pinned_r_max * 0.95
        r_hi = pinned_r_max * 1.05
        r_steps = 3
    else:
        r_lo, r_hi = venue["r_max_range"]
        r_steps = GRID_R_STEPS

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
    )

    env = CampaignEnvironment(r_threshold=venue["r_threshold"])

    # APR target for loss function: use r_threshold as the "desirable" total APR
    # The optimizer will find what incentive rate achieves this
    apr_target_total = venue["r_threshold"] * 1.2  # Aim ~20% above competitor avg

    weights = LossWeights(
        w_spend=1.0,
        w_spend_waste_penalty=2.0,
        w_apr_variance=5e5,
        w_apr_ceiling=1e8,
        w_tvl_shortfall=5e-7,
        apr_target=apr_target_total,
        apr_ceiling=0.10,
        tvl_target=target_tvl,
        apr_stability_on_total=True,
    )

    retail = RetailDepositorConfig(
        alpha_plus=0.3,
        alpha_minus_multiplier=3.0,
        response_lag_days=5.0,
        diffusion_sigma=0.015,
    )
    merc = MercenaryConfig(
        entry_threshold=0.08,
        exit_threshold=0.06,
        max_capital_usd=target_tvl * 0.1,
    )

    return optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=venue["current_tvl"],
        whale_profiles=venue.get("whales", []),
        weights=weights,
        n_paths=n_paths,
        retail_config=retail,
        mercenary_config=merc,
        verbose=False,
    )


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("🎯 Merkl Campaign Optimizer")
    st.caption(
        "Full MC simulation engine. Optimizer searches the (B, r_max) hybrid surface. "
        "Base APY fetched on-chain. Incentive rate is an **output**, not an input."
    )

    # ── Program selector ──
    program_name = st.sidebar.selectbox("Program", list(PROGRAMS.keys()))
    prog = PROGRAMS[program_name]
    venues = prog["venues"]

    total_budget = st.sidebar.number_input(
        "Total Weekly Budget ($)",
        value=int(prog["total_budget"]),
        step=10_000,
        min_value=0,
        key="total_budget",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_paths = st.sidebar.slider("MC Paths", 5, 100, MC_PATHS_DEFAULT, 5)

    # ── Fetch base APYs ──
    st.header("📡 Base APY (Fetched On-Chain)")

    if "base_apys" not in st.session_state or st.button("🔄 Refresh Base APYs"):
        with st.spinner("Fetching base APYs from on-chain / DeFiLlama..."):
            base_apy_results = fetch_all_base_apys(venues)
            st.session_state["base_apys"] = base_apy_results

    base_apys: dict[str, BaseAPYResult] = st.session_state.get("base_apys", {})

    if base_apys:
        brows = []
        for v in venues:
            r = base_apys.get(v["name"])
            if r:
                brows.append(
                    {
                        "Venue": v["name"],
                        "Base APY": r.base_apy,
                        "Source": r.source,
                    }
                )
            else:
                brows.append({"Venue": v["name"], "Base APY": 0.0, "Source": "not fetched"})

        bdf = pd.DataFrame(brows)
        st.dataframe(
            bdf.style.format({"Base APY": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Morpho sleeve detail
        for v in venues:
            r = base_apys.get(v["name"])
            if r and r.source == "morpho_graphql" and "sleeves" in r.details:
                with st.expander(f"🔍 {v['name']} — Sleeve Breakdown"):
                    srows = []
                    for s in r.details["sleeves"]:
                        srows.append(
                            {
                                "Collateral": s["collateral"],
                                "Supply ($M)": s["supply_usd"] / 1e6,
                                "APY": s["apy"],
                                "Weight": s["weight"],
                                "Weighted APY": s["apy"] * s["weight"],
                            }
                        )
                    sdf = pd.DataFrame(srows)
                    st.dataframe(
                        sdf.style.format(
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
                        f"Vault-level base APY: **{r.base_apy:.2%}** "
                        f"(active markets only: {r.details.get('active_only_apy', 0):.2%}, "
                        f"idle fraction: {r.details.get('idle_fraction', 0):.1%})"
                    )
    else:
        st.info("Click **Refresh Base APYs** to fetch current on-chain rates.")

    # ── Per-venue overrides ──
    st.header(f"📋 {program_name} — Venue Targets")
    st.caption(
        "Set target TVL and utilization. Incentive rate is computed by the optimizer. "
        "Pin budget/r_max only for contractual constraints or research team overrides."
    )

    overrides = {}
    for i, v in enumerate(venues):
        base_r = base_apys.get(v["name"])
        fetched_base = base_r.base_apy if base_r else 0.0

        with st.expander(
            f"**{v['name']}** ({v['protocol'].upper()}) — Base APY: {fetched_base:.2%}",
            expanded=(i < 2),
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                t_tvl = st.number_input(
                    "Target TVL ($M)",
                    value=v["target_tvl"] / 1e6,
                    step=10.0,
                    min_value=0.0,
                    key=f"tvl_{i}",
                )
            with c2:
                t_util = st.number_input(
                    "Target Util (%)",
                    value=v["target_util"] * 100,
                    step=1.0,
                    min_value=0.0,
                    max_value=100.0,
                    key=f"util_{i}",
                )
            with c3:
                base_override = st.number_input(
                    "Base APY Override (%) — leave 0 to use fetched",
                    value=0.0,
                    step=0.1,
                    min_value=0.0,
                    key=f"base_ov_{i}",
                )

            st.markdown("**Constraints** (optional — only for contractual/research overrides):")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                pin_b = st.checkbox("Pin Budget?", key=f"pin_b_{i}")
                pin_b_val = None
                if pin_b:
                    pin_b_val = st.number_input(
                        "Pinned Budget ($/wk)",
                        value=int(v["budget_min"]),
                        step=5000,
                        key=f"pinbval_{i}",
                    )
            with cc2:
                pin_r = st.checkbox("Pin r_max?", key=f"pin_r_{i}")
                pin_r_val = None
                if pin_r:
                    pin_r_val = (
                        st.number_input(
                            "Pinned r_max (%)",
                            value=6.0,
                            step=0.25,
                            min_value=0.0,
                            key=f"pinrval_{i}",
                        )
                        / 100.0
                    )
            with cc3:
                # Research team rate constraint — this overrides the APR target in loss
                force_rate = st.checkbox(
                    "Force incentive rate?",
                    key=f"force_r_{i}",
                    help="Research team override: force optimizer to target a specific rate",
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
                        )
                        / 100.0
                    )

            effective_base = (base_override / 100.0) if base_override > 0 else fetched_base

            overrides[v["name"]] = {
                "target_tvl": t_tvl * 1e6,
                "target_util": t_util / 100,
                "base_apy": effective_base,
                "pinned_budget": pin_b_val,
                "pinned_r_max": pin_r_val,
                "forced_rate": forced_rate,
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
                "Target TVL": ov["target_tvl"],
                "Target Util": ov["target_util"],
                "Net Supply": ns,
                "Base APY": ov["base_apy"],
                "Constraints": (
                    ("📌B " if ov["pinned_budget"] else "")
                    + ("📌r " if ov["pinned_r_max"] else "")
                    + ("🎯rate " if ov["forced_rate"] else "")
                ).strip()
                or "—",
            }
        )
    sdf = pd.DataFrame(srows)
    st.dataframe(
        sdf.style.format(
            {
                "Target TVL": "${:,.0f}",
                "Target Util": "{:.1%}",
                "Net Supply": "${:,.0f}",
                "Base APY": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── RUN ──
    st.markdown("---")
    run_btn = st.button(
        "🚀 Run Optimization (Full MC Simulation)",
        type="primary",
        use_container_width=True,
    )

    if run_btn:
        results = {}
        progress = st.progress(0, text="Initializing...")

        for idx, v in enumerate(venues):
            ov = overrides[v["name"]]
            progress.progress(idx / len(venues), text=f"Optimizing {v['name']}...")

            t0 = time.time()
            sr = run_venue_optimization(
                venue=v,
                base_apy=ov["base_apy"],
                target_tvl=ov["target_tvl"],
                target_util=ov["target_util"],
                pinned_budget=ov["pinned_budget"],
                pinned_r_max=ov["pinned_r_max"],
                n_paths=n_paths,
            )
            elapsed = time.time() - t0
            results[v["name"]] = {
                "surface": sr,
                "venue": v,
                "overrides": ov,
                "time": elapsed,
            }

        progress.progress(1.0, text="Done!")
        st.session_state["results"] = results

    # ── RESULTS ──
    if "results" not in st.session_state:
        return

    results = st.session_state["results"]

    st.header("📊 Optimal Campaign Parameters")
    st.caption(
        "B* and r_max* found by MC simulation. "
        "The incentive rate at any TVL is: min(r_max*, B*/TVL × 365/7). "
        "Campaign type (Float-like / Hybrid / MAX-like) is emergent, not chosen."
    )

    opt_rows = []
    for v in venues:
        if v["name"] not in results:
            continue
        r = results[v["name"]]
        sr = r["surface"]
        ov = r["overrides"]
        base = ov["base_apy"]
        mc = sr.optimal_mc_result

        B_star = sr.optimal_B
        r_max_star = sr.optimal_r_max
        tb = t_bind(B_star, r_max_star)
        inc_current = apr_at_tvl(B_star, v["current_tvl"], r_max_star)
        inc_target = apr_at_tvl(B_star, ov["target_tvl"], r_max_star)

        if tb < v["current_tvl"] * 0.3:
            ctype = "Float-like"
        elif tb > v["current_tvl"] * 1.5:
            ctype = "MAX-like"
        else:
            ctype = "Hybrid"

        ns = ov["target_tvl"] * (1 - ov["target_util"])

        opt_rows.append(
            {
                "Venue": v["name"],
                "B* ($/wk)": B_star,
                "r_max*": r_max_star,
                "Base APY": base,
                "Incentive @ Current TVL": inc_current,
                "Incentive @ Target TVL": inc_target,
                "Total APR @ Current": base + inc_current,
                "Total APR @ Target": base + inc_target,
                "T_bind ($M)": tb / 1e6,
                "Mean TVL ($M)": mc.mean_tvl / 1e6 if mc else 0,
                "Budget Util": mc.mean_budget_util if mc else 0,
                "Type": ctype,
                "Loss": sr.optimal_loss,
                "Net Supply ($M)": ns / 1e6,
                "TVL/$inc": ov["target_tvl"] / B_star if B_star > 0 else 0,
                "NS/$inc": ns / B_star if B_star > 0 else 0,
                "Time (s)": r["time"],
            }
        )

    odf = pd.DataFrame(opt_rows)

    # Main results table
    display_cols = [
        "Venue",
        "B* ($/wk)",
        "r_max*",
        "Base APY",
        "Incentive @ Target TVL",
        "Total APR @ Target",
        "T_bind ($M)",
        "Mean TVL ($M)",
        "Budget Util",
        "Type",
    ]
    st.dataframe(
        odf[display_cols].style.format(
            {
                "B* ($/wk)": "${:,.0f}",
                "r_max*": "{:.2%}",
                "Base APY": "{:.2%}",
                "Incentive @ Target TVL": "{:.2%}",
                "Total APR @ Target": "{:.2%}",
                "T_bind ($M)": "${:.1f}M",
                "Mean TVL ($M)": "${:.0f}M",
                "Budget Util": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Budget allocation ──
    total_opt = sum(r["B* ($/wk)"] for r in opt_rows)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.metric("Optimizer Total", f"${total_opt:,.0f}/wk")
    with bc2:
        st.metric("Budget Envelope", f"${total_budget:,.0f}/wk")
    with bc3:
        diff = total_budget - total_opt
        st.metric("Remaining", f"${diff:+,.0f}/wk")

    # ── Efficiency ──
    st.subheader("Efficiency Metrics")
    eff_cols = ["Venue", "B* ($/wk)", "Net Supply ($M)", "TVL/$inc", "NS/$inc"]
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
        mc = sr.optimal_mc_result
        B_star = sr.optimal_B
        r_star = sr.optimal_r_max

        with st.expander(f"🔬 {v['name']}"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("B*", f"${B_star:,.0f}/wk")
            with d2:
                st.metric("r_max*", f"{r_star:.2%}")
            with d3:
                st.metric("T_bind", f"${sr.optimal_t_bind / 1e6:.1f}M")
            with d4:
                st.metric("Loss", f"{sr.optimal_loss:.3e}")

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

            # Loss surface
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import LogNorm

                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                Bv = sr.grid.B_values
                rv = sr.grid.r_max_values
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
                    label="Optimal",
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
                "weekly_budget": round(row["B* ($/wk)"]),
                "r_max": round(row["r_max*"], 4),
                "base_apy": round(row["Base APY"], 4),
                "incentive_at_current_tvl": round(row["Incentive @ Current TVL"], 4),
                "incentive_at_target_tvl": round(row["Incentive @ Target TVL"], 4),
                "total_apr_at_target": round(row["Total APR @ Target"], 4),
                "t_bind": round(row["T_bind ($M)"] * 1e6),
                "campaign_type": row["Type"],
                "target_tvl": round(ov["target_tvl"]),
                "target_utilization": round(ov["target_util"], 3),
                "net_supply": round(ov["target_tvl"] * (1 - ov["target_util"])),
                "tvl_per_incentive": round(row["TVL/$inc"]),
                "ns_per_incentive": round(row["NS/$inc"]),
            }
        )
    st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()

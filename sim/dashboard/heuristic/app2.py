"""
Campaign Optimizer Dashboard — Minimal Working Version.

Computes optimal Merkl campaign parameters (B, r_max) for each venue
given program-level budget constraints and per-venue targets.

The optimizer always chooses hybrid parameters. Float and MAX are edge
cases (r_max -> inf or B -> inf). The output tells you:
- At current TVL: what APR depositors see
- At target TVL: what APR depositors see
- r_max: the cap that protects against overspend if TVL drops

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Campaign Optimizer", page_icon="🎯", layout="wide")

# ============================================================================
# VENUE DATA — your real-world numbers
# ============================================================================

PROGRAMS = {
    "RLUSD Core": {
        "asset": "RLUSD",
        "total_budget": 869_700,
        "venues": [
            {
                "name": "AAVE Core Market",
                "protocol": "AAVE",
                "target_tvl": 600e6,
                "target_util": 0.40,
                "target_rate": 0.0475,
                "base_apy": 0.02,
                "current_tvl": 600e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Euler Sentora RLUSD",
                "protocol": "Euler",
                "target_tvl": 190e6,
                "target_util": 0.385,
                "target_rate": 0.065,
                "base_apy": 0.015,
                "current_tvl": 190e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Curve RLUSD-USDC",
                "protocol": "Curve",
                "target_tvl": 75e6,
                "target_util": 0.50,
                "target_rate": 0.06,
                "base_apy": 0.005,
                "current_tvl": 75e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
        ],
    },
    "RLUSD Horizon": {
        "asset": "RLUSD",
        "total_budget": 180_500,
        "venues": [
            {
                "name": "AAVE Horizon",
                "protocol": "AAVE",
                "target_tvl": 221.5e6,
                "target_util": 0.60,
                "target_rate": 0.0425,
                "base_apy": 0.02,
                "current_tvl": 221.5e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
        ],
    },
    "PYUSD": {
        "asset": "PYUSD",
        "total_budget": 1_300_000,
        "venues": [
            {
                "name": "AAVE Core Market",
                "protocol": "AAVE",
                "target_tvl": 400e6,
                "target_util": 0.60,
                "target_rate": 0.04,
                "base_apy": 0.0332,
                "current_tvl": 400e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Kamino Main Market",
                "protocol": "Kamino",
                "target_tvl": 461.5e6,
                "target_util": 0.315,
                "target_rate": 0.0226,
                "base_apy": 0.01,
                "current_tvl": 461.5e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": 200_000,
            },
            {
                "name": "Kamino Earn Vault",
                "protocol": "Kamino",
                "target_tvl": 352e6,
                "target_util": 0.42,
                "target_rate": 0.0444,
                "base_apy": 0.015,
                "current_tvl": 352e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Kamino CLMM",
                "protocol": "Kamino",
                "target_tvl": 30e6,
                "target_util": 0.50,
                "target_rate": 0.061,
                "base_apy": 0.01,
                "current_tvl": 30e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Euler Sentora PYUSD",
                "protocol": "Euler",
                "target_tvl": 250e6,
                "target_util": 0.52,
                "target_rate": 0.065,
                "base_apy": 0.02,
                "current_tvl": 250e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Curve PYUSD-USDC",
                "protocol": "Curve",
                "target_tvl": 30e6,
                "target_util": 0.50,
                "target_rate": 0.055,
                "base_apy": 0.005,
                "current_tvl": 30e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
            {
                "name": "Morpho PYUSD",
                "protocol": "Morpho",
                "target_tvl": 100e6,
                "target_util": 0.90,
                "target_rate": 0.06,
                "base_apy": 0.0115,
                "current_tvl": 195e6,
                "budget_min": 0,
                "budget_max": 1e9,
                "pinned_budget": None,
            },
        ],
    },
}


# ============================================================================
# CORE MATH — campaign parameter computation
# ============================================================================


def implied_weekly_budget(tvl: float, rate: float) -> float:
    """B = TVL * r_incentive * 7/365"""
    return tvl * rate * 7.0 / 365.0


def incentive_apr_at_tvl(B: float, tvl: float, r_max: float) -> float:
    """r_incentive(TVL) = min(r_max, B/TVL * 365/7)"""
    if tvl <= 0:
        return r_max
    return min(r_max, B / tvl * 365.0 / 7.0)


def total_apr_at_tvl(B: float, tvl: float, r_max: float, base_apy: float) -> float:
    """Total APR depositors see = base + incentive"""
    return base_apy + incentive_apr_at_tvl(B, tvl, r_max)


def t_bind(B: float, r_max: float) -> float:
    """TVL threshold where regime switches: T_bind = B * 365/7 / r_max"""
    if r_max <= 0:
        return float("inf")
    return B * 365.0 / 7.0 / r_max


def net_supply(tvl: float, util: float) -> float:
    return tvl * (1.0 - util)


def spend_per_week(B: float, tvl: float, r_max: float) -> float:
    """Actual weekly spend = TVL * r_incentive * 7/365 (capped by B)."""
    r = incentive_apr_at_tvl(B, tvl, r_max)
    return tvl * r * 7.0 / 365.0


def campaign_type_label(B: float, r_max: float, current_tvl: float, target_tvl: float) -> str:
    """Classify the campaign regime at current and target TVL."""
    tb = t_bind(B, r_max)
    if tb <= 0:
        return "Float"
    if current_tvl < tb and target_tvl < tb:
        return "MAX-like"
    if current_tvl > tb and target_tvl > tb:
        return "Float-like"
    return "Hybrid"


def allocate_program_budget(
    venues: list[dict],
    total_budget: float,
) -> list[dict]:
    """
    Allocate a program's total budget across venues.

    Strategy:
    1. Pinned venues get their exact budget
    2. Remaining budget distributed proportional to implied budget (from targets)
    3. r_max chosen so T_bind sits ~25% below target TVL (protective buffer)
       This means at target TVL the cap doesn't bind (Float-like),
       but if TVL drops 25%+ the cap kicks in (protection).

    Returns list of venue dicts with allocated parameters.
    """
    results = []
    pinned_total = sum(v.get("pinned_budget") or 0 for v in venues if v.get("pinned_budget"))
    remaining = total_budget - pinned_total

    # Compute implied budgets for unpinned venues
    unpinned = [v for v in venues if not v.get("pinned_budget")]
    implied = [implied_weekly_budget(v["target_tvl"], v["target_rate"]) for v in unpinned]
    implied_total = sum(implied)

    for v in venues:
        if v.get("pinned_budget"):
            B = v["pinned_budget"]
        elif implied_total > 0:
            share = implied_weekly_budget(v["target_tvl"], v["target_rate"]) / implied_total
            B = remaining * share
        else:
            B = remaining / max(len(unpinned), 1)

        # Clamp to min/max
        B = max(v.get("budget_min", 0), min(v.get("budget_max", 1e12), B))

        # Choose r_max: set T_bind at 75% of target TVL (25% buffer)
        # This means at target TVL, APR = B/TVL * 365/7 (Float regime, no cap)
        # If TVL drops below 75% of target, cap kicks in at r_max
        target = v["target_tvl"]
        t_bind_target = target * 0.75  # 25% below target
        if t_bind_target > 0:
            r_max_val = B * 365.0 / 7.0 / t_bind_target
        else:
            r_max_val = 0.12  # fallback

        # Sanity clamp r_max to [2%, 15%]
        r_max_val = max(0.02, min(0.15, r_max_val))

        results.append(
            {
                **v,
                "allocated_budget": B,
                "r_max": r_max_val,
            }
        )

    # Scale unpinned to exactly match remaining budget
    unpinned_allocated = sum(r["allocated_budget"] for r in results if not r.get("pinned_budget"))
    if unpinned_allocated > 0 and abs(unpinned_allocated - remaining) > 1:
        scale = remaining / unpinned_allocated
        for r in results:
            if not r.get("pinned_budget"):
                r["allocated_budget"] *= scale
                # Recompute r_max with scaled budget
                t_bind_target = r["target_tvl"] * 0.75
                if t_bind_target > 0:
                    r["r_max"] = max(
                        0.02, min(0.15, r["allocated_budget"] * 365.0 / 7.0 / t_bind_target)
                    )

    return results


# ============================================================================
# STREAMLIT APP
# ============================================================================


def main():
    st.title("🎯 Merkl Campaign Optimizer")
    st.caption(
        "Optimal hybrid campaign parameters (B, r_max) per venue. "
        "The optimizer always picks hybrid — Float and MAX are just edge cases."
    )

    # ── Program selector ──
    programs = deepcopy(PROGRAMS)
    selected = st.sidebar.selectbox("Program", list(programs.keys()))
    prog = programs[selected]

    total_budget = st.sidebar.number_input(
        f"{selected} Total Weekly Budget ($)",
        value=int(prog["total_budget"]),
        step=10_000,
        min_value=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("T_bind Buffer")
    t_bind_buffer = st.sidebar.slider(
        "T_bind as % of target TVL",
        min_value=50,
        max_value=100,
        value=75,
        step=5,
        help="Lower = more protection (cap kicks in sooner). "
        "75% means cap activates if TVL drops 25% below target.",
    )

    # ── Per-venue overrides ──
    st.header(f"📋 {selected} — Venue Targets")

    venues = prog["venues"]
    override_venues = []

    for i, v in enumerate(venues):
        with st.expander(f"{v['protocol']} — {v['name']}", expanded=(i < 3)):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                ttv = st.number_input(
                    "Target TVL ($M)",
                    value=v["target_tvl"] / 1e6,
                    step=10.0,
                    min_value=0.0,
                    key=f"tvl_{selected}_{i}",
                )
            with c2:
                tu = st.number_input(
                    "Target Util (%)",
                    value=v["target_util"] * 100,
                    step=1.0,
                    min_value=0.0,
                    max_value=100.0,
                    key=f"util_{selected}_{i}",
                )
            with c3:
                tr = st.number_input(
                    "Incentive Rate (%)",
                    value=v["target_rate"] * 100,
                    step=0.25,
                    min_value=0.0,
                    key=f"rate_{selected}_{i}",
                )
            with c4:
                ba = st.number_input(
                    "Base APY (%)",
                    value=v["base_apy"] * 100,
                    step=0.1,
                    min_value=0.0,
                    key=f"base_{selected}_{i}",
                )
            with c5:
                pin = st.checkbox(
                    "Pin Budget?",
                    value=v.get("pinned_budget") is not None,
                    key=f"pin_{selected}_{i}",
                )
                if pin:
                    pv = st.number_input(
                        "Pinned ($/wk)",
                        value=int(
                            v.get("pinned_budget") or implied_weekly_budget(ttv * 1e6, tr / 100)
                        ),
                        step=5000,
                        key=f"pinv_{selected}_{i}",
                    )
                else:
                    pv = None

            ctv = st.number_input(
                "Current TVL ($M)",
                value=v.get("current_tvl", v["target_tvl"]) / 1e6,
                step=10.0,
                min_value=0.0,
                key=f"ctvl_{selected}_{i}",
            )

            override_venues.append(
                {
                    **v,
                    "target_tvl": ttv * 1e6,
                    "target_util": tu / 100,
                    "target_rate": tr / 100,
                    "base_apy": ba / 100,
                    "current_tvl": ctv * 1e6,
                    "pinned_budget": pv,
                }
            )

    # ── Run allocation ──
    # Override t_bind buffer in allocator
    allocs = []
    pinned_total = sum(
        v.get("pinned_budget") or 0 for v in override_venues if v.get("pinned_budget")
    )
    remaining = total_budget - pinned_total

    unpinned = [v for v in override_venues if not v.get("pinned_budget")]
    implied_list = [implied_weekly_budget(v["target_tvl"], v["target_rate"]) for v in unpinned]
    implied_total = sum(implied_list)

    for v in override_venues:
        if v.get("pinned_budget"):
            B = v["pinned_budget"]
        elif implied_total > 0:
            share = implied_weekly_budget(v["target_tvl"], v["target_rate"]) / implied_total
            B = remaining * share
        else:
            B = remaining / max(len(unpinned), 1)
        B = max(v.get("budget_min", 0), min(v.get("budget_max", 1e12), B))
        allocs.append({**v, "allocated_budget": B})

    # Scale unpinned
    unp_alloc = sum(a["allocated_budget"] for a in allocs if not a.get("pinned_budget"))
    if unp_alloc > 0 and abs(unp_alloc - remaining) > 1:
        scale = remaining / unp_alloc
        for a in allocs:
            if not a.get("pinned_budget"):
                a["allocated_budget"] *= scale

    # Compute r_max for each
    for a in allocs:
        B = a["allocated_budget"]
        tb_target = a["target_tvl"] * (t_bind_buffer / 100.0)
        if tb_target > 0:
            a["r_max"] = max(0.02, min(0.15, B * 365.0 / 7.0 / tb_target))
        else:
            a["r_max"] = 0.12

    # ── Results table ──
    st.header("📊 Optimal Campaign Parameters")

    rows = []
    for a in allocs:
        B = a["allocated_budget"]
        rm = a["r_max"]
        base = a["base_apy"]
        cur = a["current_tvl"]
        tgt = a["target_tvl"]
        util = a["target_util"]

        inc_at_current = incentive_apr_at_tvl(B, cur, rm)
        inc_at_target = incentive_apr_at_tvl(B, tgt, rm)
        total_at_current = base + inc_at_current
        total_at_target = base + inc_at_target
        tb = t_bind(B, rm)
        ns = net_supply(tgt, util)
        actual_spend_current = spend_per_week(B, cur, rm)
        actual_spend_target = spend_per_week(B, tgt, rm)
        ctype = campaign_type_label(B, rm, cur, tgt)

        rows.append(
            {
                "Venue": f"{a['protocol']} — {a['name']}",
                "Budget ($/wk)": B,
                "r_max (%)": rm * 100,
                "Base APY (%)": base * 100,
                "APR @ Current TVL (%)": total_at_current * 100,
                "Incentive @ Current (%)": inc_at_current * 100,
                "APR @ Target TVL (%)": total_at_target * 100,
                "Incentive @ Target (%)": inc_at_target * 100,
                "T_bind ($M)": tb / 1e6,
                "Target TVL ($M)": tgt / 1e6,
                "Current TVL ($M)": cur / 1e6,
                "Net Supply ($M)": ns / 1e6,
                "Spend @ Current ($/wk)": actual_spend_current,
                "Spend @ Target ($/wk)": actual_spend_target,
                "TVL / $incentive": tgt / B if B > 0 else 0,
                "NS / $incentive": ns / B if B > 0 else 0,
                "Type": ctype,
                "Pinned": "📌" if a.get("pinned_budget") else "",
            }
        )

    df = pd.DataFrame(rows)

    # Format and display
    st.dataframe(
        df.style.format(
            {
                "Budget ($/wk)": "${:,.0f}",
                "r_max (%)": "{:.2f}",
                "Base APY (%)": "{:.2f}",
                "APR @ Current TVL (%)": "{:.2f}",
                "Incentive @ Current (%)": "{:.2f}",
                "APR @ Target TVL (%)": "{:.2f}",
                "Incentive @ Target (%)": "{:.2f}",
                "T_bind ($M)": "${:.1f}",
                "Target TVL ($M)": "${:.1f}",
                "Current TVL ($M)": "${:.1f}",
                "Net Supply ($M)": "${:.1f}",
                "Spend @ Current ($/wk)": "${:,.0f}",
                "Spend @ Target ($/wk)": "${:,.0f}",
                "TVL / $incentive": "{:,.1f}",
                "NS / $incentive": "{:,.1f}",
            }
        ).background_gradient(subset=["TVL / $incentive"], cmap="RdYlGn"),
        use_container_width=True,
        height=min(600, 60 + len(rows) * 38),
    )

    # ── Summary metrics ──
    st.header("📈 Program Summary")

    tot_budget = sum(r["Budget ($/wk)"] for r in rows)
    tot_target_tvl = sum(r["Target TVL ($M)"] for r in rows)
    tot_ns = sum(r["Net Supply ($M)"] for r in rows)
    tot_spend_current = sum(r["Spend @ Current ($/wk)"] for r in rows)
    tot_spend_target = sum(r["Spend @ Target ($/wk)"] for r in rows)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Budget", f"${tot_budget:,.0f}/wk")
    with c2:
        st.metric("Target TVL", f"${tot_target_tvl:,.1f}M")
    with c3:
        st.metric("Net Supply", f"${tot_ns:,.1f}M")
    with c4:
        st.metric("Spend @ Current", f"${tot_spend_current:,.0f}/wk")
    with c5:
        st.metric("Spend @ Target", f"${tot_spend_target:,.0f}/wk")

    budget_gap = total_budget - tot_budget
    if abs(budget_gap) > 100:
        if budget_gap > 0:
            st.warning(f"⚠️ ${budget_gap:,.0f}/wk unallocated from total program budget")
        else:
            st.error(f"🚨 ${-budget_gap:,.0f}/wk over total program budget")

    # ── APR Waterfall ──
    st.header("🔍 APR Breakdown by Venue")
    st.caption("Shows base APY + incentive APR at both current and target TVL")

    for a in allocs:
        B = a["allocated_budget"]
        rm = a["r_max"]
        base = a["base_apy"]
        cur = a["current_tvl"]
        tgt = a["target_tvl"]
        tb = t_bind(B, rm)

        inc_cur = incentive_apr_at_tvl(B, cur, rm)
        inc_tgt = incentive_apr_at_tvl(B, tgt, rm)
        cap_binding_cur = "🔒 cap" if cur < tb else "📈 float"
        cap_binding_tgt = "🔒 cap" if tgt < tb else "📈 float"

        with st.expander(f"{a['protocol']} — {a['name']}"):
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Budget", f"${B:,.0f}/wk")
                st.metric("r_max (incentive cap)", f"{rm:.2%}")
            with mc2:
                st.metric("@ Current TVL", f"{(base + inc_cur):.2%}")
                st.caption(f"Base {base:.2%} + Incentive {inc_cur:.2%} {cap_binding_cur}")
            with mc3:
                st.metric("@ Target TVL", f"{(base + inc_tgt):.2%}")
                st.caption(f"Base {base:.2%} + Incentive {inc_tgt:.2%} {cap_binding_tgt}")
            with mc4:
                st.metric("T_bind", f"${tb / 1e6:.1f}M")
                st.caption(f"{tb / tgt * 100:.0f}% of target TVL" if tgt > 0 else "N/A")

            # Show APR curve across TVL range
            tvl_range = np.linspace(max(tgt * 0.3, 1e6), tgt * 1.5, 100)
            apr_curve = [total_apr_at_tvl(B, t, rm, base) * 100 for t in tvl_range]
            inc_curve = [incentive_apr_at_tvl(B, t, rm) * 100 for t in tvl_range]

            chart_df = pd.DataFrame(
                {
                    "TVL ($M)": tvl_range / 1e6,
                    "Total APR (%)": apr_curve,
                    "Incentive APR (%)": inc_curve,
                    "Base APY (%)": [base * 100] * len(tvl_range),
                }
            ).set_index("TVL ($M)")

            st.line_chart(chart_df, height=250)

            # Mark key points
            st.caption(
                f"T_bind = ${tb / 1e6:.1f}M (vertical regime switch) · "
                f"Current TVL = ${cur / 1e6:.1f}M · "
                f"Target TVL = ${tgt / 1e6:.1f}M"
            )

    # ── Export ──
    st.header("📤 Export")
    if st.button("Generate JSON"):
        export = {
            "program": selected,
            "total_budget": total_budget,
            "t_bind_buffer_pct": t_bind_buffer,
            "generated_at": time.strftime("%Y-%m-%d %H:%M UTC"),
            "venues": [
                {
                    "venue": f"{a['protocol']} — {a['name']}",
                    "asset": prog["asset"],
                    "weekly_budget": round(a["allocated_budget"]),
                    "r_max": round(a["r_max"], 4),
                    "base_apy": a["base_apy"],
                    "target_tvl": a["target_tvl"],
                    "target_utilization": a["target_util"],
                    "t_bind": round(t_bind(a["allocated_budget"], a["r_max"])),
                    "apr_at_current_tvl": round(
                        total_apr_at_tvl(
                            a["allocated_budget"], a["current_tvl"], a["r_max"], a["base_apy"]
                        ),
                        4,
                    ),
                    "apr_at_target_tvl": round(
                        total_apr_at_tvl(
                            a["allocated_budget"], a["target_tvl"], a["r_max"], a["base_apy"]
                        ),
                        4,
                    ),
                    "incentive_at_target": round(
                        incentive_apr_at_tvl(a["allocated_budget"], a["target_tvl"], a["r_max"]), 4
                    ),
                    "campaign_type": campaign_type_label(
                        a["allocated_budget"], a["r_max"], a["current_tvl"], a["target_tvl"]
                    ),
                    "pinned": a.get("pinned_budget") is not None,
                }
                for a in allocs
            ],
        }
        st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()

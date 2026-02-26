"""
Multi-Venue Campaign Optimizer Dashboard.

Supports:
- Per-program budget allocation (RLUSD Core, RLUSD Horizon, PYUSD)
- Per-venue target TVL / utilization / rate overrides
- Budget pinning (contractual constraints)
- Real-time weight adjustment (no re-simulation)
- Cross-venue efficiency comparison
- Pre-computed surface loading

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Campaign Optimizer — Multi-Venue",
    page_icon="🎯",
    layout="wide",
)

RESULTS_DIR = Path("results")

# Search for venues.json in multiple locations
_VENUES_SEARCH = [
    Path("venues.json"),
    Path("configs/venues.json"),
    Path(__file__).parent.parent / "venues.json",
    Path(__file__).parent.parent / "configs" / "venues.json",
]


def find_config_path() -> Path | None:
    for p in _VENUES_SEARCH:
        if p.exists():
            return p
    return None


# ============================================================================
# DATA LOADING
# ============================================================================


@st.cache_data
def load_venue_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_surfaces(result_dir: str) -> dict | None:
    p = Path(result_dir)
    npz = p / "surfaces.npz"
    if not npz.exists():
        return None
    data = dict(np.load(npz))
    if "feasibility_mask" in data:
        data["feasibility_mask"] = data["feasibility_mask"].astype(bool)
    return data


@st.cache_data
def load_meta(result_dir: str) -> dict | None:
    p = Path(result_dir) / "metadata.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_allocation(result_dir: str) -> dict | None:
    p = Path(result_dir) / "allocation.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def find_result_dirs() -> dict[str, str]:
    if not RESULTS_DIR.exists():
        return {}
    dirs = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir():
            if (d / "surfaces.npz").exists() or (d / "allocation.json").exists():
                dirs[d.name] = str(d)
            # Check subdirectories (multi-venue puts per-venue surfaces in subdirs)
            for sd in d.iterdir():
                if sd.is_dir() and (sd / "surfaces.npz").exists():
                    key = f"{d.name}/{sd.name}"
                    dirs[key] = str(sd)
    return dirs


# ============================================================================
# RECOMPUTE HELPERS
# ============================================================================


def recompute_loss(surfaces: dict, weights: dict[str, float]) -> np.ndarray:
    loss = np.zeros_like(surfaces["loss_surface"])
    mapping = {
        "spend": "component_spend",
        "apr_variance": "component_apr_variance",
        "apr_ceiling": "component_apr_ceiling",
        "tvl_shortfall": "component_tvl_shortfall",
        "merkl_fee": "component_merkl_fee",
    }
    for k, skey in mapping.items():
        if skey in surfaces:
            loss += weights.get(k, 1.0) * surfaces[skey]
    return loss


def find_optimal(loss: np.ndarray, feas: np.ndarray) -> tuple[int, int]:
    masked = np.where(feas, loss, np.inf)
    return np.unravel_index(np.argmin(masked), masked.shape)


def implied_budget(target_tvl: float, rate: float) -> float:
    return target_tvl * rate * 7 / 365


def net_supply(tvl: float, util: float) -> float:
    return tvl * (1 - util)


def tvl_per_incentive(tvl: float, budget: float) -> float:
    return tvl / budget if budget > 0 else 0


def net_supply_per_incentive(tvl: float, util: float, budget: float) -> float:
    return net_supply(tvl, util) / budget if budget > 0 else 0


# ============================================================================
# PLOTTING
# ============================================================================


def plot_surface(B, r, loss, feas, oi, oj, title="Loss Surface", base_apy=0.0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    pl = np.where(feas, loss, np.nan)
    v = pl[~np.isnan(pl)]
    norm = None
    if len(v) > 0 and v.max() / max(v.min(), 1e-10) > 100:
        norm = LogNorm(vmin=max(v.min(), 1e-10), vmax=v.max())
    im = ax.pcolormesh(r * 100, B / 1000, pl, cmap="viridis_r", norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label="Loss", shrink=0.8)
    ax.plot(r[oj] * 100, B[oi] / 1000, "*", color="red", ms=14, mec="white", mew=1.5)
    xlabel = "Incentive APR Cap r_max (%)"
    if base_apy > 0:
        xlabel += f"  [total at cap = r_max + {base_apy:.1%} base]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Budget B ($k/wk)")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_component_surfaces(surfaces: dict, feas: np.ndarray, B_vals, r_vals):
    """Plot individual loss component surfaces side-by-side."""
    import matplotlib.pyplot as plt

    components = {
        "Spend Cost": "component_spend",
        "APR Variance": "component_apr_variance",
        "APR Ceiling": "component_apr_ceiling",
        "TVL Shortfall": "component_tvl_shortfall",
        "Merkl Fee": "component_merkl_fee",
    }

    available = {k: v for k, v in components.items() if v in surfaces}
    if not available:
        return

    n = len(available)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, key) in enumerate(available.items()):
        ax = axes[idx]
        data = np.where(feas, surfaces[key], np.nan)
        im = ax.pcolormesh(r_vals * 100, B_vals / 1000, data, cmap="viridis_r", shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("r_max (%)", fontsize=8)
        ax.set_ylabel("B ($k/wk)", fontsize=8)

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("🎯 Multi-Venue Campaign Optimizer")
    st.caption("Configure targets per venue, allocate budgets across programs, explore surfaces.")

    # ── Load config ──
    cfg_path = find_config_path()
    if cfg_path is None:
        st.error(
            "Config not found. Place `venues.json` in the project root or `configs/` directory.\n\n"
            "Expected locations:\n" + "\n".join(f"- `{p}`" for p in _VENUES_SEARCH)
        )
        return

    st.sidebar.caption(f"Config: `{cfg_path}`")
    raw_cfg = load_venue_config(str(cfg_path))
    programs = raw_cfg.get("programs", {})
    result_dirs = find_result_dirs()

    # ============================================================
    # SIDEBAR: Program & Global Controls
    # ============================================================

    st.sidebar.header("Programs")
    selected_program = st.sidebar.selectbox("Program", list(programs.keys()))
    prog = programs[selected_program]
    venues_cfg = prog["venues"]

    total_budget = st.sidebar.number_input(
        "Total Weekly Budget ($)",
        value=int(prog["total_weekly_budget"]),
        step=10000,
        key="total_budget",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Loss Weights (all venues)")
    st.sidebar.caption("Adjust to reweight pre-computed surfaces instantly.")
    w = {
        "spend": st.sidebar.slider("Spend minimization", 0.0, 5.0, 1.0, 0.1, key="ws"),
        "apr_variance": st.sidebar.slider("APR stability", 0.0, 5.0, 1.0, 0.1, key="wav"),
        "apr_ceiling": st.sidebar.slider("APR ceiling", 0.0, 5.0, 1.0, 0.1, key="wac"),
        "tvl_shortfall": st.sidebar.slider("TVL target", 0.0, 5.0, 1.0, 0.1, key="wts"),
        "merkl_fee": st.sidebar.slider("Fee efficiency", 0.0, 5.0, 1.0, 0.1, key="wmf"),
    }

    # ============================================================
    # VENUE CONFIGURATION TABLE (read-only summary)
    # ============================================================

    st.header(f"📋 {selected_program} — Venue Configuration")

    rows = []
    for v in venues_cfg:
        b_implied = implied_budget(v["target_tvl"], v["target_incentive_rate"])
        base = v.get("base_apy", 0)
        rows.append(
            {
                "Venue": v["name"],
                "Protocol": v["protocol"].capitalize(),
                "Target TVL ($M)": v["target_tvl"] / 1e6,
                "Target Util (%)": v["target_utilization"] * 100,
                "Net Supply ($M)": net_supply(v["target_tvl"], v["target_utilization"]) / 1e6,
                "Base APY (%)": base * 100,
                "Target Incentive (%)": v["target_incentive_rate"] * 100,
                "Total APR (%)": (base + v["target_incentive_rate"]) * 100,
                "Implied Budget ($/wk)": b_implied,
                "Pinned": "📌" if v.get("pinned_budget") else "",
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format(
            {
                "Target TVL ($M)": "{:.1f}",
                "Target Util (%)": "{:.1f}",
                "Net Supply ($M)": "{:.1f}",
                "Base APY (%)": "{:.2f}",
                "Target Incentive (%)": "{:.2f}",
                "Total APR (%)": "{:.2f}",
                "Implied Budget ($/wk)": "${:,.0f}",
            }
        ),
        use_container_width=True,
        height=min(400, 60 + len(rows) * 35),
    )

    total_implied = sum(
        implied_budget(v["target_tvl"], v["target_incentive_rate"]) for v in venues_cfg
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Implied Budget", f"${total_implied:,.0f}/wk")
    with col_b:
        st.metric("Allocated Budget", f"${total_budget:,.0f}/wk")
    with col_c:
        delta = total_budget - total_implied
        st.metric(
            "Budget Gap",
            f"${delta:+,.0f}/wk",
            delta=f"{'surplus' if delta >= 0 else 'deficit'}",
            delta_color="normal" if delta >= 0 else "inverse",
        )

    # ============================================================
    # VENUE OVERRIDE EXPANDERS (per-venue editable targets)
    # ============================================================

    st.header("⚙️ Per-Venue Overrides")
    st.caption("Expand any venue to adjust targets, pin budget, or modify constraints.")

    overrides = {}
    for i, v in enumerate(venues_cfg):
        base = v.get("base_apy", 0)
        with st.expander(f"{v['name']} ({v['protocol'].capitalize()}) — base {base:.1%}"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                t_tvl = st.number_input(
                    "Target TVL ($M)",
                    value=v["target_tvl"] / 1e6,
                    step=10.0,
                    key=f"tvl_{i}",
                )
            with c2:
                t_util = st.number_input(
                    "Target Util (%)",
                    value=v["target_utilization"] * 100,
                    step=1.0,
                    min_value=0.0,
                    max_value=100.0,
                    key=f"util_{i}",
                )
            with c3:
                t_rate = st.number_input(
                    "Target Incentive Rate (%)",
                    value=v["target_incentive_rate"] * 100,
                    step=0.25,
                    min_value=0.0,
                    key=f"rate_{i}",
                )
            with c4:
                pin = st.checkbox(
                    "Pin Budget?", value=v.get("pinned_budget") is not None, key=f"pin_{i}"
                )
                if pin:
                    pin_val = st.number_input(
                        "Pinned Budget ($/wk)",
                        value=int(
                            v.get("pinned_budget") or implied_budget(t_tvl * 1e6, t_rate / 100)
                        ),
                        step=5000,
                        key=f"pinval_{i}",
                    )
                else:
                    pin_val = None

            st.caption(
                f"Total APR at target: {base + t_rate / 100:.2%} "
                f"(base {base:.2%} + incentive {t_rate / 100:.2%})"
            )

            overrides[v["name"]] = {
                "target_tvl": t_tvl * 1e6,
                "target_utilization": t_util / 100,
                "target_incentive_rate": t_rate / 100,
                "pinned_budget": pin_val,
            }

    # ============================================================
    # EFFICIENCY METRICS
    # ============================================================

    st.header("📊 Efficiency Metrics")

    eff_rows = []
    for v in venues_cfg:
        ov = overrides.get(v["name"], {})
        tvl = ov.get("target_tvl", v["target_tvl"])
        util = ov.get("target_utilization", v["target_utilization"])
        rate = ov.get("target_incentive_rate", v["target_incentive_rate"])
        base = v.get("base_apy", 0)
        budget = ov.get("pinned_budget") or implied_budget(tvl, rate)
        ns = net_supply(tvl, util)

        eff_rows.append(
            {
                "Venue": v["name"],
                "Target TVL ($M)": tvl / 1e6,
                "Net Supply ($M)": ns / 1e6,
                "Base (%)": base * 100,
                "Incentive (%)": rate * 100,
                "Total (%)": (base + rate) * 100,
                "Budget ($/wk)": budget,
                "TVL / $incentive": tvl_per_incentive(tvl, budget),
                "NetSupply / $incentive": net_supply_per_incentive(tvl, util, budget),
                "Budget Share (%)": 0,
            }
        )

    tot_b = sum(r["Budget ($/wk)"] for r in eff_rows)
    for r in eff_rows:
        r["Budget Share (%)"] = r["Budget ($/wk)"] / tot_b * 100 if tot_b > 0 else 0

    eff_df = pd.DataFrame(eff_rows)
    st.dataframe(
        eff_df.style.format(
            {
                "Target TVL ($M)": "{:.1f}",
                "Net Supply ($M)": "{:.1f}",
                "Base (%)": "{:.2f}",
                "Incentive (%)": "{:.2f}",
                "Total (%)": "{:.2f}",
                "Budget ($/wk)": "${:,.0f}",
                "TVL / $incentive": "{:,.1f}",
                "NetSupply / $incentive": "{:,.1f}",
                "Budget Share (%)": "{:.1f}",
            }
        ).background_gradient(subset=["TVL / $incentive"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # ============================================================
    # PRE-COMPUTED SURFACE EXPLORATION
    # ============================================================

    st.header("🔬 Surface Explorer")

    if not result_dirs:
        st.info(
            "No pre-computed surfaces found in `results/`. Run:\n\n"
            "```bash\n"
            "python scripts/compute_surface.py --venue morpho_pyusd --output results/morpho_pyusd\n"
            "# or multi-venue:\n"
            "python scripts/compute_surface.py --multi --config configs/venues_test.json --output results/test_allocation\n"
            "```"
        )
    else:
        sel_dir = st.selectbox("Result set", list(result_dirs.keys()))
        dir_path = result_dirs[sel_dir]

        # ── Multi-venue allocation result ──
        alloc = load_allocation(dir_path)
        if alloc:
            st.subheader("Multi-Venue Allocation Result")

            alloc_rows = []
            for a in alloc.get("allocations", []):
                base = a.get("base_apy", 0)
                inc_apr = a.get("mean_incentive_apr", a["mean_apr"] - base)
                alloc_rows.append(
                    {
                        "Venue": a["name"],
                        "Protocol": a["protocol"],
                        "B ($/wk)": a["weekly_budget"],
                        "r_max (%)": a["apr_cap"] * 100,
                        "Base (%)": base * 100,
                        "Mean Incentive (%)": inc_apr * 100,
                        "Mean Total (%)": a["mean_apr"] * 100,
                        "T_bind ($M)": a["t_bind"] / 1e6,
                        "Mean TVL ($M)": a["mean_tvl"] / 1e6,
                        "Budget Util (%)": a["budget_utilization"] * 100,
                        "Cascade": a["cascade_depth"],
                        "Feasible": "✓" if a["is_feasible"] else "✗",
                        "Share (%)": a["budget_share"] * 100,
                        "Pin": "📌" if a.get("was_pinned") else "",
                    }
                )

            alloc_df = pd.DataFrame(alloc_rows)
            st.dataframe(
                alloc_df.style.format(
                    {
                        "B ($/wk)": "${:,.0f}",
                        "r_max (%)": "{:.2f}",
                        "Base (%)": "{:.2f}",
                        "Mean Incentive (%)": "{:.2f}",
                        "Mean Total (%)": "{:.2f}",
                        "T_bind ($M)": "{:.1f}",
                        "Mean TVL ($M)": "{:.1f}",
                        "Budget Util (%)": "{:.1f}",
                        "Cascade": "{:.1f}",
                        "Share (%)": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

            acol1, acol2, acol3 = st.columns(3)
            with acol1:
                st.metric("Total Budget", f"${alloc.get('total_budget', 0):,.0f}/wk")
            with acol2:
                st.metric("Total Loss", f"{alloc.get('total_loss', 0):.3e}")
            with acol3:
                st.metric("λ*", f"{alloc.get('lagrange_multiplier', 0):.3e}")

        # ── Single-venue surface ──
        surfaces = load_surfaces(dir_path)
        if surfaces is not None:
            meta = load_meta(dir_path)
            base_apy = meta.get("grid", {}).get("base_apy", 0) if meta else 0

            B_vals = surfaces["B_values"]
            r_vals = surfaces["r_max_values"]
            feas = surfaces["feasibility_mask"]

            custom_loss = recompute_loss(surfaces, w)
            oi, oj = find_optimal(custom_loss, feas)

            venue_label = sel_dir.split("/")[-1] if "/" in sel_dir else sel_dir
            st.subheader(f"Surface: {venue_label}")
            if base_apy > 0:
                st.caption(f"Base APY: {base_apy:.2%} — r_max values are incentive-only caps")

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Optimal B", f"${B_vals[oi]:,.0f}/wk")
            with mc2:
                inc_cap = r_vals[oj]
                total_cap = inc_cap + base_apy
                label = f"{inc_cap:.2%}"
                if base_apy > 0:
                    label += f" (total: {total_cap:.2%})"
                st.metric("Optimal r_max", label)
            with mc3:
                tb = B_vals[oi] * (365 / 7) / r_vals[oj] if r_vals[oj] > 0 else 0
                st.metric("T_bind", f"${tb:,.0f}")
            with mc4:
                st.metric("Loss", f"{custom_loss[oi, oj]:.3e}")

            if meta and "optimal_diagnostics" in meta:
                d = meta["optimal_diagnostics"]
                inc_apr = d.get("mean_incentive_apr", d["mean_apr"] - base_apy)
                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1:
                    st.metric("Mean Total APR", f"{d['mean_apr']:.2%}")
                    st.caption(f"base {base_apy:.2%} + incentive {inc_apr:.2%}")
                with dc2:
                    st.metric("APR Range", f"{d['apr_p5']:.1%}–{d['apr_p95']:.1%}")
                with dc3:
                    st.metric("Mean TVL", f"${d['mean_tvl'] / 1e6:.0f}M")
                with dc4:
                    st.metric("Budget Util", f"{d.get('mean_budget_util', 0):.0%}")

            tab_surf, tab_comp, tab_dual, tab_sens = st.tabs(
                ["Loss Surface", "Components", "Duality Map", "Sensitivity"]
            )

            with tab_surf:
                plot_surface(
                    B_vals,
                    r_vals,
                    custom_loss,
                    feas,
                    oi,
                    oj,
                    f"Loss Surface — {venue_label}",
                    base_apy=base_apy,
                )

            with tab_comp:
                st.caption("Individual loss components (before weighting)")
                plot_component_surfaces(surfaces, feas, B_vals, r_vals)

            with tab_dual:
                tol = st.slider("Tolerance (%)", 1, 20, 5, key="dual_tol") / 100
                opt_loss = custom_loss[oi, oj]
                thresh = opt_loss * (1 + tol)
                near = []
                for i in range(len(B_vals)):
                    for j in range(len(r_vals)):
                        if not feas[i, j]:
                            continue
                        if custom_loss[i, j] <= thresh:
                            inc = r_vals[j]
                            near.append(
                                {
                                    "B ($/wk)": f"${B_vals[i]:,.0f}",
                                    "r_max (%)": f"{inc:.2%}",
                                    "Total at cap": f"{inc + base_apy:.2%}",
                                    "T_bind ($M)": f"${B_vals[i] * (365 / 7) / r_vals[j] / 1e6:.1f}M"
                                    if r_vals[j] > 0
                                    else "∞",
                                    "Loss": f"{custom_loss[i, j]:.3e}",
                                    "vs Optimal": f"+{(custom_loss[i, j] / opt_loss - 1) * 100:.1f}%",
                                }
                            )
                st.caption(f"{len(near)} configs within {tol:.0%} of optimum")
                if near:
                    st.dataframe(near, use_container_width=True)

            with tab_sens:
                if meta and "sensitivity" in meta:
                    sa = meta["sensitivity"]
                    st.markdown(f"**{sa['interpretation']}**")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.metric("Condition Number", f"{sa['condition_number']:.1f}")
                    with sc2:
                        steep = sa["steep_direction"]
                        st.metric("Steep Direction", f"B: {steep[0]:.3f}, r_max: {steep[1]:.3f}")
                    st.caption(
                        "High condition number = loss surface is elongated. "
                        "You must be precise on the steep direction but have freedom on the flat direction."
                    )
                else:
                    st.info("Sensitivity data not available for this result set.")

    # ============================================================
    # CAMPAIGN RECOMMENDATION EXPORT
    # ============================================================

    st.header("📤 Campaign Recommendation")

    rec_rows = []
    for v in venues_cfg:
        ov = overrides.get(v["name"], {})
        tvl = ov.get("target_tvl", v["target_tvl"])
        util = ov.get("target_utilization", v["target_utilization"])
        rate = ov.get("target_incentive_rate", v["target_incentive_rate"])
        base = v.get("base_apy", 0)
        budget = ov.get("pinned_budget") or implied_budget(tvl, rate)

        rec_rows.append(
            {
                "Venue": v["name"],
                "Asset": v["asset_symbol"],
                "Protocol": v["protocol"].capitalize(),
                "Weekly Budget ($)": round(budget),
                "Recommended r_max (%)": rate * 100,
                "Base APY (%)": base * 100,
                "Total APR (%)": (base + rate) * 100,
                "Campaign Type": "MAX" if budget > implied_budget(tvl, rate) * 1.2 else "Hybrid",
                "Target TVL ($M)": tvl / 1e6,
                "Net Supply ($M)": net_supply(tvl, util) / 1e6,
                "Notes": "📌 Pinned" if ov.get("pinned_budget") else "",
            }
        )

    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(
        rec_df.style.format(
            {
                "Weekly Budget ($)": "${:,.0f}",
                "Recommended r_max (%)": "{:.2f}",
                "Base APY (%)": "{:.2f}",
                "Total APR (%)": "{:.2f}",
                "Target TVL ($M)": "{:.1f}",
                "Net Supply ($M)": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    total_rec = sum(r["Weekly Budget ($)"] for r in rec_rows)
    st.caption(f"Total recommended: ${total_rec:,.0f}/wk vs. allocated: ${total_budget:,.0f}/wk")

    # Export button
    if st.button("📋 Export Recommendation as JSON"):
        export = {
            "program": selected_program,
            "total_budget": total_budget,
            "generated_at": time.strftime("%Y-%m-%d %H:%M UTC"),
            "venues": [
                {
                    "name": r["Venue"],
                    "asset": r["Asset"],
                    "protocol": r["Protocol"],
                    "weekly_budget": r["Weekly Budget ($)"],
                    "r_max": r["Recommended r_max (%)"] / 100,
                    "base_apy": r["Base APY (%)"] / 100,
                    "total_apr": r["Total APR (%)"] / 100,
                    "campaign_type": r["Campaign Type"],
                    "target_tvl": r["Target TVL ($M)"] * 1e6,
                }
                for r in rec_rows
            ],
        }
        st.code(json.dumps(export, indent=2), language="json")


if __name__ == "__main__":
    main()

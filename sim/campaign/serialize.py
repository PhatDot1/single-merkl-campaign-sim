"""
Serialization for pre-computed surface results.

Save the heavy computation once, load instantly in the dashboard.
Stores numpy arrays as .npz and metadata as JSON.

Usage:
    # After running optimization
    result = optimize_surface(...)
    save_surface_result(result, "results/morpho_pyusd_20260210")

    # In dashboard
    result = load_surface_result("results/morpho_pyusd_20260210")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .optimizer import SurfaceGrid, SurfaceResult


def save_surface_result(result: SurfaceResult, output_dir: str) -> None:
    """
    Save SurfaceResult to disk.

    Creates:
    - {output_dir}/surfaces.npz — all numpy surfaces
    - {output_dir}/metadata.json — grid params, scalar diagnostics
    - {output_dir}/mc_diagnostics.json — per-point MC diagnostics

    Args:
        result: Completed SurfaceResult from optimize_surface()
        output_dir: Directory to save into (created if needed)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Numpy surfaces ──
    arrays = {
        "B_values": result.grid.B_values,
        "r_max_values": result.grid.r_max_values,
        "loss_surface": result.loss_surface,
        "loss_std_surface": result.loss_std_surface,
        "feasibility_mask": result.feasibility_mask.astype(np.uint8),
        "avg_apr_surface": result.avg_apr_surface,
        "avg_incentive_apr_surface": result.avg_incentive_apr_surface,
        "avg_tvl_surface": result.avg_tvl_surface,
        "budget_util_surface": result.budget_util_surface,
        "cascade_depth_surface": result.cascade_depth_surface,
        "mercenary_frac_surface": result.mercenary_frac_surface,
        "cap_binding_surface": result.cap_binding_surface,
    }

    # Add component surfaces
    for k, v in result.component_surfaces.items():
        arrays[f"component_{k}"] = v

    np.savez_compressed(out / "surfaces.npz", **arrays)

    # ── Metadata ──
    meta = {
        "grid": {
            "epoch_duration_days": result.grid.epoch_duration_days,
            "merkl_fee_rate": result.grid.merkl_fee_rate,
            "dt_days": result.grid.dt_days,
            "horizon_days": result.grid.horizon_days,
            "base_apy": result.grid.base_apy,
        },
        "optimal": {
            "B": result.optimal_B,
            "r_max": result.optimal_r_max,
            "t_bind": result.optimal_t_bind,
            "loss": result.optimal_loss,
            "indices": list(result.optimal_indices),
        },
        "sensitivity": result.sensitivity_analysis(),
        "duality_5pct": [
            {k: v for k, v in d.items() if k != "grid_indices"} for d in result.duality_map(0.05)
        ],
        "stability_boundary": result.stability_boundary(),
    }

    # Add optimal point MC diagnostics if available
    mc = result.optimal_mc_result
    if mc:
        meta["optimal_diagnostics"] = {
            "mean_apr": mc.mean_apr,
            "mean_incentive_apr": mc.mean_incentive_apr,
            "base_apy": mc.base_apy,
            "std_apr": mc.std_apr,
            "apr_p5": mc.apr_p5,
            "apr_p95": mc.apr_p95,
            "mean_tvl": mc.mean_tvl,
            "tvl_min_p5": mc.tvl_min_p5,
            "mean_spend": mc.mean_spend,
            "mean_budget_util": mc.mean_budget_util,
            "mean_cascade_depth": mc.mean_cascade_depth,
            "max_cascade_depth": mc.max_cascade_depth,
            "mean_mercenary_fraction": mc.mean_mercenary_fraction,
            "mean_time_cap_binding": mc.mean_time_cap_binding,
            "is_feasible": mc.is_feasible,
        }

    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── Per-point MC summary (for dashboard drill-down) ──
    mc_summary = {}
    for (i, j), mc_result in result.mc_results.items():
        mc_summary[f"{i},{j}"] = {
            "B": float(mc_result.B),
            "r_max": float(mc_result.r_max),
            "t_bind": float(mc_result.t_bind),
            "base_apy": float(mc_result.base_apy),
            "mean_loss": float(mc_result.mean_loss),
            "mean_apr": float(mc_result.mean_apr),
            "mean_incentive_apr": float(mc_result.mean_incentive_apr),
            "apr_p5": float(mc_result.apr_p5),
            "apr_p95": float(mc_result.apr_p95),
            "mean_tvl": float(mc_result.mean_tvl),
            "mean_budget_util": float(mc_result.mean_budget_util),
            "mean_cascade_depth": float(mc_result.mean_cascade_depth),
            "is_feasible": mc_result.is_feasible,
        }

    with open(out / "mc_diagnostics.json", "w") as f:
        json.dump(mc_summary, f, indent=2)


def load_surface_result(input_dir: str) -> SurfaceResult:
    """
    Load pre-computed SurfaceResult from disk.

    Args:
        input_dir: Directory containing surfaces.npz and metadata.json

    Returns:
        SurfaceResult (without raw MC path_results — only summary data)
    """
    inp = Path(input_dir)

    # ── Load numpy surfaces ──
    data = np.load(inp / "surfaces.npz")

    grid = SurfaceGrid(
        B_values=data["B_values"],
        r_max_values=data["r_max_values"],
    )

    # Read grid params from metadata
    with open(inp / "metadata.json") as f:
        meta = json.load(f)

    grid_meta = meta.get("grid", {})
    grid.epoch_duration_days = grid_meta.get("epoch_duration_days", 7)
    grid.merkl_fee_rate = grid_meta.get("merkl_fee_rate", 0.015)
    grid.dt_days = grid_meta.get("dt_days", 0.25)
    grid.horizon_days = grid_meta.get("horizon_days", 28)
    grid.base_apy = grid_meta.get("base_apy", 0.0)

    # Component surfaces
    component_surfaces = {}
    for key in data.files:
        if key.startswith("component_"):
            name = key[len("component_") :]
            component_surfaces[name] = data[key]

    # Handle avg_incentive_apr_surface (may not exist in older results)
    avg_incentive_apr = (
        data["avg_incentive_apr_surface"]
        if "avg_incentive_apr_surface" in data
        else np.zeros_like(data["avg_apr_surface"])
    )

    result = SurfaceResult(
        grid=grid,
        loss_surface=data["loss_surface"],
        loss_std_surface=data["loss_std_surface"],
        feasibility_mask=data["feasibility_mask"].astype(bool),
        component_surfaces=component_surfaces,
        avg_apr_surface=data["avg_apr_surface"],
        avg_incentive_apr_surface=avg_incentive_apr,
        avg_tvl_surface=data["avg_tvl_surface"],
        budget_util_surface=data["budget_util_surface"],
        cascade_depth_surface=data["cascade_depth_surface"],
        mercenary_frac_surface=data["mercenary_frac_surface"],
        cap_binding_surface=data["cap_binding_surface"],
    )

    return result


def load_metadata(input_dir: str) -> dict:
    """Load just the metadata (fast, no numpy)."""
    with open(Path(input_dir) / "metadata.json") as f:
        return json.load(f)


def load_mc_diagnostics(input_dir: str) -> dict:
    """Load per-point MC diagnostics for dashboard drill-down."""
    with open(Path(input_dir) / "mc_diagnostics.json") as f:
        return json.load(f)

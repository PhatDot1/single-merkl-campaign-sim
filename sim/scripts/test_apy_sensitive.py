#!/usr/bin/env python
"""
Smoke-test for APYSensitiveAgent rename + realistic PYUSD/RLUSD venues.

Tests:
  1. Single-venue: Morpho PYUSD with APYSensitiveConfig (floor_apr=4%, sensitivity=0.5)
  2. Single-venue: Euler Sentora RLUSD without APYSensitive (baseline)
  3. Multi-venue: RLUSD Core program (Aave + Euler + Curve) with APYSensitive
  4. Multi-venue: 2 PYUSD venues (Morpho + Euler) with APYSensitive

Uses venue_registry for addresses/targets, lightweight MC (30 paths, 10x10 grid).
No live data fetching — uses synthetic whales for speed.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure .env is loaded for RPC URLs
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

import numpy as np
from campaign.agents import (
    APYSensitiveConfig,
    WhaleProfile,
)
from campaign.engine import LossWeights
from campaign.multi_venue import VenueSpec, allocate_budget
from campaign.optimizer import SurfaceGrid, optimize_surface
from campaign.state import CampaignEnvironment
from campaign.venue_registry import (
    PROGRAM_BUDGETS,
    get_program_venues,
    get_venue,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

N_PATHS = 30  # Fast smoke test
GRID_B = 10
GRID_R = 10
DT_DAYS = 0.5
HORIZON_DAYS = 28


# Synthetic whale profiles (skip live fetch for speed)
def _make_whales(tvl: float, r_thresh: float) -> list[WhaleProfile]:
    """Create 3 synthetic whale profiles scaled to TVL.

    WhaleProfile.exit_threshold is a computed property:
      exit_threshold = alt_rate + risk_premium - switching_cost / position
    We set alt_rate to produce the desired exit threshold.
    """
    return [
        WhaleProfile(
            whale_id="whale_large",
            position_usd=tvl * 0.08,
            alt_rate=r_thresh * 0.65,  # Sticky — exits only at very low APR
            risk_premium=0.005,
            switching_cost_usd=5_000,
            whale_type="institutional",
            exit_delay_days=3.0,
        ),
        WhaleProfile(
            whale_id="whale_medium",
            position_usd=tvl * 0.04,
            alt_rate=r_thresh * 0.80,  # Moderate threshold
            risk_premium=0.005,
            switching_cost_usd=1_000,
            whale_type="quant_desk",
            exit_delay_days=1.5,
        ),
        WhaleProfile(
            whale_id="whale_small",
            position_usd=tvl * 0.02,
            alt_rate=r_thresh * 0.90,  # Exits quickly
            risk_premium=0.005,
            switching_cost_usd=500,
            whale_type="opportunistic",
            exit_delay_days=1.0,
        ),
    ]


def _mc_sensitive_stats(mc):
    """Extract sensitive_fraction and time_below_floor from MC path_results."""
    if not mc or not mc.path_results:
        return 0.0, 0.0
    sf = float(np.mean([r.sensitive_fraction for r in mc.path_results]))
    tbf = float(np.mean([r.time_below_floor for r in mc.path_results]))
    return sf, tbf


# ──────────────────────────────────────────────────────────────
# Test 1: Single venue — Morpho PYUSD WITH APYSensitive
# ──────────────────────────────────────────────────────────────


def test_single_venue_morpho_pyusd():
    print("=" * 70)
    print("TEST 1: Single-Venue — Morpho PYUSD with APYSensitiveConfig")
    print("=" * 70)

    v = get_venue("pyusd-morpho-sentora")
    tvl = v.current_tvl
    target = v.target_tvl
    r_thresh = 0.045  # PYUSD typical competitor rate
    base_apy = 0.012  # Morpho organic yield ~1.2%

    apy_sensitive = APYSensitiveConfig(
        floor_apr=0.04,  # 4% total APR floor
        sensitivity=0.5,  # Moderate reactivity
        leverage_multiple=3.0,  # Typical 3x leveraged position
        max_sensitive_tvl=tvl * 0.10,
    )

    grid = SurfaceGrid.from_ranges(
        B_min=30_000,
        B_max=200_000,
        B_steps=GRID_B,
        r_max_min=0.03,
        r_max_max=0.08,
        r_max_steps=GRID_R,
        dt_days=DT_DAYS,
        horizon_days=HORIZON_DAYS,
        base_apy=base_apy,
    )

    weights = LossWeights(
        w_spend=1.0,
        w_apr_variance=3.0,
        w_apr_ceiling=5.0,
        w_tvl_shortfall=8.0,
        w_mercenary=6.0,
        w_whale_proximity=6.0,
        w_apr_floor=7.0,
        apr_target=base_apy + 0.055,
        apr_ceiling=0.10,
        tvl_target=target,
        apr_stability_on_total=True,
        apr_floor=apy_sensitive.floor_apr,
        apr_floor_sensitivity=apy_sensitive.sensitivity,
    )

    whales = _make_whales(tvl, r_thresh)
    env = CampaignEnvironment(r_threshold=r_thresh)

    t0 = time.time()
    sr = optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=tvl,
        whale_profiles=whales,
        weights=weights,
        n_paths=N_PATHS,
        apy_sensitive_config=apy_sensitive,
        verbose=False,
    )
    elapsed = time.time() - t0

    mc = sr.optimal_mc_result
    sf, tbf = _mc_sensitive_stats(mc)

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Optimal B*: ${sr.optimal_B:,.0f}/wk")
    print(f"  Optimal r_max*: {sr.optimal_r_max:.2%}")
    print(f"  Optimal loss: {sr.optimal_loss:.4e}")
    if mc:
        print(f"  Mean total APR: {mc.mean_apr:.2%}")
        print(f"  Mean incentive APR: {mc.mean_incentive_apr:.2%}")
        print(f"  Mean TVL: ${mc.mean_tvl / 1e6:.1f}M")
        print(f"  Budget util: {mc.mean_budget_util:.1%}")
        print(f"  Mercenary frac: {mc.mean_mercenary_fraction:.1%}")
        print(f"  Sensitive frac: {sf:.1%}")
        print(f"  Time below floor: {tbf:.1%}")

    assert sr.optimal_loss < float("inf"), "Loss should be finite"
    assert sr.optimal_B > 0, "Budget should be positive"
    print("  ✅ PASS")
    return sr


# ──────────────────────────────────────────────────────────────
# Test 2: Single venue — Euler RLUSD without APYSensitive
# ──────────────────────────────────────────────────────────────


def test_single_venue_euler_rlusd():
    print("\n" + "=" * 70)
    print("TEST 2: Single-Venue — Euler Sentora RLUSD (no APYSensitive)")
    print("=" * 70)

    v = get_venue("rlusd-euler-v2-sentora")
    tvl = v.current_tvl
    target = v.target_tvl
    r_thresh = 0.040  # RLUSD blended benchmark
    base_apy = 0.015

    grid = SurfaceGrid.from_ranges(
        B_min=80_000,
        B_max=350_000,
        B_steps=GRID_B,
        r_max_min=0.03,
        r_max_max=0.07,
        r_max_steps=GRID_R,
        dt_days=DT_DAYS,
        horizon_days=HORIZON_DAYS,
        base_apy=base_apy,
    )

    weights = LossWeights(
        w_spend=1.0,
        w_apr_variance=3.0,
        w_apr_ceiling=5.0,
        w_tvl_shortfall=8.0,
        w_mercenary=6.0,
        w_whale_proximity=6.0,
        apr_target=base_apy + 0.055,
        apr_ceiling=0.10,
        tvl_target=target,
        apr_stability_on_total=True,
    )

    whales = _make_whales(tvl, r_thresh)
    env = CampaignEnvironment(r_threshold=r_thresh)

    t0 = time.time()
    sr = optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=tvl,
        whale_profiles=whales,
        weights=weights,
        n_paths=N_PATHS,
        apy_sensitive_config=None,  # No APY-sensitive agent
        verbose=False,
    )
    elapsed = time.time() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Optimal B*: ${sr.optimal_B:,.0f}/wk")
    print(f"  Optimal r_max*: {sr.optimal_r_max:.2%}")
    print(f"  Optimal loss: {sr.optimal_loss:.4e}")
    mc = sr.optimal_mc_result
    if mc:
        print(f"  Mean total APR: {mc.mean_apr:.2%}")
        print(f"  Mean incentive APR: {mc.mean_incentive_apr:.2%}")
        print(f"  Mean TVL: ${mc.mean_tvl / 1e6:.1f}M")
        print(f"  Budget util: {mc.mean_budget_util:.1%}")
        print(f"  Mercenary frac: {mc.mean_mercenary_fraction:.1%}")

    assert sr.optimal_loss < float("inf"), "Loss should be finite"
    print("  ✅ PASS")
    return sr


# ──────────────────────────────────────────────────────────────
# Test 3: Multi-venue — RLUSD Core (3 venues) with APYSensitive
# ──────────────────────────────────────────────────────────────


def test_multi_venue_rlusd_core():
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Venue — RLUSD Core (Aave + Euler + Curve)")
    print("         with APYSensitiveConfig on lending venues")
    print("=" * 70)

    total_budget = PROGRAM_BUDGETS["RLUSD Core"]
    r_thresh = 0.040

    specs = []
    for v in get_program_venues("RLUSD Core"):
        is_lending = v.protocol in ("aave", "euler", "morpho")
        base = {"aave": 0.025, "euler": 0.015, "curve": 0.01}.get(v.protocol, 0.01)

        apy_cfg = None
        if is_lending:
            apy_cfg = APYSensitiveConfig(
                floor_apr=0.035,
                sensitivity=0.3,
                leverage_multiple=1.0,  # Non-leveraged (yield chasers)
                max_sensitive_tvl=v.current_tvl * 0.08,
            )

        specs.append(
            VenueSpec(
                name=v.name,
                asset_symbol=v.asset,
                protocol=v.protocol,
                current_tvl=v.current_tvl,
                current_utilization=v.target_util,
                target_tvl=v.target_tvl,
                target_utilization=v.target_util,
                target_incentive_rate=0.055,
                base_apy=base,
                budget_min=v.budget_min,
                budget_max=v.budget_max,
                whale_profiles=_make_whales(v.current_tvl, r_thresh),
                env=CampaignEnvironment(r_threshold=r_thresh),
                apy_sensitive_config=apy_cfg,
            )
        )

    t0 = time.time()
    result = allocate_budget(
        venues=specs,
        total_budget=total_budget,
        n_paths=N_PATHS,
        budget_steps=GRID_B,
        r_max_steps=GRID_R,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Total budget: ${total_budget:,.0f}/wk")
    print(f"  Allocated: ${result.budget_allocated:,.0f}/wk")
    for alloc in result.allocations:
        print(
            f"  {alloc.name}: ${alloc.weekly_budget:,.0f}/wk, "
            f"r_max={alloc.apr_cap:.2%}, TVL=${alloc.mean_tvl / 1e6:.1f}M"
        )

    assert result.budget_allocated > 0, "Should allocate some budget"
    assert len(result.allocations) == len(specs), "Should have all venues"
    print("\n  ✅ PASS")
    return result


# ──────────────────────────────────────────────────────────────
# Test 4: Multi-venue — 2 PYUSD venues with APYSensitive
# ──────────────────────────────────────────────────────────────


def test_multi_venue_pyusd_subset():
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Venue — PYUSD (Morpho + Euler Sentora)")
    print("         with APYSensitiveConfig (leveraged on Morpho)")
    print("=" * 70)

    total_budget = 400_000  # Subset of PYUSD budget for these 2 venues
    r_thresh = 0.045

    # Morpho PYUSD — high leveraged concentration
    morpho = get_venue("pyusd-morpho-sentora")
    morpho_spec = VenueSpec(
        name=morpho.name,
        asset_symbol="PYUSD",
        protocol="morpho",
        current_tvl=morpho.current_tvl,
        current_utilization=0.44,
        target_tvl=morpho.target_tvl,
        target_utilization=0.90,
        target_incentive_rate=0.06,
        base_apy=0.012,
        budget_min=30_000,
        budget_max=200_000,
        whale_profiles=_make_whales(morpho.current_tvl, r_thresh),
        env=CampaignEnvironment(r_threshold=r_thresh),
        apy_sensitive_config=APYSensitiveConfig(
            floor_apr=0.04,
            sensitivity=0.6,  # High sensitivity — leveraged positions
            leverage_multiple=3.0,
            max_sensitive_tvl=morpho.current_tvl * 0.15,
        ),
    )

    # Euler Sentora PYUSD — moderate sensitivity
    euler = get_venue("pyusd-euler-v2-sentora")
    euler_spec = VenueSpec(
        name=euler.name,
        asset_symbol="PYUSD",
        protocol="euler",
        current_tvl=euler.current_tvl,
        current_utilization=0.52,
        target_tvl=euler.target_tvl,
        target_utilization=0.52,
        target_incentive_rate=0.065,
        base_apy=0.02,
        budget_min=150_000,
        budget_max=350_000,
        whale_profiles=_make_whales(euler.current_tvl, r_thresh),
        env=CampaignEnvironment(r_threshold=r_thresh),
        apy_sensitive_config=APYSensitiveConfig(
            floor_apr=0.035,
            sensitivity=0.3,  # Lower sensitivity — yield chasers
            leverage_multiple=1.0,
            max_sensitive_tvl=euler.current_tvl * 0.08,
        ),
    )

    t0 = time.time()
    result = allocate_budget(
        venues=[morpho_spec, euler_spec],
        total_budget=total_budget,
        n_paths=N_PATHS,
        budget_steps=GRID_B,
        r_max_steps=GRID_R,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Total budget: ${total_budget:,.0f}/wk")
    for alloc in result.allocations:
        sr = result.venue_surfaces.get(alloc.name)
        mc = sr.optimal_mc_result if sr else None
        sf, tbf = _mc_sensitive_stats(mc)
        print(f"\n  {alloc.name}:")
        print(f"    Budget: ${alloc.weekly_budget:,.0f}/wk ({alloc.budget_share:.1%})")
        print(f"    r_max: {alloc.apr_cap:.2%}")
        print(f"    Mean total APR: {alloc.mean_apr:.2%}")
        print(f"    Mean TVL: ${alloc.mean_tvl / 1e6:.1f}M")
        if mc:
            print(f"    Sensitive frac: {sf:.1%}")
            print(f"    Time below floor: {tbf:.1%}")

    assert result.budget_allocated > 0
    print("\n  ✅ PASS")
    return result


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 APYSensitiveAgent Rename + Realistic Venue Test\n")
    t_total = time.time()

    try:
        test_single_venue_morpho_pyusd()
        test_single_venue_euler_rlusd()
        test_multi_venue_rlusd_core()
        test_multi_venue_pyusd_subset()
    except Exception as e:
        import traceback

        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"✅ ALL 4 TESTS PASSED in {time.time() - t_total:.1f}s")
    print(f"{'=' * 70}")

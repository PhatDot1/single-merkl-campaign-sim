"""
Multi-venue budget allocation (Section 7.6).

Given N venues with a shared total budget B_total, find the allocation
(B_i, r_max_i) for each venue that minimizes combined loss subject to
the shared budget constraint: sum(B_i) = B_total.

Architecture:
1. Each venue runs its own optimize_surface() independently
2. The allocator reads the marginal loss curves from each surface
3. Lagrangian optimization distributes budget where dL/dB is steepest
4. Venues can be "pinned" (fixed B and/or r_max) for contractual reasons

The key insight: once surfaces are pre-computed, the allocation step
is cheap — it's just reading gradients off the surfaces and solving
a 1D root-finding problem for the Lagrange multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import APYSensitiveConfig, MercenaryConfig, RetailDepositorConfig, WhaleProfile
from .engine import LossWeights, run_monte_carlo
from .optimizer import SurfaceGrid, SurfaceResult, optimize_surface
from .state import CampaignConfig, CampaignEnvironment

# ============================================================================
# VENUE SPECIFICATION
# ============================================================================


@dataclass
class VenueSpec:
    """
    Specification for a single venue in the multi-venue allocation.

    Users configure these — either from live data or manually.
    """

    name: str
    asset_symbol: str  # "PYUSD" or "RLUSD"
    protocol: str  # "morpho", "euler", "aave", "curve", "kamino"

    # Current state
    current_tvl: float  # Current TVL (USD)
    current_utilization: float  # Current utilization (decimal, 0.40 = 40%)

    # Targets (user-specified)
    target_tvl: float  # Desired TVL
    target_utilization: float  # Desired utilization (for net supply calc)
    target_incentive_rate: float  # Target APR (decimal, 0.065 = 6.5%)

    # Base / organic yield for this venue
    base_apy: float = 0.0  # decimal (0.03 = 3%)

    # Budget constraints
    budget_min: float = 0.0  # Minimum weekly budget (contractual floor)
    budget_max: float = float("inf")  # Maximum weekly budget

    # Campaign constraints
    r_max_min: float = 0.04  # Minimum APR cap to explore
    r_max_max: float = 0.12  # Maximum APR cap to explore

    # Pinning: if set, these override the optimizer for this venue
    pinned_budget: float | None = None  # Fixed weekly budget
    pinned_r_max: float | None = None  # Fixed APR cap

    # Simulation parameters
    whale_profiles: list[WhaleProfile] = field(default_factory=list)
    retail_config: RetailDepositorConfig | None = None
    mercenary_config: MercenaryConfig | None = None
    apy_sensitive_config: APYSensitiveConfig | None = None
    env: CampaignEnvironment | None = None
    weights: LossWeights | None = None

    # Pre-computed surface (filled by allocator)
    surface_result: SurfaceResult | None = None

    @property
    def net_supply(self) -> float:
        """Net supply = TVL * (1 - utilization)."""
        return self.target_tvl * (1 - self.target_utilization)

    @property
    def implied_weekly_budget(self) -> float:
        """Budget implied by target TVL and incentive rate: B = TVL * r * 7/365."""
        return self.target_tvl * self.target_incentive_rate * 7 / 365

    @property
    def is_pinned(self) -> bool:
        return self.pinned_budget is not None

    @property
    def tvl_per_incentive_dollar(self) -> float:
        """TVL per dollar of weekly incentive spend."""
        b = self.implied_weekly_budget
        return self.target_tvl / b if b > 0 else 0


# ============================================================================
# ALLOCATION RESULT
# ============================================================================


@dataclass
class VenueAllocation:
    """Optimal allocation for a single venue."""

    name: str
    asset_symbol: str
    protocol: str

    # Allocated parameters
    weekly_budget: float  # B_i (USD/week)
    apr_cap: float  # r_max_i (decimal, incentive only)
    t_bind: float  # T_bind_i (USD)

    # Diagnostics from surface
    loss: float
    mean_apr: float  # TOTAL APR (base + incentive)
    mean_incentive_apr: float  # Incentive APR only
    base_apy: float  # Organic yield
    apr_range: tuple[float, float]  # (p5, p95) of TOTAL APR
    mean_tvl: float
    budget_utilization: float
    cascade_depth: float
    is_feasible: bool

    # Budget attribution
    budget_share: float  # fraction of total budget
    marginal_loss: float  # dL/dB at this point

    # User targets for reference
    target_tvl: float = 0
    target_incentive_rate: float = 0
    was_pinned: bool = False


@dataclass
class MultiVenueResult:
    """Complete result of multi-venue allocation."""

    total_budget: float
    allocations: list[VenueAllocation]
    total_loss: float
    lagrange_multiplier: float  # lambda* for the budget constraint

    # Per-venue surfaces (for dashboard)
    venue_surfaces: dict[str, SurfaceResult] = field(default_factory=dict)

    @property
    def budget_allocated(self) -> float:
        return sum(a.weekly_budget for a in self.allocations)

    @property
    def budget_remaining(self) -> float:
        return self.total_budget - self.budget_allocated

    def summary(self) -> str:
        lines = [
            "=" * 110,
            "MULTI-VENUE BUDGET ALLOCATION",
            "=" * 110,
            f"Total weekly budget: ${self.total_budget:,.0f}",
            f"Budget allocated:    ${self.budget_allocated:,.0f}",
            f"Budget remaining:    ${self.budget_remaining:,.0f}",
            f"Combined loss:       {self.total_loss:.4e}",
            f"Lagrange multiplier: {self.lagrange_multiplier:.4e}",
            "",
            f"{'Venue':<30} {'B/wk':>10} {'r_max':>8} {'Base':>6} {'Total APR':>10} "
            f"{'T_bind':>12} {'TVL':>12} {'Share':>7} {'Pin':>5}",
            "-" * 110,
        ]
        for a in self.allocations:
            lines.append(
                f"{a.name:<30} ${a.weekly_budget:>8,.0f} "
                f"{a.apr_cap:>7.2%} {a.base_apy:>5.1%} "
                f"{a.mean_apr:>9.2%} ${a.t_bind:>10,.0f} "
                f"${a.mean_tvl:>10,.0f} "
                f"{a.budget_share:>6.1%} {'*' if a.was_pinned else '':>5}"
            )
        lines.append("-" * 110)
        totals_tvl = sum(a.mean_tvl for a in self.allocations)
        lines.append(
            f"{'TOTAL':<30} ${self.budget_allocated:>8,.0f} "
            f"{'':>7} {'':>6} "
            f"{'':>9} {'':>12} "
            f"${totals_tvl:>10,.0f} "
            f"{'100.0%':>7}"
        )
        lines.append("")
        return "\n".join(lines)


# ============================================================================
# MULTI-VENUE ALLOCATOR
# ============================================================================


def compute_venue_surface(
    spec: VenueSpec,
    n_paths: int = 50,
    budget_steps: int = 12,
    r_max_steps: int = 13,
    base_seed: int = 42,
    verbose: bool = False,
) -> SurfaceResult:
    """
    Compute the loss surface for a single venue.

    Uses the venue's own parameters (whales, retail config, etc.)
    or defaults if not provided.
    """
    grid = SurfaceGrid.from_ranges(
        B_min=max(spec.budget_min, 10_000, spec.implied_weekly_budget * 0.3),
        B_max=min(spec.budget_max, spec.implied_weekly_budget * 3),
        B_steps=budget_steps,
        r_max_min=spec.r_max_min,
        r_max_max=spec.r_max_max,
        r_max_steps=r_max_steps,
        dt_days=0.5,
        horizon_days=28,
        base_apy=spec.base_apy,  # Propagate base APY to grid -> configs
    )

    env = spec.env or CampaignEnvironment(
        r_threshold=spec.target_incentive_rate * 0.7,
    )

    # Propagate APY-sensitive floor/sensitivity into loss weights if config provided
    _lc = spec.apy_sensitive_config
    weights = spec.weights or LossWeights(
        w_spend=1.0,
        w_apr_variance=3.0,
        w_apr_ceiling=5.0,
        w_tvl_shortfall=8.0,
        w_mercenary=6.0,
        w_whale_proximity=6.0,
        w_apr_floor=7.0,
        apr_target=spec.base_apy + spec.target_incentive_rate,  # Total APR target
        apr_ceiling=0.10,
        tvl_target=spec.target_tvl,
        apr_stability_on_total=True,
        apr_floor=_lc.floor_apr if _lc else 0.0,
        apr_floor_sensitivity=_lc.sensitivity if _lc else 0.0,
    )

    result = optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=spec.current_tvl,
        whale_profiles=spec.whale_profiles,
        weights=weights,
        n_paths=n_paths,
        retail_config=spec.retail_config,
        mercenary_config=spec.mercenary_config,
        apy_sensitive_config=spec.apy_sensitive_config,
        base_seed=base_seed,
        verbose=verbose,
    )

    return result


def _marginal_loss_at_budget(
    surface: SurfaceResult,
    B: float,
    r_max: float | None = None,
) -> float:
    """
    Estimate dL/dB at a given budget level on the surface.

    If r_max is None, uses the optimal r_max for each B slice.
    Uses finite differences on the pre-computed surface.
    """
    B_vals = surface.grid.B_values
    r_vals = surface.grid.r_max_values
    L = surface.loss_surface
    feas = surface.feasibility_mask

    # Find the B index closest to requested B
    i = int(np.argmin(np.abs(B_vals - B)))

    if r_max is not None:
        j = int(np.argmin(np.abs(r_vals - r_max)))
    else:
        # Find optimal r_max for this B slice
        row = np.where(feas[i], L[i], np.inf)
        j = int(np.argmin(row))

    # Centered finite difference for dL/dB
    if i == 0:
        dL = L[min(i + 1, L.shape[0] - 1), j] - L[i, j]
        dB = B_vals[min(i + 1, len(B_vals) - 1)] - B_vals[i]
    elif i == L.shape[0] - 1:
        dL = L[i, j] - L[max(i - 1, 0), j]
        dB = B_vals[i] - B_vals[max(i - 1, 0)]
    else:
        dL = L[i + 1, j] - L[i - 1, j]
        dB = B_vals[i + 1] - B_vals[i - 1]

    return dL / dB if abs(dB) > 0 else 0.0


def _optimal_at_budget(
    surface: SurfaceResult,
    B: float,
) -> tuple[float, float, int, int]:
    """
    Find the optimal r_max and loss for a given budget level.

    Returns (r_max*, loss*, i, j) indices.
    """
    B_vals = surface.grid.B_values
    r_vals = surface.grid.r_max_values
    L = surface.loss_surface
    feas = surface.feasibility_mask

    i = int(np.argmin(np.abs(B_vals - B)))
    row = np.where(feas[i], L[i], np.inf)
    j = int(np.argmin(row))

    return float(r_vals[j]), float(L[i, j]), i, j


def allocate_budget(
    venues: list[VenueSpec],
    total_budget: float,
    n_paths: int = 50,
    budget_steps: int = 12,
    r_max_steps: int = 13,
    base_seed: int = 42,
    verbose: bool = True,
) -> MultiVenueResult:
    """
    Allocate total weekly budget across N venues.

    Algorithm (Lagrangian with marginal equalization):
    1. Compute loss surface for each non-pinned venue
    2. Subtract pinned budgets from total
    3. For remaining budget, find lambda* such that:
       sum_i B_i(lambda) = B_remaining
       where B_i(lambda) satisfies dL_i/dB_i = lambda for each venue
    4. This is a 1D root-finding problem (bisection on lambda)

    Pinned venues are excluded from optimization — their budget is fixed,
    and the remaining venues split what's left.
    """
    if verbose:
        print("=" * 80)
        print(f"MULTI-VENUE ALLOCATION: {len(venues)} venues, ${total_budget:,.0f}/week total")
        print("=" * 80)

    # ── Step 1: Separate pinned vs optimizable venues ──
    pinned = [v for v in venues if v.is_pinned]
    optimizable = [v for v in venues if not v.is_pinned]

    pinned_budget = sum(v.pinned_budget for v in pinned)
    remaining_budget = total_budget - pinned_budget

    if remaining_budget < 0:
        raise ValueError(
            f"Pinned budgets (${pinned_budget:,.0f}) exceed total budget (${total_budget:,.0f})"
        )

    if verbose:
        print(f"Pinned venues: {len(pinned)} (${pinned_budget:,.0f}/week)")
        print(f"Optimizable venues: {len(optimizable)} (${remaining_budget:,.0f}/week remaining)")

    # ── Step 2: Compute surfaces for all venues ──
    surfaces: dict[str, SurfaceResult] = {}

    for idx, spec in enumerate(venues):
        if verbose:
            print(f"\n[{idx + 1}/{len(venues)}] Computing surface: {spec.name}")
            if spec.base_apy > 0:
                print(f"  Base APY: {spec.base_apy:.2%}")

        if spec.is_pinned and spec.pinned_r_max is not None:
            # Fully pinned — run single-point MC instead of full surface
            cfg = CampaignConfig(
                weekly_budget=spec.pinned_budget,
                apr_cap=spec.pinned_r_max,
                base_apy=spec.base_apy,
            )
            env = spec.env or CampaignEnvironment(
                r_threshold=spec.target_incentive_rate * 0.7,
            )
            weights = spec.weights or LossWeights(
                tvl_target=spec.target_tvl,
                apr_target=spec.base_apy + spec.target_incentive_rate,
                apr_stability_on_total=True,
            )
            mc = run_monte_carlo(
                config=cfg,
                env=env,
                initial_tvl=spec.current_tvl,
                whale_profiles=spec.whale_profiles,
                weights=weights,
                n_paths=n_paths,
                retail_config=spec.retail_config,
                mercenary_config=spec.mercenary_config,
                apy_sensitive_config=spec.apy_sensitive_config,
                base_seed=base_seed + idx * 100000,
            )
            # Create a minimal 1x1 surface result
            grid = SurfaceGrid(
                B_values=np.array([spec.pinned_budget]),
                r_max_values=np.array([spec.pinned_r_max]),
                base_apy=spec.base_apy,
            )
            sr = SurfaceResult(
                grid=grid,
                loss_surface=np.array([[mc.mean_loss]]),
                loss_std_surface=np.array([[mc.std_loss]]),
                feasibility_mask=np.array([[mc.is_feasible]]),
                component_surfaces={k: np.array([[v]]) for k, v in mc.loss_components.items()},
                avg_apr_surface=np.array([[mc.mean_apr]]),
                avg_incentive_apr_surface=np.array([[mc.mean_incentive_apr]]),
                avg_tvl_surface=np.array([[mc.mean_tvl]]),
                budget_util_surface=np.array([[mc.mean_budget_util]]),
                cascade_depth_surface=np.array([[mc.mean_cascade_depth]]),
                mercenary_frac_surface=np.array([[mc.mean_mercenary_fraction]]),
                cap_binding_surface=np.array([[mc.mean_time_cap_binding]]),
                mc_results={(0, 0): mc},
            )
            surfaces[spec.name] = sr
        else:
            sr = compute_venue_surface(
                spec,
                n_paths=n_paths,
                budget_steps=budget_steps,
                r_max_steps=r_max_steps,
                base_seed=base_seed + idx * 100000,
                verbose=verbose,
            )
            surfaces[spec.name] = sr
            spec.surface_result = sr

    # ── Step 3: Allocate remaining budget via marginal equalization ──
    if not optimizable or remaining_budget <= 0:
        # All pinned or no budget left — just report
        allocations = _build_allocations(venues, surfaces, {})
        return MultiVenueResult(
            total_budget=total_budget,
            allocations=allocations,
            total_loss=sum(a.loss for a in allocations),
            lagrange_multiplier=0.0,
            venue_surfaces=surfaces,
        )

    # Find marginal loss range across venues
    marginals = []
    for spec in optimizable:
        sr = surfaces[spec.name]
        B_lo = sr.grid.B_values[0]
        B_hi = sr.grid.B_values[-1]
        m_lo = _marginal_loss_at_budget(sr, B_lo)
        m_hi = _marginal_loss_at_budget(sr, B_hi)
        marginals.extend([m_lo, m_hi])

    lambda_lo = min(marginals) * 2  # More negative = more benefit from spending
    lambda_hi = max(marginals) * 0.5

    # Ensure lambda_lo < lambda_hi
    if lambda_lo >= lambda_hi:
        lambda_lo, lambda_hi = lambda_hi - abs(lambda_hi), lambda_lo + abs(lambda_lo)

    def _budget_for_lambda(lam: float) -> float:
        """Total budget consumed when each venue sets dL/dB = lambda."""
        total = 0.0
        for spec in optimizable:
            sr = surfaces[spec.name]
            # Find the B where marginal loss ~ lambda
            best_B = sr.grid.B_values[0]
            best_dist = float("inf")
            for B_val in sr.grid.B_values:
                m = _marginal_loss_at_budget(sr, B_val)
                dist = abs(m - lam)
                if dist < best_dist:
                    best_dist = dist
                    best_B = B_val
            # Clamp to venue constraints
            best_B = max(spec.budget_min, min(spec.budget_max, best_B))
            total += best_B
        return total

    # Bisection to find lambda* such that sum B_i(lambda*) = remaining_budget
    for _ in range(50):
        lambda_mid = (lambda_lo + lambda_hi) / 2
        consumed = _budget_for_lambda(lambda_mid)
        if consumed > remaining_budget:
            lambda_hi = lambda_mid  # Spend less -> need higher (less negative) lambda
        else:
            lambda_lo = lambda_mid
        if abs(consumed - remaining_budget) < remaining_budget * 0.01:
            break

    lambda_star = (lambda_lo + lambda_hi) / 2

    # ── Step 4: Read off final allocations ──
    budget_map: dict[str, float] = {}
    for spec in optimizable:
        sr = surfaces[spec.name]
        best_B = sr.grid.B_values[0]
        best_dist = float("inf")
        for B_val in sr.grid.B_values:
            m = _marginal_loss_at_budget(sr, B_val)
            dist = abs(m - lambda_star)
            if dist < best_dist:
                best_dist = dist
                best_B = B_val
        best_B = max(spec.budget_min, min(spec.budget_max, best_B))
        budget_map[spec.name] = best_B

    # Scale to exactly match remaining budget
    alloc_total = sum(budget_map.values())
    if alloc_total > 0:
        scale = remaining_budget / alloc_total
        budget_map = {k: v * scale for k, v in budget_map.items()}

    allocations = _build_allocations(venues, surfaces, budget_map)

    result = MultiVenueResult(
        total_budget=total_budget,
        allocations=allocations,
        total_loss=sum(a.loss for a in allocations),
        lagrange_multiplier=lambda_star,
        venue_surfaces=surfaces,
    )

    if verbose:
        print(result.summary())

    return result


def _build_allocations(
    venues: list[VenueSpec],
    surfaces: dict[str, SurfaceResult],
    budget_map: dict[str, float],
) -> list[VenueAllocation]:
    """Build VenueAllocation objects from surfaces and budget assignments."""
    total_B = sum(v.pinned_budget if v.is_pinned else budget_map.get(v.name, 0) for v in venues)

    allocations = []
    for spec in venues:
        sr = surfaces[spec.name]

        if spec.is_pinned:
            B = spec.pinned_budget
            r_max = spec.pinned_r_max or sr.optimal_r_max
        else:
            B = budget_map.get(spec.name, 0)
            r_max, loss, i, j = _optimal_at_budget(sr, B)

        # Get diagnostics at this (B, r_max) point
        B_vals = sr.grid.B_values
        r_vals = sr.grid.r_max_values
        i = int(np.argmin(np.abs(B_vals - B)))
        j = int(np.argmin(np.abs(r_vals - r_max)))

        mc = sr.mc_results.get((i, j))

        t_bind = B * (365.0 / 7.0) / r_max if r_max > 0 else float("inf")
        marginal = _marginal_loss_at_budget(sr, B, r_max) if sr.grid.shape[0] > 1 else 0.0

        allocations.append(
            VenueAllocation(
                name=spec.name,
                asset_symbol=spec.asset_symbol,
                protocol=spec.protocol,
                weekly_budget=B,
                apr_cap=r_max,
                t_bind=t_bind,
                loss=float(sr.loss_surface[i, j]),
                mean_apr=float(sr.avg_apr_surface[i, j]) if sr.avg_apr_surface.size > 0 else 0,
                mean_incentive_apr=(
                    float(sr.avg_incentive_apr_surface[i, j])
                    if sr.avg_incentive_apr_surface.size > 0
                    else 0
                ),
                base_apy=sr.grid.base_apy,
                apr_range=((mc.apr_p5, mc.apr_p95) if mc else (0, 0)),
                mean_tvl=float(sr.avg_tvl_surface[i, j]) if sr.avg_tvl_surface.size > 0 else 0,
                budget_utilization=mc.mean_budget_util if mc else 0,
                cascade_depth=mc.mean_cascade_depth if mc else 0,
                is_feasible=bool(sr.feasibility_mask[i, j]),
                budget_share=B / total_B if total_B > 0 else 0,
                marginal_loss=marginal,
                target_tvl=spec.target_tvl,
                target_incentive_rate=spec.target_incentive_rate,
                was_pinned=spec.is_pinned,
            )
        )

    return allocations

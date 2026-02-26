"""
Surface optimizer for campaign parameter selection.

Outer loop: grid search over (B, r_max) surface, running Monte Carlo
at each point, then identifying the optimal configuration.

Includes:
- Grid construction
- Parallel execution
- Hessian / sensitivity analysis at the optimum
- Duality map (which configurations achieve similar loss)
- Visualization helpers
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .agents import (
    APYSensitiveConfig,
    MercenaryConfig,
    RetailDepositorConfig,
    WhaleProfile,
)
from .engine import (
    LossWeights,
    MonteCarloResult,
    run_monte_carlo,
)
from .state import CampaignConfig, CampaignEnvironment

# ============================================================================
# GRID DEFINITION
# ============================================================================


@dataclass
class SurfaceGrid:
    """
    2D grid over the (B, r_max) parameter surface.

    r_max values are INCENTIVE APR caps (not total APR).
    base_apy is the venue's organic yield, added on top.
    """

    B_values: np.ndarray  # Weekly budget values (USD)
    r_max_values: np.ndarray  # APR cap values (decimal, INCENTIVE only)

    # Fixed campaign parameters
    epoch_duration_days: int = 7
    merkl_fee_rate: float = 0.015
    dt_days: float = 0.25
    horizon_days: int = 28
    base_apy: float = 0.0  # Venue's organic yield (passed to CampaignConfig)
    supply_cap: float = 0.0  # USD (0 = unlimited) — passed through to CampaignConfig

    @classmethod
    def from_ranges(
        cls,
        B_min: float,
        B_max: float,
        B_steps: int,
        r_max_min: float,
        r_max_max: float,
        r_max_steps: int,
        **kwargs,
    ) -> SurfaceGrid:
        """
        Create grid from ranges.

        Args:
            B_min, B_max: Budget range (USD/week)
            B_steps: Number of budget grid points
            r_max_min, r_max_max: APR cap range (decimal, INCENTIVE only)
            r_max_steps: Number of APR cap grid points
        """
        return cls(
            B_values=np.linspace(B_min, B_max, B_steps),
            r_max_values=np.linspace(r_max_min, r_max_max, r_max_steps),
            **kwargs,
        )

    @classmethod
    def from_t_bind_centered(
        cls,
        current_tvl: float,
        B_center: float,
        B_half_range: float,
        B_steps: int,
        t_bind_min_frac: float = 0.5,
        t_bind_max_frac: float = 1.2,
        r_max_steps: int = 20,
        **kwargs,
    ) -> SurfaceGrid:
        """
        Create grid centered on expected operating point, with r_max
        derived from T_bind fractions of current TVL.

        This parameterization is more natural: instead of specifying
        r_max directly, you specify what fraction of current TVL should
        trigger the regime switch.
        """
        B_values = np.linspace(B_center - B_half_range, B_center + B_half_range, B_steps)

        t_bind_values = np.linspace(
            current_tvl * t_bind_min_frac,
            current_tvl * t_bind_max_frac,
            r_max_steps,
        )
        r_max_values = B_center * (365.0 / 7.0) / t_bind_values
        r_max_values = r_max_values[::-1]

        return cls(B_values=B_values, r_max_values=r_max_values, **kwargs)

    def make_config(
        self, B: float, r_max: float, whale_profiles: list | None = None
    ) -> CampaignConfig:
        """Create CampaignConfig for a grid point, including base_apy, supply_cap, and whale_profiles."""
        return CampaignConfig(
            weekly_budget=B,
            apr_cap=r_max,
            base_apy=self.base_apy,
            supply_cap=self.supply_cap,
            epoch_duration_days=self.epoch_duration_days,
            merkl_fee_rate=self.merkl_fee_rate,
            dt_days=self.dt_days,
            horizon_days=self.horizon_days,
            whale_profiles=tuple(whale_profiles) if whale_profiles else (),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.B_values), len(self.r_max_values))

    def t_bind_surface(self) -> np.ndarray:
        """Compute T_bind at each grid point."""
        B_grid, r_grid = np.meshgrid(self.B_values, self.r_max_values, indexing="ij")
        return np.where(r_grid > 0, B_grid * (365.0 / 7.0) / r_grid, np.inf)


# ============================================================================
# SURFACE OPTIMIZATION RESULT
# ============================================================================


@dataclass
class SurfaceResult:
    """Complete result of surface optimization."""

    grid: SurfaceGrid

    # Loss surface: shape (n_B, n_r_max)
    loss_surface: np.ndarray
    loss_std_surface: np.ndarray
    feasibility_mask: np.ndarray  # True = feasible

    # Component surfaces
    component_surfaces: dict[str, np.ndarray] = field(default_factory=dict)

    # Diagnostic surfaces
    avg_apr_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_incentive_apr_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_tvl_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    budget_util_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    cascade_depth_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    mercenary_frac_surface: np.ndarray = field(default_factory=lambda: np.array([]))
    cap_binding_surface: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-point detailed results
    mc_results: dict[tuple[int, int], MonteCarloResult] = field(default_factory=dict)

    @property
    def optimal_indices(self) -> tuple[int, int]:
        """Indices of optimal (feasible, minimum loss) point."""
        masked = np.where(self.feasibility_mask, self.loss_surface, np.inf)
        return np.unravel_index(np.argmin(masked), masked.shape)

    @property
    def optimal_B(self) -> float:
        i, _ = self.optimal_indices
        return float(self.grid.B_values[i])

    @property
    def optimal_r_max(self) -> float:
        _, j = self.optimal_indices
        return float(self.grid.r_max_values[j])

    @property
    def optimal_t_bind(self) -> float:
        if self.optimal_r_max <= 0:
            return float("inf")
        return self.optimal_B * (365.0 / 7.0) / self.optimal_r_max

    @property
    def optimal_loss(self) -> float:
        i, j = self.optimal_indices
        return float(self.loss_surface[i, j])

    @property
    def optimal_mc_result(self) -> MonteCarloResult | None:
        return self.mc_results.get(self.optimal_indices)

    def hessian_at_optimum(self) -> np.ndarray:
        """
        Numerical Hessian of loss surface at the optimum.

        Returns:
            2x2 Hessian matrix [[d²L/dB², d²L/dBdr], [d²L/dBdr, d²L/dr²]]
        """
        i, j = self.optimal_indices
        L = self.loss_surface
        dB = self.grid.B_values[1] - self.grid.B_values[0] if len(self.grid.B_values) > 1 else 1
        dr = (
            self.grid.r_max_values[1] - self.grid.r_max_values[0]
            if len(self.grid.r_max_values) > 1
            else 1
        )

        def _L(ii, jj):
            ii = max(0, min(ii, L.shape[0] - 1))
            jj = max(0, min(jj, L.shape[1] - 1))
            return L[ii, jj]

        d2L_dB2 = (_L(i + 1, j) - 2 * _L(i, j) + _L(i - 1, j)) / (dB**2)
        d2L_dr2 = (_L(i, j + 1) - 2 * _L(i, j) + _L(i, j - 1)) / (dr**2)
        d2L_dBdr = (_L(i + 1, j + 1) - _L(i + 1, j - 1) - _L(i - 1, j + 1) + _L(i - 1, j - 1)) / (
            4 * dB * dr
        )

        return np.array([[d2L_dB2, d2L_dBdr], [d2L_dBdr, d2L_dr2]])

    def sensitivity_analysis(self) -> dict:
        """
        Eigendecomposition of Hessian at optimum.

        Returns dict with:
        - eigenvalues: [λ_steep, λ_flat]
        - eigenvectors: [[steep_direction], [flat_direction]]
        - condition_number: |λ_max/λ_min|
        - interpretation: string describing what matters
        """
        H = self.hessian_at_optimum()
        eigvals, eigvecs = np.linalg.eigh(H)

        order = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        condition = abs(eigvals[0] / eigvals[1]) if abs(eigvals[1]) > 1e-12 else float("inf")

        steep_dir = eigvecs[:, 0]
        b_importance = abs(steep_dir[0])
        r_importance = abs(steep_dir[1])

        if b_importance > 2 * r_importance:
            interp = "Budget (B) is the critical parameter; APR cap (r_max) is secondary"
        elif r_importance > 2 * b_importance:
            interp = "APR cap (r_max) is the critical parameter; budget (B) is secondary"
        else:
            interp = "Both B and r_max matter roughly equally"

        return {
            "eigenvalues": eigvals.tolist(),
            "eigenvectors": eigvecs.tolist(),
            "condition_number": condition,
            "steep_direction": steep_dir.tolist(),
            "flat_direction": eigvecs[:, 1].tolist(),
            "interpretation": interp,
        }

    def duality_map(self, tolerance: float = 0.05) -> list[dict]:
        """
        Find configurations achieving loss within tolerance of optimum.
        """
        opt_loss = self.optimal_loss
        threshold = opt_loss * (1 + tolerance)

        near_optimal = []
        for i in range(len(self.grid.B_values)):
            for j in range(len(self.grid.r_max_values)):
                if not self.feasibility_mask[i, j]:
                    continue
                loss = self.loss_surface[i, j]
                if loss <= threshold:
                    B = self.grid.B_values[i]
                    r = self.grid.r_max_values[j]
                    near_optimal.append(
                        {
                            "B": float(B),
                            "r_max": float(r),
                            "t_bind": float(B * (365.0 / 7.0) / r) if r > 0 else float("inf"),
                            "loss": float(loss),
                            "loss_ratio": float(loss / opt_loss) if opt_loss > 0 else float("inf"),
                            "grid_indices": (i, j),
                        }
                    )

        return sorted(near_optimal, key=lambda x: x["loss"])

    def stability_boundary(self) -> list[tuple[float, float]]:
        """
        Extract the feasibility boundary on the (B, r_max) surface.
        """
        boundary = []
        mask = self.feasibility_mask
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if not mask[i, j]:
                    continue
                neighbors = [
                    (i - 1, j),
                    (i + 1, j),
                    (i, j - 1),
                    (i, j + 1),
                ]
                for ni, nj in neighbors:
                    if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
                        if not mask[ni, nj]:
                            boundary.append(
                                (
                                    float(self.grid.B_values[i]),
                                    float(self.grid.r_max_values[j]),
                                )
                            )
                            break
        return boundary

    def summary(self) -> str:
        """Human-readable summary of optimization result."""
        sa = self.sensitivity_analysis()
        dual = self.duality_map()
        base = self.grid.base_apy

        lines = [
            "=" * 60,
            "CAMPAIGN SURFACE OPTIMIZATION RESULT",
            "=" * 60,
            "",
            f"Grid: {self.grid.shape[0]} B values x {self.grid.shape[1]} r_max values",
            f"Base APY: {base:.2%}",
            f"Feasible points: {int(self.feasibility_mask.sum())} / {self.feasibility_mask.size}",
            "",
            "--Optimal Configuration --",
            f"  B* (weekly budget):  ${self.optimal_B:,.0f}",
            f"  r_max* (APR cap):    {self.optimal_r_max:.2%} (incentive only)",
            f"  Total APR at cap:    {self.optimal_r_max + base:.2%} (base {base:.2%} + cap {self.optimal_r_max:.2%})",
            f"  T_bind*:             ${self.optimal_t_bind:,.0f}",
            f"  Loss:                {self.optimal_loss:.4e}",
            "",
        ]

        mc = self.optimal_mc_result
        if mc:
            lines.extend(
                [
                    "--Optimal Point Diagnostics --",
                    f"  Mean total APR:      {mc.mean_apr:.2%} (base {base:.2%} + incentive {mc.mean_incentive_apr:.2%})",
                    f"  Mean incentive APR:  {mc.mean_incentive_apr:.2%}",
                    f"  APR range (p5-p95):  [{mc.apr_p5:.2%}, {mc.apr_p95:.2%}] (total)",
                    f"  Mean TVL:            ${mc.mean_tvl:,.0f}",
                    f"  TVL min (p5):        ${mc.tvl_min_p5:,.0f}",
                    f"  Budget utilization:  {mc.mean_budget_util:.1%}",
                    f"  Mean cascade depth:  {mc.mean_cascade_depth:.1f}",
                    f"  Max cascade depth:   {mc.max_cascade_depth}",
                    f"  Mercenary fraction:  {mc.mean_mercenary_fraction:.1%}",
                    f"  Time cap binding:    {mc.mean_time_cap_binding:.1%}",
                    "",
                ]
            )

        lines.extend(
            [
                "--Sensitivity Analysis --",
                f"  Condition number:    {sa['condition_number']:.1f}",
                f"  Interpretation:      {sa['interpretation']}",
                f"  Steep direction:     [{sa['steep_direction'][0]:.3f}, {sa['steep_direction'][1]:.3f}]",
                f"  Flat direction:      [{sa['flat_direction'][0]:.3f}, {sa['flat_direction'][1]:.3f}]",
                "",
                "--Duality Map (within 5% of optimum) --",
                f"  {len(dual)} near-optimal configurations",
            ]
        )

        if dual:
            t_binds = [d["t_bind"] for d in dual]
            lines.append(f"  T_bind range:        ${min(t_binds):,.0f} - ${max(t_binds):,.0f}")
            B_range = [d["B"] for d in dual]
            r_range = [d["r_max"] for d in dual]
            lines.append(f"  B range:             ${min(B_range):,.0f} — ${max(B_range):,.0f}")
            lines.append(f"  r_max range:         {min(r_range):.2%} — {max(r_range):.2%}")

        lines.append("")
        return "\n".join(lines)


# ============================================================================
# SURFACE OPTIMIZER (main entry point)
# ============================================================================


def optimize_surface(
    grid: SurfaceGrid,
    env: CampaignEnvironment,
    initial_tvl: float,
    whale_profiles: list[WhaleProfile],
    weights: LossWeights,
    n_paths: int = 100,
    retail_config: RetailDepositorConfig | None = None,
    mercenary_config: MercenaryConfig | None = None,
    apy_sensitive_config: APYSensitiveConfig | None = None,
    base_seed: int = 42,
    cascade_tolerance: int = 3,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SurfaceResult:
    """
    Run full surface optimization.

    For each (B, r_max) on the grid, runs M Monte Carlo paths and
    evaluates the loss functional. Returns the complete loss surface
    with optimal point, sensitivity analysis, and duality map.
    """
    n_B, n_r = grid.shape
    total = n_B * n_r

    loss_surface = np.full((n_B, n_r), np.inf)
    loss_std_surface = np.full((n_B, n_r), np.inf)
    feasibility_mask = np.zeros((n_B, n_r), dtype=bool)

    component_keys = ["spend", "apr_variance", "apr_ceiling", "tvl_shortfall", "merkl_fee"]
    component_surfaces = {k: np.full((n_B, n_r), np.inf) for k in component_keys}
    avg_apr_surface = np.zeros((n_B, n_r))
    avg_incentive_apr_surface = np.zeros((n_B, n_r))
    avg_tvl_surface = np.zeros((n_B, n_r))
    budget_util_surface = np.zeros((n_B, n_r))
    cascade_depth_surface = np.zeros((n_B, n_r))
    mercenary_frac_surface = np.zeros((n_B, n_r))
    cap_binding_surface = np.zeros((n_B, n_r))

    mc_results = {}
    count = 0

    if verbose and grid.base_apy > 0:
        print(f"  Base APY for this venue: {grid.base_apy:.2%}")

    for i, B in enumerate(grid.B_values):
        for j, r_max in enumerate(grid.r_max_values):
            count += 1

            if verbose and count % max(1, total // 20) == 0:
                if grid.base_apy > 0:
                    print(
                        f"  [{count}/{total}] B=${B:,.0f}, r_max={r_max:.2%} (total at cap: {r_max + grid.base_apy:.2%})"
                    )
                else:
                    print(f"  [{count}/{total}] B=${B:,.0f}, r_max={r_max:.2%}")

            if progress_callback:
                progress_callback(count, total)

            config = grid.make_config(B, r_max, whale_profiles=whale_profiles)

            mc = run_monte_carlo(
                config=config,
                env=env,
                initial_tvl=initial_tvl,
                whale_profiles=whale_profiles,
                weights=weights,
                n_paths=n_paths,
                retail_config=retail_config,
                mercenary_config=mercenary_config,
                apy_sensitive_config=apy_sensitive_config,
                base_seed=base_seed + count * 10000,
                cascade_tolerance=cascade_tolerance,
            )

            mc_results[(i, j)] = mc
            loss_surface[i, j] = mc.mean_loss
            loss_std_surface[i, j] = mc.std_loss
            feasibility_mask[i, j] = mc.is_feasible

            for k in component_keys:
                component_surfaces[k][i, j] = mc.loss_components.get(k, np.inf)

            avg_apr_surface[i, j] = mc.mean_apr
            avg_incentive_apr_surface[i, j] = mc.mean_incentive_apr
            avg_tvl_surface[i, j] = mc.mean_tvl
            budget_util_surface[i, j] = mc.mean_budget_util
            cascade_depth_surface[i, j] = mc.mean_cascade_depth
            mercenary_frac_surface[i, j] = mc.mean_mercenary_fraction
            cap_binding_surface[i, j] = mc.mean_time_cap_binding

    result = SurfaceResult(
        grid=grid,
        loss_surface=loss_surface,
        loss_std_surface=loss_std_surface,
        feasibility_mask=feasibility_mask,
        component_surfaces=component_surfaces,
        avg_apr_surface=avg_apr_surface,
        avg_incentive_apr_surface=avg_incentive_apr_surface,
        avg_tvl_surface=avg_tvl_surface,
        budget_util_surface=budget_util_surface,
        cascade_depth_surface=cascade_depth_surface,
        mercenary_frac_surface=mercenary_frac_surface,
        cap_binding_surface=cap_binding_surface,
        mc_results=mc_results,
    )

    if verbose:
        print(result.summary())

    return result

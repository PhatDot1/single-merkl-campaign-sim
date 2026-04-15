"""
Grid search over incentive allocations with Monte Carlo profit evaluation.

For each candidate allocation (I_supply, I_borrow) on a grid:
  1. Draw M samples from posterior distributions of (beta_s, beta_b, delta)
  2. Simulate TVL trajectory forward H days under each sample
  3. Compute expected profit = E[NII] - I_supply - I_borrow
  4. Select the allocation that maximizes expected profit

The grid spans current allocation +/- a configurable range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bayesian_incentives.models.elasticity import BayesianPosterior
from bayesian_incentives.models.interest_rate import InterestRateModel


@dataclass
class GridPoint:
    """Single evaluation point on the incentive grid."""

    i_supply: float
    i_borrow: float
    expected_profit: float
    profit_std: float
    profit_5th: float
    profit_95th: float


@dataclass
class GridSearchResult:
    """Full grid search output."""

    grid: list[GridPoint]
    optimal: GridPoint
    grid_shape: tuple[int, int]
    n_mc_samples: int
    horizon_days: int

    def profit_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (supply_vals, borrow_vals, profit_2d) for plotting."""
        n_s, n_b = self.grid_shape
        supply_vals = np.array([g.i_supply for g in self.grid[:n_s * n_b:n_b]])
        borrow_vals = np.array([g.i_borrow for g in self.grid[:n_b]])
        profit_2d = np.array([g.expected_profit for g in self.grid]).reshape(
            n_s, n_b
        )
        return supply_vals, borrow_vals, profit_2d

    def summary(self) -> str:
        return (
            f"Grid search: {self.grid_shape[0]}x{self.grid_shape[1]} points, "
            f"{self.n_mc_samples} MC samples, {self.horizon_days}-day horizon\n"
            f"Optimal: I_supply=${self.optimal.i_supply:,.2f}, "
            f"I_borrow=${self.optimal.i_borrow:,.2f}\n"
            f"  E[profit]=${self.optimal.expected_profit:,.2f}/day "
            f"(std=${self.optimal.profit_std:,.2f})\n"
            f"  5th-95th: [${self.optimal.profit_5th:,.2f}, "
            f"${self.optimal.profit_95th:,.2f}]"
        )


class GridSearchOptimizer:
    """Evaluate expected profit over a 2D incentive grid using Monte Carlo.

    Parameters
    ----------
    rate_model : InterestRateModel or None
        Interest rate model for NII computation. If None, uses defaults.
    n_mc : int
        Number of Monte Carlo samples per grid point (default 2000).
    horizon : int
        Projection horizon in days (default 30).
    """

    def __init__(
        self,
        rate_model: InterestRateModel | None = None,
        n_mc: int = 2000,
        horizon: int = 30,
    ) -> None:
        self.rate_model = rate_model or InterestRateModel()
        self.n_mc = n_mc
        self.horizon = horizon

    def _simulate_tvl(
        self,
        current_tvl: float,
        current_i_supply: float,
        current_i_borrow: float,
        new_i_supply: float,
        new_i_borrow: float,
        beta_s: np.ndarray,
        beta_b: np.ndarray,
        delta: np.ndarray,
        equilibrium: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate TVL trajectories under a new incentive allocation.

        S_{t+h} = mu + (1-delta)^h * (S_t - mu)
                  + sum_{j=1}^{h} (1-delta)^{h-j} * (beta_s*dI_s + beta_b*dI_b)

        Returns array of shape (n_mc,) with terminal TVL values.
        """
        M = len(beta_s)
        d_is = new_i_supply - current_i_supply
        d_ib = new_i_borrow - current_i_borrow

        decay_factor = (1 - delta) ** self.horizon  # (M,)

        # Mean-reversion component
        tvl_mean_revert = equilibrium + decay_factor * (current_tvl - equilibrium)

        # Incentive impulse accumulated over horizon
        # sum_{j=1}^{H} (1-delta)^{H-j} = ((1-(1-delta)^H) / delta)
        safe_delta = np.where(delta > 1e-10, delta, 1e-10)
        impulse_sum = (1 - (1 - delta) ** self.horizon) / safe_delta
        incentive_effect = impulse_sum * (beta_s * d_is + beta_b * d_ib)

        terminal_tvl = tvl_mean_revert + incentive_effect

        # Ensure non-negative
        return np.maximum(terminal_tvl, 0.0)

    def _compute_profit(
        self,
        tvl: np.ndarray,
        utilization: float,
        i_supply: float,
        i_borrow: float,
    ) -> np.ndarray:
        """Compute daily profit for each MC sample.

        profit = NII(S, U) - I_supply - I_borrow
        """
        nii = np.array([
            self.rate_model.net_interest_income(s, utilization, daily=True)
            for s in tvl
        ])
        return nii - i_supply - i_borrow

    def optimize(
        self,
        current_tvl: float,
        current_utilization: float,
        current_i_supply: float,
        current_i_borrow: float,
        posterior_beta_s: BayesianPosterior,
        posterior_beta_b: BayesianPosterior,
        decay_alpha: float,
        decay_beta: float,
        equilibrium_tvl: float,
        grid_n: int = 20,
        supply_range: tuple[float, float] | None = None,
        borrow_range: tuple[float, float] | None = None,
        seed: int = 42,
    ) -> GridSearchResult:
        """Run grid search optimization.

        Parameters
        ----------
        current_tvl : float
            Current TVL level.
        current_utilization : float
            Current utilization ratio.
        current_i_supply, current_i_borrow : float
            Current daily incentive allocation.
        posterior_beta_s, posterior_beta_b : BayesianPosterior
            Posterior distributions for supply/borrow elasticity.
        decay_alpha, decay_beta : float
            Beta distribution parameters for the decay rate prior.
        equilibrium_tvl : float
            Long-run equilibrium TVL from AR(1) model.
        grid_n : int
            Grid resolution per dimension (default 20 -> 20x20 grid).
        supply_range, borrow_range : tuple, optional
            (min, max) for supply/borrow incentive grid.
            Defaults to [0, 2 * current].
        seed : int
            Random seed.

        Returns
        -------
        GridSearchResult
        """
        rng = np.random.default_rng(seed)

        # Default ranges: 0 to 2x current (or at least $100)
        if supply_range is None:
            supply_range = (0.0, max(2 * current_i_supply, 100.0))
        if borrow_range is None:
            borrow_range = (0.0, max(2 * current_i_borrow, 100.0))

        supply_vals = np.linspace(supply_range[0], supply_range[1], grid_n)
        borrow_vals = np.linspace(borrow_range[0], borrow_range[1], grid_n)

        # Draw posterior samples once
        beta_s_samples = posterior_beta_s.sample(self.n_mc, rng=rng)
        beta_b_samples = posterior_beta_b.sample(self.n_mc, rng=rng)
        delta_samples = rng.beta(decay_alpha, decay_beta, size=self.n_mc)

        grid_points: list[GridPoint] = []

        for i_s in supply_vals:
            for i_b in borrow_vals:
                terminal_tvl = self._simulate_tvl(
                    current_tvl=current_tvl,
                    current_i_supply=current_i_supply,
                    current_i_borrow=current_i_borrow,
                    new_i_supply=i_s,
                    new_i_borrow=i_b,
                    beta_s=beta_s_samples,
                    beta_b=beta_b_samples,
                    delta=delta_samples,
                    equilibrium=equilibrium_tvl,
                    rng=rng,
                )

                profits = self._compute_profit(
                    terminal_tvl, current_utilization, i_s, i_b
                )

                grid_points.append(
                    GridPoint(
                        i_supply=i_s,
                        i_borrow=i_b,
                        expected_profit=float(np.mean(profits)),
                        profit_std=float(np.std(profits)),
                        profit_5th=float(np.percentile(profits, 5)),
                        profit_95th=float(np.percentile(profits, 95)),
                    )
                )

        optimal = max(grid_points, key=lambda g: g.expected_profit)

        return GridSearchResult(
            grid=grid_points,
            optimal=optimal,
            grid_shape=(grid_n, grid_n),
            n_mc_samples=self.n_mc,
            horizon_days=self.horizon,
        )

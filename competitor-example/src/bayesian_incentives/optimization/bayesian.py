"""
Bayesian closed-loop incentive optimizer.

Implements the full iterative workflow:
  1. Posterior update (elasticity + decay)
  2. Monte Carlo sampling from posteriors
  3. Grid evaluation of expected profit
  4. Optimal allocation selection
  5. Observation of realized response
  6. Dataset augmentation and return to step 1

This is the top-level orchestrator that users interact with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from bayesian_incentives.models.elasticity import (
    BayesianPosterior,
    ElasticityModel,
    OLSResult,
)
from bayesian_incentives.models.decay import DecayModel, DecayResult
from bayesian_incentives.models.interest_rate import InterestRateModel, RateParams
from bayesian_incentives.optimization.grid_search import (
    GridSearchOptimizer,
    GridSearchResult,
)


@dataclass
class IterationResult:
    """Output of a single optimization iteration."""

    iteration: int
    ols_supply: OLSResult | None
    ols_borrow: OLSResult | None
    posterior_supply: BayesianPosterior
    posterior_borrow: BayesianPosterior
    decay: DecayResult
    grid_result: GridSearchResult
    optimal_supply: float
    optimal_borrow: float
    expected_profit: float
    converged: bool = False

    def summary(self) -> str:
        lines = [f"=== Iteration {self.iteration} ==="]

        if self.ols_supply:
            lines.append(f"Supply elasticity: beta={self.ols_supply.beta:,.2f}, "
                         f"R2={self.ols_supply.r_squared:.4%}, p={self.ols_supply.p_value:.4f}")
        if self.ols_borrow:
            lines.append(f"Borrow elasticity: beta={self.ols_borrow.beta:,.2f}, "
                         f"R2={self.ols_borrow.r_squared:.4%}, p={self.ols_borrow.p_value:.4f}")

        lines.append(
            f"Posteriors: beta_s ~ N({self.posterior_supply.mu:,.2f}, "
            f"{self.posterior_supply.sigma:,.2f}), "
            f"beta_b ~ N({self.posterior_borrow.mu:,.2f}, "
            f"{self.posterior_borrow.sigma:,.2f})"
        )
        lines.append(
            f"Decay: rho={self.decay.rho:.6f}, "
            f"half-life={self.decay.half_life:.1f} days"
        )
        lines.append(
            f"Optimal: I_supply=${self.optimal_supply:,.2f}, "
            f"I_borrow=${self.optimal_borrow:,.2f}"
        )
        lines.append(f"E[profit]=${self.expected_profit:,.2f}/day")
        if self.converged:
            lines.append("** CONVERGED **")
        return "\n".join(lines)


class BayesianOptimizer:
    """End-to-end Bayesian incentive optimizer with closed-loop feedback.

    Parameters
    ----------
    prior_beta_mu : float
        Prior mean for elasticity coefficients (default 0).
    prior_beta_sigma : float
        Prior std for elasticity coefficients (default 1e6).
    rate_params : RateParams or dict, optional
        Interest rate model parameters.
    n_mc : int
        Monte Carlo samples per grid point.
    horizon : int
        Profit projection horizon in days.
    grid_n : int
        Grid resolution per dimension.
    convergence_cv : float
        Convergence threshold for posterior coefficient of variation.
    convergence_profit_tol : float
        Convergence threshold for profit change between iterations.
    """

    def __init__(
        self,
        prior_beta_mu: float = 0.0,
        prior_beta_sigma: float = 1e6,
        rate_params: RateParams | dict | None = None,
        n_mc: int = 2000,
        horizon: int = 30,
        grid_n: int = 20,
        convergence_cv: float = 0.10,
        convergence_profit_tol: float = 10.0,
    ) -> None:
        self.prior_beta_mu = prior_beta_mu
        self.prior_beta_sigma = prior_beta_sigma
        self.rate_model = InterestRateModel(rate_params)
        self.n_mc = n_mc
        self.horizon = horizon
        self.grid_n = grid_n
        self.convergence_cv = convergence_cv
        self.convergence_profit_tol = convergence_profit_tol

        self._history: list[IterationResult] = []
        self._prev_profit: float | None = None

    @property
    def history(self) -> list[IterationResult]:
        """All past iteration results."""
        return self._history

    def run_iteration(
        self,
        df: pd.DataFrame,
        tvl_col: str = "tvl",
        incentive_supply_col: str | None = "incentive_supply",
        incentive_borrow_col: str | None = "incentive_borrow",
        incentive_total_col: str = "incentive_total",
        utilization_col: str | None = "utilization",
        supply_range: tuple[float, float] | None = None,
        borrow_range: tuple[float, float] | None = None,
        seed: int = 42,
    ) -> IterationResult:
        """Run one iteration of the closed-loop optimization.

        Parameters
        ----------
        df : pd.DataFrame
            Market data with at minimum tvl and incentive columns.
        tvl_col : str
            Column name for TVL.
        incentive_supply_col : str or None
            Column for supply-side incentives.
        incentive_borrow_col : str or None
            Column for borrow-side incentives.
        incentive_total_col : str
            Column for total incentives (used if supply/borrow not available).
        utilization_col : str or None
            Column for utilization. If None, defaults to 0.5.
        supply_range, borrow_range : tuple, optional
            Override grid range for supply/borrow incentive search.
        seed : int
            Random seed.

        Returns
        -------
        IterationResult
        """
        iteration = len(self._history) + 1

        tvl = df[tvl_col].values
        d_tvl = np.diff(tvl)

        # Determine if we have supply/borrow split
        has_split = (
            incentive_supply_col is not None
            and incentive_supply_col in df.columns
            and incentive_borrow_col is not None
            and incentive_borrow_col in df.columns
        )

        if has_split:
            inc_s = df[incentive_supply_col].values
            inc_b = df[incentive_borrow_col].values
            d_inc_s = np.diff(inc_s)
            d_inc_b = np.diff(inc_b)
            current_i_s = float(inc_s[-1])
            current_i_b = float(inc_b[-1])
        else:
            inc_total = df[incentive_total_col].values
            d_inc_total = np.diff(inc_total)
            # Split total equally for grid search
            d_inc_s = d_inc_total
            d_inc_b = d_inc_total
            current_i_s = float(inc_total[-1]) / 2
            current_i_b = float(inc_total[-1]) / 2

        current_tvl = float(tvl[-1])

        if utilization_col and utilization_col in df.columns:
            current_util = float(df[utilization_col].iloc[-1])
        else:
            current_util = 0.5  # default assumption

        # ── Step 1: Elasticity estimation ───────────────────────────────
        ols_supply = ols_borrow = None

        model_s = ElasticityModel(
            prior_mu=self.prior_beta_mu, prior_sigma=self.prior_beta_sigma
        )
        ols_supply = model_s.fit_ols(d_inc_s, d_tvl)
        post_s = model_s.fit_posterior(d_inc_s, d_tvl)

        model_b = ElasticityModel(
            prior_mu=self.prior_beta_mu, prior_sigma=self.prior_beta_sigma
        )
        ols_borrow = model_b.fit_ols(d_inc_b, d_tvl)
        post_b = model_b.fit_posterior(d_inc_b, d_tvl)

        # ── Step 2: Decay estimation ────────────────────────────────────
        decay_model = DecayModel()
        decay_result = decay_model.fit(tvl)
        alpha_d, beta_d = decay_model.decay_prior_params()
        equilibrium = decay_result.equilibrium or current_tvl

        # ── Step 3: Grid search optimization ────────────────────────────
        grid_opt = GridSearchOptimizer(
            rate_model=self.rate_model,
            n_mc=self.n_mc,
            horizon=self.horizon,
        )

        grid_result = grid_opt.optimize(
            current_tvl=current_tvl,
            current_utilization=current_util,
            current_i_supply=current_i_s,
            current_i_borrow=current_i_b,
            posterior_beta_s=post_s,
            posterior_beta_b=post_b,
            decay_alpha=alpha_d,
            decay_beta=beta_d,
            equilibrium_tvl=equilibrium,
            grid_n=self.grid_n,
            supply_range=supply_range,
            borrow_range=borrow_range,
            seed=seed,
        )

        # ── Step 4: Convergence check ───────────────────────────────────
        converged = False
        expected_profit = grid_result.optimal.expected_profit

        cv_s = abs(post_s.sigma / post_s.mu) if abs(post_s.mu) > 1e-10 else float("inf")
        cv_b = abs(post_b.sigma / post_b.mu) if abs(post_b.mu) > 1e-10 else float("inf")

        if cv_s < self.convergence_cv and cv_b < self.convergence_cv:
            converged = True

        if self._prev_profit is not None:
            if abs(expected_profit - self._prev_profit) < self.convergence_profit_tol:
                converged = True

        self._prev_profit = expected_profit

        result = IterationResult(
            iteration=iteration,
            ols_supply=ols_supply,
            ols_borrow=ols_borrow,
            posterior_supply=post_s,
            posterior_borrow=post_b,
            decay=decay_result,
            grid_result=grid_result,
            optimal_supply=grid_result.optimal.i_supply,
            optimal_borrow=grid_result.optimal.i_borrow,
            expected_profit=expected_profit,
            converged=converged,
        )

        self._history.append(result)
        return result

    def run_backtest(
        self,
        df: pd.DataFrame,
        window_size: int = 90,
        step_size: int = 30,
        **kwargs: Any,
    ) -> list[IterationResult]:
        """Run rolling-window backtest over historical data.

        Simulates the closed-loop process: for each window, fit posteriors
        on the trailing data and produce an optimal recommendation.

        Parameters
        ----------
        df : pd.DataFrame
            Full historical dataset.
        window_size : int
            Number of days in each estimation window.
        step_size : int
            Days between iterations (observation period).
        **kwargs
            Additional keyword arguments passed to run_iteration.

        Returns
        -------
        list of IterationResult
        """
        results = []
        start = 0
        while start + window_size <= len(df):
            window = df.iloc[start : start + window_size]
            result = self.run_iteration(window, **kwargs)
            results.append(result)
            start += step_size

        return results

    def recommendation(self) -> dict[str, Any]:
        """Generate a human-readable recommendation from the latest iteration.

        Returns
        -------
        dict with recommendation details.
        """
        if not self._history:
            return {"status": "No iterations run yet."}

        latest = self._history[-1]

        # Classify based on elasticity significance
        supply_significant = (
            latest.ols_supply is not None and latest.ols_supply.significant
        )
        borrow_significant = (
            latest.ols_borrow is not None and latest.ols_borrow.significant
        )

        supply_r2 = latest.ols_supply.r_squared if latest.ols_supply else 0
        borrow_r2 = latest.ols_borrow.r_squared if latest.ols_borrow else 0

        if supply_r2 < 0.01 and borrow_r2 < 0.01:
            assessment = "INELASTIC"
            rationale = (
                "Incentives explain <1% of TVL variance on both sides. "
                "Recommend reducing to minimum viable levels."
            )
        elif supply_significant and not borrow_significant:
            assessment = "SUPPLY_RESPONSIVE"
            rationale = (
                "Supply-side incentives show statistically significant effect. "
                "Consider maintaining supply incentives, reduce borrow."
            )
        elif borrow_significant and not supply_significant:
            assessment = "BORROW_RESPONSIVE"
            rationale = (
                "Borrow-side incentives show statistically significant effect. "
                "Consider maintaining borrow incentives, reduce supply."
            )
        else:
            assessment = "ELASTIC"
            rationale = (
                "Both supply and borrow incentives show significant effects. "
                "Optimize allocation per grid search results."
            )

        return {
            "assessment": assessment,
            "rationale": rationale,
            "optimal_supply": latest.optimal_supply,
            "optimal_borrow": latest.optimal_borrow,
            "expected_daily_profit": latest.expected_profit,
            "posterior_supply_ci95": latest.posterior_supply.ci_95,
            "posterior_borrow_ci95": latest.posterior_borrow.ci_95,
            "decay_half_life": latest.decay.half_life,
            "converged": latest.converged,
            "iteration": latest.iteration,
        }

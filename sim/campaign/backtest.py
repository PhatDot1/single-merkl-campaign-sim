"""
Rolling-window historical backtest for incentive analysis.

Slides a fixed estimation window over historical pool data, fitting the full
analysis pipeline (elasticity, decay, regimes) at each step.

Useful for:
  - Validating that elasticity estimates are stable over time
  - Detecting when a venue transitions between market regimes
  - Checking whether the campaign horizon is appropriate across periods
  - Producing time-stamped parameter estimates for dashboards

Usage
-----
    from campaign.historical import fetch_pool_history
    from campaign.backtest import run_rolling_backtest

    history = fetch_pool_history("your-pool-id", days=365)
    result = run_rolling_backtest(history, window_size=90, step_size=30)
    print(result.summary())

    # Access time-series of half-life estimates
    for hl in result.half_lives:
        print(f"{hl:.1f} days")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analysis import AnalysisReport, generate_report
from .historical import PoolHistory
from .state import IRMParams


@dataclass
class BacktestWindow:
    """Single window output from rolling backtest."""

    window_start: str    # ISO date (YYYY-MM-DD)
    window_end: str      # ISO date
    n_observations: int
    report: AnalysisReport


@dataclass
class BacktestResult:
    """Full rolling-window backtest output."""

    pool_id: str
    window_size: int
    step_size: int
    windows: list[BacktestWindow] = field(default_factory=list)

    @property
    def elasticity_betas(self) -> list[float]:
        """Time series of OLS beta estimates (TVL change per $1 incentive change)."""
        return [
            w.report.elasticity_total.beta
            for w in self.windows
            if w.report.elasticity_total is not None
        ]

    @property
    def elasticity_r2s(self) -> list[float]:
        """Time series of R² from elasticity regression."""
        return [
            w.report.elasticity_total.r_squared
            for w in self.windows
            if w.report.elasticity_total is not None
        ]

    @property
    def elasticity_significant_frac(self) -> float:
        """Fraction of windows where elasticity was statistically significant (p<0.05)."""
        total = [
            w.report.elasticity_total
            for w in self.windows
            if w.report.elasticity_total is not None
        ]
        if not total:
            return 0.0
        return sum(1 for r in total if r.significant) / len(total)

    @property
    def half_lives(self) -> list[float]:
        """Time series of TVL half-life estimates (days)."""
        return [
            w.report.decay.half_life
            for w in self.windows
            if w.report.decay is not None and w.report.decay.half_life is not None
        ]

    @property
    def equilibrium_tvls(self) -> list[float]:
        """Time series of equilibrium TVL estimates from AR(1)."""
        return [
            w.report.decay.equilibrium
            for w in self.windows
            if w.report.decay is not None and w.report.decay.equilibrium is not None
        ]

    @property
    def window_dates(self) -> list[tuple[str, str]]:
        """(start, end) date pairs for each window."""
        return [(w.window_start, w.window_end) for w in self.windows]

    def parameter_stability(self) -> dict:
        """
        Assess stability of key estimated parameters across windows.

        Stable parameters → model is robust over time → higher confidence
        in using them to calibrate simulation inputs.
        """
        stats: dict = {}

        betas = self.elasticity_betas
        if betas:
            stats["elasticity_beta"] = {
                "mean": float(np.mean(betas)),
                "std": float(np.std(betas)),
                "cv": float(np.std(betas) / abs(np.mean(betas))) if np.mean(betas) != 0 else float("inf"),
                "min": float(np.min(betas)),
                "max": float(np.max(betas)),
            }

        r2s = self.elasticity_r2s
        if r2s:
            stats["r_squared"] = {
                "mean": float(np.mean(r2s)),
                "std": float(np.std(r2s)),
                "fraction_significant": self.elasticity_significant_frac,
            }

        hls = self.half_lives
        if hls:
            stats["half_life_days"] = {
                "mean": float(np.mean(hls)),
                "std": float(np.std(hls)),
                "min": float(np.min(hls)),
                "max": float(np.max(hls)),
            }

        eqs = self.equilibrium_tvls
        if eqs:
            stats["equilibrium_tvl"] = {
                "mean": float(np.mean(eqs)),
                "std": float(np.std(eqs)),
                "cv": float(np.std(eqs) / abs(np.mean(eqs))) if np.mean(eqs) != 0 else float("inf"),
            }

        return stats

    def summary(self) -> str:
        lines = [
            f"Rolling backtest: {self.pool_id}",
            f"  {len(self.windows)} windows × {self.window_size}d"
            f"  (step = {self.step_size}d)",
        ]

        betas = self.elasticity_betas
        if betas:
            lines.append(
                f"  Elasticity beta:  "
                f"mean={np.mean(betas):,.0f}  std={np.std(betas):,.0f}"
                f"  (significant in {self.elasticity_significant_frac:.0%} of windows)"
            )

        r2s = self.elasticity_r2s
        if r2s:
            lines.append(
                f"  R² (mean±std):    {np.mean(r2s):.3%} ± {np.std(r2s):.3%}"
            )

        hls = self.half_lives
        if hls:
            lines.append(
                f"  Half-life:        "
                f"mean={np.mean(hls):.1f}d  std={np.std(hls):.1f}d"
                f"  range=[{np.min(hls):.1f}, {np.max(hls):.1f}]"
            )

        eqs = self.equilibrium_tvls
        if eqs:
            lines.append(
                f"  Equilibrium TVL:  "
                f"${np.mean(eqs) / 1e6:.1f}M ± ${np.std(eqs) / 1e6:.1f}M"
            )

        return "\n".join(lines)


def run_rolling_backtest(
    history: PoolHistory,
    window_size: int = 90,
    step_size: int = 30,
    irm: Optional[IRMParams] = None,
    current_utilization: float = 0.8,
    bootstrap_n: int = 200,
) -> BacktestResult:
    """
    Run a rolling-window backtest over historical pool data.

    For each window, the full analysis pipeline is run independently:
    elasticity, decay, independence, regime analysis, profitability.

    Parameters
    ----------
    history : PoolHistory
        Full historical dataset from fetch_pool_history().
    window_size : int
        Days per estimation window (default 90).
        Smaller windows are noisier but detect structural breaks faster.
    step_size : int
        Window advance per step in days (default 30).
        step_size < window_size creates overlapping windows.
    irm : IRMParams, optional
        Venue interest rate model for profitability analysis.
    current_utilization : float
        Utilization ratio for profitability snapshots (0.0–1.0).
    bootstrap_n : int
        Bootstrap iterations per window (default 200, lower than full analysis).

    Returns
    -------
    BacktestResult
        Contains all per-window AnalysisReports and aggregated statistics.
    """
    pts = history.points
    n = len(pts)

    result = BacktestResult(
        pool_id=history.pool_id,
        window_size=window_size,
        step_size=step_size,
    )

    start = 0
    while start + window_size <= n:
        end = start + window_size
        window_pts = pts[start:end]

        sub = PoolHistory(
            pool_id=history.pool_id,
            project=history.project,
            symbol=history.symbol,
            chain=history.chain,
            points=window_pts,
        )

        report = generate_report(
            sub,
            irm=irm,
            current_utilization=current_utilization,
            bootstrap_n=bootstrap_n,
        )

        result.windows.append(
            BacktestWindow(
                window_start=window_pts[0].timestamp,
                window_end=window_pts[-1].timestamp,
                n_observations=len(window_pts),
                report=report,
            )
        )

        start += step_size

    return result

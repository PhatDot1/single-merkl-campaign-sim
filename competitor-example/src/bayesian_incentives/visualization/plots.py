"""
Visualization functions for all analysis outputs.

All functions return (fig, ax) tuples for composability.
Minimal annotation style: text near elements, no decorative arrows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from bayesian_incentives.models.elasticity import OLSResult, BayesianPosterior
from bayesian_incentives.models.decay import DecayResult
from bayesian_incentives.optimization.grid_search import GridSearchResult


def _style_ax(ax: Axes) -> None:
    """Apply minimal consistent styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ── Elasticity ──────────────────────────────────────────────────────────

def plot_elasticity(
    d_incentive: np.ndarray,
    d_tvl: np.ndarray,
    ols: OLSResult,
    title: str = "TVL Elasticity to Incentive Changes",
    figsize: tuple[float, float] = (8, 5),
) -> tuple[Figure, Axes]:
    """Scatter of d_TVL vs d_Incentive with OLS fit line."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    mask = np.isfinite(d_incentive) & np.isfinite(d_tvl)
    x, y = d_incentive[mask], d_tvl[mask]

    ax.scatter(x, y, alpha=0.4, s=15, color="#4a7c9b", edgecolors="none")

    # Fit line
    x_range = np.linspace(x.min(), x.max(), 100)
    y_hat = ols.alpha + ols.beta * x_range
    ax.plot(x_range, y_hat, color="#c0392b", linewidth=1.5)

    # Annotation near the line
    ax.text(
        0.02, 0.95,
        f"beta={ols.beta:,.0f}  R²={ols.r_squared:.2%}  p={ols.p_value:.3f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
    )

    ax.set_xlabel("Δ Incentive ($/day)", fontsize=10)
    ax.set_ylabel("Δ TVL ($)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig, ax


def plot_lagged_elasticity(
    lagged: dict[int, OLSResult],
    title: str = "Lagged Elasticity (R² by lag)",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[Figure, Axes]:
    """Bar chart of R² at different lags."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    lags = sorted(lagged.keys())
    r2s = [lagged[k].r_squared for k in lags]
    colors = ["#c0392b" if lagged[k].p_value < 0.05 else "#bdc3c7" for k in lags]

    ax.bar([str(k) for k in lags], r2s, color=colors)
    ax.set_xlabel("Lag (days)", fontsize=10)
    ax.set_ylabel("R²", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))

    fig.tight_layout()
    return fig, ax


# ── Decay / Persistence ────────────────────────────────────────────────

def plot_decay(
    decay: DecayResult,
    horizons: int = 90,
    title: str = "Impulse Response (shock decay)",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[Figure, Axes]:
    """Plot impulse response function rho^t."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    t = np.arange(horizons)
    ir = decay.impulse_response(horizons)

    ax.plot(t, ir, color="#2c3e50", linewidth=1.5)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")

    if decay.half_life and decay.half_life < horizons:
        ax.axvline(decay.half_life, color="#c0392b", linewidth=0.8, linestyle=":")
        ax.text(
            decay.half_life + 1, 0.52,
            f"t½ = {decay.half_life:.0f}d",
            fontsize=9, color="#c0392b",
        )

    ax.set_xlabel("Days after shock", fontsize=10)
    ax.set_ylabel("Remaining effect", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    return fig, ax


# ── Regimes ─────────────────────────────────────────────────────────────

def plot_regimes(
    dates: np.ndarray,
    tvl: np.ndarray,
    regime_mask: np.ndarray,
    regime_name: str = "Regime",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure, Axes]:
    """TVL time series with regime periods shaded."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    ax.plot(dates, tvl, color="#2c3e50", linewidth=0.8)

    # Shade regime periods
    in_regime = False
    start = None
    for i, m in enumerate(regime_mask):
        if m and not in_regime:
            start = i
            in_regime = True
        elif not m and in_regime:
            ax.axvspan(dates[start], dates[i - 1], alpha=0.15, color="#e74c3c")
            in_regime = False
    if in_regime:
        ax.axvspan(dates[start], dates[-1], alpha=0.15, color="#e74c3c")

    ax.set_ylabel("TVL ($)", fontsize=10)
    ax.set_title(f"TVL with {regime_name} periods highlighted", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))

    fig.tight_layout()
    return fig, ax


# ── Posterior ───────────────────────────────────────────────────────────

def plot_posterior(
    posterior: BayesianPosterior,
    prior_mu: float = 0.0,
    prior_sigma: float = 1e6,
    title: str = "Posterior vs Prior: Elasticity",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[Figure, Axes]:
    """Plot posterior density for beta with optional prior overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    # Posterior
    x_range = np.linspace(
        posterior.mu - 4 * posterior.sigma,
        posterior.mu + 4 * posterior.sigma,
        300,
    )
    from scipy.stats import norm

    y_post = norm.pdf(x_range, posterior.mu, posterior.sigma)
    ax.plot(x_range, y_post, color="#2c3e50", linewidth=1.5, label="Posterior")
    ax.fill_between(x_range, y_post, alpha=0.15, color="#2c3e50")

    # 95% CI shading
    ci = posterior.ci_95
    ci_mask = (x_range >= ci[0]) & (x_range <= ci[1])
    ax.fill_between(x_range[ci_mask], y_post[ci_mask], alpha=0.25, color="#3498db")

    # Zero line
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.text(
        0.02, 0.95,
        f"mu={posterior.mu:,.2f}, sigma={posterior.sigma:,.2f}\n"
        f"95% CI: [{ci[0]:,.2f}, {ci[1]:,.2f}]",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
    )

    ax.set_xlabel("Beta (elasticity coefficient)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    return fig, ax


# ── Profit surface ──────────────────────────────────────────────────────

def plot_profit_surface(
    grid_result: GridSearchResult,
    title: str = "Expected Daily Profit Surface",
    figsize: tuple[float, float] = (8, 6),
) -> tuple[Figure, Axes]:
    """Heatmap of expected profit over the incentive grid."""
    fig, ax = plt.subplots(figsize=figsize)

    supply_vals, borrow_vals, profit_2d = grid_result.profit_matrix()

    im = ax.imshow(
        profit_2d.T,
        origin="lower",
        aspect="auto",
        extent=[
            supply_vals[0], supply_vals[-1],
            borrow_vals[0], borrow_vals[-1],
        ],
        cmap="RdYlGn",
    )
    fig.colorbar(im, ax=ax, label="E[Daily Profit] ($)")

    # Mark optimal
    opt = grid_result.optimal
    ax.plot(opt.i_supply, opt.i_borrow, "k*", markersize=12)
    ax.text(
        opt.i_supply, opt.i_borrow + (borrow_vals[-1] - borrow_vals[0]) * 0.03,
        f"${opt.expected_profit:,.0f}/day",
        fontsize=9, ha="center", fontweight="bold",
    )

    ax.set_xlabel("Supply Incentive ($/day)", fontsize=10)
    ax.set_ylabel("Borrow Incentive ($/day)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig, ax


# ── Convergence ─────────────────────────────────────────────────────────

def plot_convergence(
    history: list,
    title: str = "Optimization Convergence",
    figsize: tuple[float, float] = (8, 5),
) -> tuple[Figure, Axes]:
    """Plot posterior sigma and expected profit over iterations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    _style_ax(ax1)
    _style_ax(ax2)

    iters = [h.iteration for h in history]
    sigma_s = [h.posterior_supply.sigma for h in history]
    sigma_b = [h.posterior_borrow.sigma for h in history]
    profits = [h.expected_profit for h in history]

    ax1.plot(iters, sigma_s, "o-", markersize=4, label="sigma_supply", color="#2c3e50")
    ax1.plot(iters, sigma_b, "s-", markersize=4, label="sigma_borrow", color="#c0392b")
    ax1.set_ylabel("Posterior Sigma", fontsize=10)
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title(title, fontsize=11, fontweight="bold")

    ax2.plot(iters, profits, "D-", markersize=4, color="#27ae60")
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel("E[Profit] ($/day)", fontsize=10)

    fig.tight_layout()
    return fig, (ax1, ax2)


# ── Profitability time series ───────────────────────────────────────────

def plot_profitability_ts(
    dates: np.ndarray,
    nii: np.ndarray,
    incentives: np.ndarray,
    profit: np.ndarray,
    title: str = "Daily Profitability",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, Axes]:
    """Stacked area / line chart of NII, incentive cost, and net profit."""
    fig, ax = plt.subplots(figsize=figsize)
    _style_ax(ax)

    ax.plot(dates, nii, label="NII", color="#27ae60", linewidth=1)
    ax.plot(dates, -incentives, label="-Incentives", color="#e74c3c", linewidth=1)
    ax.plot(dates, profit, label="Net Profit", color="#2c3e50", linewidth=1.5)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.fill_between(dates, profit, 0, where=profit >= 0, alpha=0.1, color="#27ae60")
    ax.fill_between(dates, profit, 0, where=profit < 0, alpha=0.1, color="#e74c3c")

    ax.set_ylabel("$/day", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    return fig, ax

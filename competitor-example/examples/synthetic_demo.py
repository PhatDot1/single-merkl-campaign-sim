"""
Example: End-to-end analysis on synthetic market data.

This demonstrates how to:
  1. Generate or load market data
  2. Run the full analysis pipeline (report)
  3. Run Bayesian optimization
  4. Produce visualizations

Replace the synthetic data generator with your own CSV/DataFrame
to analyze real protocol data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayesian_incentives.data.loader import load_market_data
from bayesian_incentives.data.transforms import (
    compute_first_differences,
    compute_log_returns,
    compute_realized_volatility,
)
from bayesian_incentives.analysis.report import generate_report
from bayesian_incentives.optimization.bayesian import BayesianOptimizer
from bayesian_incentives.visualization.plots import (
    plot_elasticity,
    plot_decay,
    plot_posterior,
    plot_profit_surface,
)


# ── Step 1: Generate synthetic market data ──────────────────────────────

def make_synthetic_market(
    n_days: int = 365,
    tvl_init: float = 1e8,
    rho: float = 0.995,
    incentive_mean: float = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a realistic synthetic market dataset.

    The data mimics a lending protocol with:
      - AR(1) TVL dynamics (incentive-inelastic)
      - Random walk incentive adjustments
      - Kinked interest rate model
      - Correlated price series
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # TVL: AR(1) with NO incentive dependence
    equilibrium = tvl_init
    c = equilibrium * (1 - rho)
    tvl = np.zeros(n_days)
    tvl[0] = tvl_init
    for t in range(1, n_days):
        tvl[t] = c + rho * tvl[t - 1] + rng.normal(0, tvl_init * 0.005)

    # Incentives: random walk (independent of TVL)
    inc_supply = np.maximum(
        incentive_mean / 2 + rng.normal(0, 20, n_days).cumsum(), 0
    )
    inc_borrow = np.maximum(
        incentive_mean / 2 + rng.normal(0, 15, n_days).cumsum(), 0
    )

    # Utilization: mean-reverting around 0.65
    util = 0.65 + 0.05 * np.sin(np.linspace(0, 6 * np.pi, n_days))
    util += rng.normal(0, 0.02, n_days).cumsum() * 0.01
    util = np.clip(util, 0.3, 0.98)

    # Price: GBM
    price = np.zeros(n_days)
    price[0] = 3000
    for t in range(1, n_days):
        price[t] = price[t - 1] * np.exp(rng.normal(0.0001, 0.02))

    # Rates (simplified)
    borrow_apr = 0.02 + 0.05 * util
    supply_apr = borrow_apr * util * 0.9

    df = pd.DataFrame(
        {
            "tvl": tvl,
            "incentive_supply": inc_supply,
            "incentive_borrow": inc_borrow,
            "incentive_total": inc_supply + inc_borrow,
            "utilization": util,
            "price": price,
            "borrow_apr": borrow_apr,
            "supply_apr": supply_apr,
            "borrow_volume": tvl * util,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ── Step 2: Run analysis ───────────────────────────────────────────────

def main():
    print("Generating synthetic market data...")
    df = make_synthetic_market()
    print(f"  {len(df)} observations, {df.index[0].date()} to {df.index[-1].date()}")
    print()

    # Full analysis report
    print("Running full analysis pipeline...")
    report = generate_report(df, market_name="Synthetic Market")
    print(report.summary())
    print()

    # ── Step 3: Bayesian optimization ───────────────────────────────────
    print("Running Bayesian optimizer...")
    optimizer = BayesianOptimizer(
        n_mc=1000,
        horizon=30,
        grid_n=15,
    )
    result = optimizer.run_iteration(df)
    print(result.summary())
    print()

    rec = optimizer.recommendation()
    print(f"Assessment: {rec['assessment']}")
    print(f"Rationale:  {rec['rationale']}")
    print(f"Optimal:    I_supply=${rec['optimal_supply']:,.2f}, I_borrow=${rec['optimal_borrow']:,.2f}")
    print(f"E[profit]:  ${rec['expected_daily_profit']:,.2f}/day")
    print()

    # ── Step 4: Visualizations ──────────────────────────────────────────
    print("Generating plots...")

    # Elasticity scatter
    d_inc = np.diff(df["incentive_total"].values)
    d_tvl = np.diff(df["tvl"].values)
    fig, _ = plot_elasticity(d_inc, d_tvl, report.ols_total)
    fig.savefig("elasticity_scatter.png", dpi=150)
    print("  -> elasticity_scatter.png")

    # Decay impulse response
    fig, _ = plot_decay(report.decay)
    fig.savefig("decay_impulse_response.png", dpi=150)
    print("  -> decay_impulse_response.png")

    # Posterior
    fig, _ = plot_posterior(report.posterior_total)
    fig.savefig("posterior_beta.png", dpi=150)
    print("  -> posterior_beta.png")

    # Profit surface
    fig, _ = plot_profit_surface(result.grid_result)
    fig.savefig("profit_surface.png", dpi=150)
    print("  -> profit_surface.png")

    plt.close("all")
    print("\nDone.")


if __name__ == "__main__":
    main()

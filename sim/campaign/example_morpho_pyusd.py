"""
Example: Optimize Merkl campaign parameters for Morpho PYUSD vault.

Uses real parameters from the Feb 10 analysis:
- Current TVL: ~$160M
- Current campaign: MAX at 6%, budget ~$120k → $170k/week
- Whale concentration: top-5 hold ~56% of deposits
- PayPal goal: market efficiency, reduce incentive dependence
- Constraints: no APR spikes, no mercenary TVL

This script runs the full surface optimization and outputs
recommended (B*, r_max*) with sensitivity analysis.
"""

from campaign.agents import (  # noqa: E402
    MercenaryConfig,
    RetailDepositorConfig,
    WhaleProfile,
)
from campaign.engine import LossWeights, run_monte_carlo
from campaign.optimizer import SurfaceGrid, SurfaceResult, optimize_surface
from campaign.state import CampaignConfig, CampaignEnvironment


def build_morpho_pyusd_environment() -> CampaignEnvironment:
    """
    Build environment from current market data.

    Competitor rates as of Feb 2026:
    - Aave PYUSD Core: ~3.32% incentive rate
    - Euler Sentora PYUSD: ~6.50%
    - Kamino Earn: ~4.06%
    - Curve PYUSD-USDC: ~6.68%
    """
    return CampaignEnvironment(
        competitor_rates={
            "aave_pyusd_core": 0.0332,
            "euler_sentora_pyusd": 0.065,
            "kamino_earn": 0.0406,
            "curve_pyusd_usdc": 0.0668,
        },
        # Weighted average of competitors (weighted by TVL or equal-weight)
        r_threshold=0.045,
        r_threshold_lo=0.035,  # If Aave dominates flow
        r_threshold_hi=0.055,  # If Euler/Curve dominate
        gas_price_gwei=25.0,
        eth_price_usd=2800.0,
    )


def build_morpho_pyusd_whales() -> list[WhaleProfile]:
    """
    Build whale profiles from on-chain data.

    From the analysis:
    Top 1 = 15.14% of $160M = ~$24.2M
    Top 2 = 12.81% = ~$20.5M
    Top 3 = 12.25% = ~$19.6M
    Top 4 = 8.24% = ~$13.2M
    Top 5 = 7.40% = ~$11.8M
    Total top-5: ~56% = ~$89.4M
    """
    tvl = 160_000_000

    profiles = [
        WhaleProfile(
            whale_id="whale_1",
            position_usd=tvl * 0.1514,  # $24.2M
            alt_rate=0.05,  # Euler Sentora is attractive alternative
            risk_premium=0.003,
            switching_cost_usd=2000,  # Large position = significant gas/slippage
            exit_delay_days=3.0,  # Likely institutional, slower to react
            reentry_delay_days=10.0,
            hysteresis_band=0.008,
            whale_type="institutional",
        ),
        WhaleProfile(
            whale_id="whale_2",
            position_usd=tvl * 0.1281,  # $20.5M
            alt_rate=0.048,
            risk_premium=0.004,
            switching_cost_usd=1800,
            exit_delay_days=2.0,
            reentry_delay_days=7.0,
            hysteresis_band=0.006,
            whale_type="quant_desk",
        ),
        WhaleProfile(
            whale_id="whale_3",
            position_usd=tvl * 0.1225,  # $19.6M
            alt_rate=0.045,
            risk_premium=0.003,
            switching_cost_usd=1500,
            exit_delay_days=2.5,
            reentry_delay_days=8.0,
            hysteresis_band=0.007,
            whale_type="institutional",
        ),
        WhaleProfile(
            whale_id="whale_4",
            position_usd=tvl * 0.0824,  # $13.2M
            alt_rate=0.052,
            risk_premium=0.005,
            switching_cost_usd=1000,
            exit_delay_days=1.5,  # Smaller, more responsive
            reentry_delay_days=5.0,
            hysteresis_band=0.005,
            whale_type="quant_desk",
        ),
        WhaleProfile(
            whale_id="whale_5",
            position_usd=tvl * 0.074,  # $11.8M
            alt_rate=0.05,
            risk_premium=0.004,
            switching_cost_usd=800,
            exit_delay_days=1.0,  # Most responsive
            reentry_delay_days=4.0,
            hysteresis_band=0.005,
            whale_type="opportunistic",
        ),
    ]

    # Print whale summary
    for p in profiles:
        print(
            f"  {p.whale_id}: ${p.position_usd / 1e6:.1f}M, "
            f"exit_threshold={p.exit_threshold:.2%}, "
            f"type={p.whale_type}"
        )

    return profiles


def build_loss_weights_paypal_aligned() -> LossWeights:
    """
    Loss weights aligned with PayPal's goals:
    - Primary: reduce incentive dependence (high w_spend)
    - Primary: hit TVL/efficiency targets (high w_tvl_shortfall)
    - Secondary: APR stability for client vaults (moderate w_apr_variance)
    - Secondary: prevent APR runaway (moderate w_apr_ceiling)
    """
    return LossWeights(
        w_spend=1.0,
        w_spend_waste_penalty=2.0,
        w_apr_variance=5e5,
        w_apr_ceiling=1e8,
        w_tvl_shortfall=5e-7,
        apr_target=0.055,
        apr_ceiling=0.10,
        tvl_target=150_000_000,
    )


def run_single_point_test():
    """
    Quick test: run Monte Carlo at the proposed configuration
    (B=$170k, r_max=7%) to validate the simulation works.
    """
    print("=" * 60)
    print("SINGLE POINT TEST: B=$170k, r_max=7%")
    print("=" * 60)

    env = build_morpho_pyusd_environment()
    whales = build_morpho_pyusd_whales()
    weights = build_loss_weights_paypal_aligned()

    config = CampaignConfig(
        weekly_budget=170_000,
        apr_cap=0.07,
        epoch_duration_days=7,
        dt_days=0.25,
        horizon_days=28,
    )

    print(f"\nConfig: B=${config.weekly_budget:,}, r_max={config.apr_cap:.1%}")
    print(f"T_bind = ${config.t_bind:,.0f}")
    print("Current TVL = $160,000,000")
    print(f"T_bind / TVL = {config.t_bind / 160_000_000:.1%}")
    print()

    mc = run_monte_carlo(
        config=config,
        env=env,
        initial_tvl=160_000_000,
        whale_profiles=whales,
        weights=weights,
        n_paths=50,  # Quick test
        retail_config=RetailDepositorConfig(
            alpha_plus=0.3,
            alpha_minus_multiplier=3.0,
            response_lag_days=5.0,
            diffusion_sigma=0.015,
        ),
        mercenary_config=MercenaryConfig(
            entry_threshold=0.08,
            exit_threshold=0.06,
            max_capital_usd=15_000_000,
        ),
        base_seed=42,
    )

    print(f"Mean loss:           {mc.mean_loss:.4e}")
    print(f"Mean APR:            {mc.mean_apr:.2%}")
    print(f"APR range (p5-p95):  [{mc.apr_p5:.2%}, {mc.apr_p95:.2%}]")
    print(f"Mean TVL:            ${mc.mean_tvl:,.0f}")
    print(f"TVL min (p5):        ${mc.tvl_min_p5:,.0f}")
    print(f"Budget utilization:  {mc.mean_budget_util:.1%}")
    print(f"Mean cascade depth:  {mc.mean_cascade_depth:.2f}")
    print(f"Mercenary fraction:  {mc.mean_mercenary_fraction:.1%}")
    print(f"Time cap binding:    {mc.mean_time_cap_binding:.1%}")
    print(f"Feasible:            {mc.is_feasible}")
    print()

    # Loss breakdown
    print("Loss components:")
    for k, v in mc.loss_components.items():
        print(f"  {k}: {v:.4e}")

    return mc


def run_surface_optimization():
    """
    Full surface optimization over (B, r_max) grid.
    """
    print("=" * 60)
    print("SURFACE OPTIMIZATION: Morpho PYUSD")
    print("=" * 60)

    env = build_morpho_pyusd_environment()
    whales = build_morpho_pyusd_whales()
    weights = build_loss_weights_paypal_aligned()

    # Grid: centered on expected operating point
    # B: $100k — $250k (around proposed $170k)
    # r_max: 4% — 12% (covering MAX-like through Float-like)
    grid = SurfaceGrid.from_ranges(
        B_min=100_000,
        B_max=250_000,
        B_steps=16,
        r_max_min=0.04,
        r_max_max=0.12,
        r_max_steps=17,
        dt_days=0.25,
        horizon_days=28,
    )

    print(f"\nGrid: {grid.shape[0]} × {grid.shape[1]} = {grid.shape[0] * grid.shape[1]} points")
    print(f"B range: ${grid.B_values[0]:,.0f} — ${grid.B_values[-1]:,.0f}")
    print(f"r_max range: {grid.r_max_values[0]:.1%} — {grid.r_max_values[-1]:.1%}")
    print("Monte Carlo paths per point: 50")
    print(f"Total simulations: {grid.shape[0] * grid.shape[1] * 50:,}")
    print()

    result = optimize_surface(
        grid=grid,
        env=env,
        initial_tvl=160_000_000,
        whale_profiles=whales,
        weights=weights,
        n_paths=50,
        retail_config=RetailDepositorConfig(
            alpha_plus=0.3,
            alpha_minus_multiplier=3.0,
            response_lag_days=5.0,
            diffusion_sigma=0.015,
        ),
        mercenary_config=MercenaryConfig(
            entry_threshold=0.08,
            exit_threshold=0.06,
            max_capital_usd=15_000_000,
        ),
        base_seed=42,
        cascade_tolerance=3,
        verbose=True,
    )

    # Duality analysis
    print("\n── Near-Optimal Configurations (within 5% of optimum) ──")
    dual = result.duality_map(tolerance=0.05)
    for d in dual[:10]:
        print(
            f"  B=${d['B']:,.0f}, r_max={d['r_max']:.2%}, "
            f"T_bind=${d['t_bind']:,.0f}, "
            f"loss_ratio={d['loss_ratio']:.3f}"
        )

    # Stability boundary
    boundary = result.stability_boundary()
    if boundary:
        print(f"\n── Stability Boundary ({len(boundary)} points) ──")
        for b, r in boundary[:5]:
            print(f"  B=${b:,.0f}, r_max={r:.2%}")

    return result


def run_robustness_check(base_result: SurfaceResult):
    """
    Re-run optimization at different r_threshold values
    to check sensitivity to competitor rate assumptions.
    """
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK: r_threshold sensitivity")
    print("=" * 60)

    env = build_morpho_pyusd_environment()
    whales = build_morpho_pyusd_whales()
    weights = build_loss_weights_paypal_aligned()

    grid = SurfaceGrid.from_ranges(
        B_min=130_000,
        B_max=210_000,
        B_steps=9,
        r_max_min=0.05,
        r_max_max=0.10,
        r_max_steps=11,
        dt_days=0.25,
        horizon_days=28,
    )

    for r_thresh in [0.035, 0.045, 0.055]:
        env_variant = env.copy()
        env_variant.r_threshold = r_thresh

        result = optimize_surface(
            grid=grid,
            env=env_variant,
            initial_tvl=160_000_000,
            whale_profiles=whales,
            weights=weights,
            n_paths=30,
            base_seed=42,
            verbose=False,
        )

        print(
            f"  r_threshold={r_thresh:.1%}: "
            f"B*=${result.optimal_B:,.0f}, "
            f"r_max*={result.optimal_r_max:.2%}, "
            f"T_bind*=${result.optimal_t_bind:,.0f}, "
            f"loss={result.optimal_loss:.4e}"
        )


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    # Step 1: Quick validation
    mc = run_single_point_test()

    # Step 2: Full surface optimization
    result = run_surface_optimization()

    # Step 3: Robustness check
    run_robustness_check(result)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"  Weekly budget:  ${result.optimal_B:,.0f}")
    print(f"  APR cap:        {result.optimal_r_max:.2%}")
    print(f"  T_bind:         ${result.optimal_t_bind:,.0f}")
    print("  Campaign type:  Hybrid")

    sa = result.sensitivity_analysis()
    print(f"\n  {sa['interpretation']}")
    print(f"  Condition number: {sa['condition_number']:.1f}")

    dual = result.duality_map()
    if len(dual) > 1:
        print(f"\n  {len(dual)} near-optimal alternatives available")
        print("  (flexibility to adjust B or r_max within ~5% loss tolerance)")

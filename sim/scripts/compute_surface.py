"""
Offline computation script.

Modes:
    1. Single venue (static config):  --venue morpho_pyusd
    2. Single venue (live data):      --live --vault-address 0x... --vault-type morpho
    3. Multi-venue allocation:        --multi --config configs/venues_test.json

Usage:
    python scripts/compute_surface.py --venue morpho_pyusd --output results/morpho_pyusd
    python scripts/compute_surface.py --multi --config configs/venues_test.json --output results/allocation
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from campaign.agents import MercenaryConfig, RetailDepositorConfig, WhaleProfile
from campaign.engine import LossWeights
from campaign.multi_venue import (
    VenueSpec,
    allocate_budget,
)
from campaign.optimizer import SurfaceGrid, optimize_surface
from campaign.serialize import save_surface_result
from campaign.state import CampaignEnvironment

# ============================================================================
# CONFIG PATH RESOLUTION
# ============================================================================

# Search order for config files
CONFIG_SEARCH_PATHS = [
    Path("."),  # current directory
    Path("configs"),  # configs/ subdirectory
    Path(__file__).parent.parent / "configs",  # relative to script location
]


def resolve_config_path(config_path: str) -> Path:
    """
    Resolve a config file path, searching multiple locations.

    If the path is absolute or exists as given, use it directly.
    Otherwise search CONFIG_SEARCH_PATHS.
    """
    p = Path(config_path)
    if p.is_absolute() or p.exists():
        return p

    # Search in known locations
    for base in CONFIG_SEARCH_PATHS:
        candidate = base / p
        if candidate.exists():
            return candidate
        # Also try just the filename in each directory
        candidate = base / p.name
        if candidate.exists():
            return candidate

    # Nothing found — return original (will fail with clear error)
    raise FileNotFoundError(
        f"Config file '{config_path}' not found.\n"
        f"Searched in: {[str(p) for p in CONFIG_SEARCH_PATHS]}\n"
        f"Make sure the file exists in one of these locations."
    )


# ============================================================================
# LIVE DATA MODE (single venue)
# ============================================================================


def compute_from_live_data(
    vault_address: str,
    vault_type: str,
    asset_symbol: str,
    budget_min: float,
    budget_max: float,
    output_dir: str,
    n_paths: int = 100,
    tvl_target: float | None = None,
):
    """Full pipeline: fetch live data -> calibrate -> compute surface -> save."""
    from campaign.data import fetch_and_calibrate

    params = fetch_and_calibrate(
        vault_address=vault_address,
        vault_type=vault_type,
        asset_symbol=asset_symbol,
        weekly_budget_range=(budget_min, budget_max),
        tvl_target=tvl_target,
    )

    print(f"\nComputing surface for: {params.venue_name}")
    print(f"Grid: {params.grid.shape}")
    print(f"Paths per point: {n_paths}")

    result = optimize_surface(
        grid=params.grid,
        env=params.env,
        initial_tvl=params.initial_tvl,
        whale_profiles=params.whale_profiles,
        weights=params.weights,
        n_paths=n_paths,
        retail_config=params.retail_config,
        mercenary_config=params.mercenary_config,
        verbose=True,
    )

    save_surface_result(result, output_dir)
    print(f"Saved to: {output_dir}")
    return result


# ============================================================================
# MULTI-VENUE MODE
# ============================================================================


def load_venue_config(config_path: str) -> tuple[list[VenueSpec], float]:
    """
    Load multi-venue configuration from JSON.

    Expected format:
    {
      "total_weekly_budget": 1500000,
      "venues": [
        {
          "name": "Morpho PYUSD",
          "asset_symbol": "PYUSD",
          "protocol": "morpho",
          "current_tvl": 195000000,
          "current_utilization": 0.37,
          "target_tvl": 100000000,
          "target_utilization": 0.90,
          "target_incentive_rate": 0.06,
          "base_apy": 0.022,
          "budget_min": 50000,
          "budget_max": 300000,
          "r_max_min": 0.04,
          "r_max_max": 0.12,
          "pinned_budget": null,
          "pinned_r_max": null,
          "whale_profiles": []
        },
        ...
      ]
    }
    """
    resolved = resolve_config_path(config_path)
    print(f"Loading config from: {resolved}")

    with open(resolved) as f:
        cfg = json.load(f)

    total_budget = cfg["total_weekly_budget"]
    specs = []

    for v in cfg["venues"]:
        whales = []
        for wp in v.get("whale_profiles", []):
            whales.append(WhaleProfile(**wp))

        retail = None
        if v.get("retail_config"):
            retail = RetailDepositorConfig(**v["retail_config"])

        merc = None
        if v.get("mercenary_config"):
            merc = MercenaryConfig(**v["mercenary_config"])

        env = None
        if v.get("competitor_rates"):
            env = CampaignEnvironment(
                competitor_rates=v["competitor_rates"],
                r_threshold=v.get("r_threshold", 0.045),
            )

        weights = None
        if v.get("weights"):
            weights = LossWeights(**v["weights"])

        specs.append(
            VenueSpec(
                name=v["name"],
                asset_symbol=v["asset_symbol"],
                protocol=v["protocol"],
                current_tvl=v["current_tvl"],
                current_utilization=v.get("current_utilization", 0.4),
                target_tvl=v["target_tvl"],
                target_utilization=v.get("target_utilization", 0.5),
                target_incentive_rate=v.get("target_incentive_rate", 0.06),
                base_apy=v.get("base_apy", 0.0),
                budget_min=v.get("budget_min", 0),
                budget_max=v.get("budget_max", float("inf")),
                r_max_min=v.get("r_max_min", 0.04),
                r_max_max=v.get("r_max_max", 0.12),
                pinned_budget=v.get("pinned_budget"),
                pinned_r_max=v.get("pinned_r_max"),
                whale_profiles=whales,
                retail_config=retail,
                mercenary_config=merc,
                env=env,
                weights=weights,
            )
        )

    return specs, total_budget


def compute_multi_venue(
    config_path: str,
    output_dir: str,
    n_paths: int = 50,
):
    """Run multi-venue allocation from config file."""
    specs, total_budget = load_venue_config(config_path)

    t0 = time.time()
    result = allocate_budget(
        venues=specs,
        total_budget=total_budget,
        n_paths=n_paths,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save allocation summary
    alloc_data = {
        "total_budget": result.total_budget,
        "total_loss": result.total_loss,
        "lagrange_multiplier": result.lagrange_multiplier,
        "allocations": [
            {
                "name": a.name,
                "asset_symbol": a.asset_symbol,
                "protocol": a.protocol,
                "weekly_budget": a.weekly_budget,
                "apr_cap": a.apr_cap,
                "t_bind": a.t_bind,
                "loss": a.loss,
                "mean_apr": a.mean_apr,
                "mean_incentive_apr": a.mean_incentive_apr,
                "base_apy": a.base_apy,
                "apr_range": list(a.apr_range),
                "mean_tvl": a.mean_tvl,
                "budget_utilization": a.budget_utilization,
                "cascade_depth": a.cascade_depth,
                "is_feasible": a.is_feasible,
                "budget_share": a.budget_share,
                "marginal_loss": a.marginal_loss,
                "target_tvl": a.target_tvl,
                "target_incentive_rate": a.target_incentive_rate,
                "was_pinned": a.was_pinned,
            }
            for a in result.allocations
        ],
    }
    with open(out / "allocation.json", "w") as f:
        json.dump(alloc_data, f, indent=2, default=str)

    # Save per-venue surfaces
    for name, sr in result.venue_surfaces.items():
        venue_dir = out / name.replace(" ", "_").lower()
        save_surface_result(sr, str(venue_dir))

    print(f"\nComputation time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"Saved to: {output_dir}")
    return result


# ============================================================================
# STATIC VENUE CONFIGURATIONS
# ============================================================================


def morpho_pyusd_config() -> dict:
    """
    Morpho PYUSD vault — Feb 2026 parameters.

    Base APY computed from weighted average of market allocations:
    - sUSDe: 2.33% APY, ~$38.3M supply
    - cbBTC: 3.13% APY, ~$26.0M supply
    - Syrup USDC: 3.63% APY, ~$12.8M supply
    - sUSDS: 2.70% APY, ~$8.4M supply
    - wstETH: 3.40% APY, ~$0.01M supply
    - Idle (None market): 0.00% APY, ~$109.8M supply
    Total ~$195.4M, Idle ~$109.8M, Borrowed ~$85.6M
    Weighted avg base APY across ACTIVE markets: ~2.64%
    Weighted avg base APY across ALL supply (inc idle): ~1.15%

    We use the ALL-supply weighted average since depositors earn the
    vault-level blended rate, not the individual market rates.
    The vault distributes across idle + active proportionally.
    """
    # Base APY: vault-level weighted average including idle capital
    # This is what depositors actually earn as organic yield before incentives.
    # From morpho_vault_analysis_current.py output:
    #   Total assets: ~$195.4M, Utilization: ~43.8%
    #   Weighted across all allocations including 0% idle: ~1.15%
    # However, V2 avgNetApy from GraphQL may be more accurate if available.
    # Using conservative estimate based on allocation data.
    BASE_APY = 0.0115  # ~1.15% vault-level blended APY

    return {
        "name": "morpho_pyusd",
        "initial_tvl": 160_000_000,
        "env": CampaignEnvironment(
            competitor_rates={
                "aave_pyusd_core": 0.0332,
                "euler_sentora_pyusd": 0.065,
                "kamino_earn": 0.0406,
                "curve_pyusd_usdc": 0.0668,
            },
            r_threshold=0.045,
            r_threshold_lo=0.035,
            r_threshold_hi=0.055,
            gas_price_gwei=25.0,
            eth_price_usd=2800.0,
        ),
        "whales": [
            WhaleProfile(
                whale_id="whale_1",
                position_usd=160e6 * 0.1514,
                alt_rate=0.05,
                risk_premium=0.003,
                switching_cost_usd=2000,
                exit_delay_days=3.0,
                reentry_delay_days=10.0,
                hysteresis_band=0.008,
                whale_type="institutional",
            ),
            WhaleProfile(
                whale_id="whale_2",
                position_usd=160e6 * 0.1281,
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
                position_usd=160e6 * 0.1225,
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
                position_usd=160e6 * 0.0824,
                alt_rate=0.052,
                risk_premium=0.005,
                switching_cost_usd=1000,
                exit_delay_days=1.5,
                reentry_delay_days=5.0,
                hysteresis_band=0.005,
                whale_type="quant_desk",
            ),
            WhaleProfile(
                whale_id="whale_5",
                position_usd=160e6 * 0.074,
                alt_rate=0.05,
                risk_premium=0.004,
                switching_cost_usd=800,
                exit_delay_days=1.0,
                reentry_delay_days=4.0,
                hysteresis_band=0.005,
                whale_type="opportunistic",
            ),
        ],
        "retail_config": RetailDepositorConfig(
            alpha_plus=0.3,
            alpha_minus_multiplier=3.0,
            response_lag_days=5.0,
            diffusion_sigma=0.015,
        ),
        "mercenary_config": MercenaryConfig(
            entry_threshold=0.08,
            exit_threshold=0.06,
            max_capital_usd=15_000_000,
        ),
        "weights": LossWeights(
            w_spend=1.0,
            w_spend_waste_penalty=2.0,
            w_apr_variance=5e5,
            w_apr_ceiling=1e8,
            w_tvl_shortfall=5e-7,
            apr_target=0.055,  # TOTAL APR target (base + incentive)
            apr_ceiling=0.10,  # TOTAL APR ceiling
            tvl_target=150_000_000,
            apr_stability_on_total=True,  # Stability measured on total APR
        ),
        "grid": SurfaceGrid.from_ranges(
            B_min=100_000,
            B_max=250_000,
            B_steps=16,
            r_max_min=0.04,
            r_max_max=0.12,
            r_max_steps=17,
            dt_days=0.25,
            horizon_days=28,
            base_apy=BASE_APY,
        ),
    }


def euler_rlusd_config() -> dict:
    """Euler RLUSD cluster — Feb 2026 parameters."""
    BASE_APY = 0.015  # Estimated organic yield for Euler RLUSD

    return {
        "name": "euler_rlusd",
        "initial_tvl": 190_000_000,
        "env": CampaignEnvironment(
            competitor_rates={
                "aave_rlusd_core": 0.0475,
                "curve_rlusd_usdc": 0.06,
            },
            r_threshold=0.05,
            gas_price_gwei=25.0,
            eth_price_usd=2800.0,
        ),
        "whales": [
            WhaleProfile(
                whale_id="whale_1",
                position_usd=190e6 * 0.20,
                alt_rate=0.05,
                risk_premium=0.005,
                switching_cost_usd=3000,
                exit_delay_days=3.0,
                reentry_delay_days=10.0,
                hysteresis_band=0.01,
                whale_type="institutional",
            ),
        ],
        "retail_config": RetailDepositorConfig(
            alpha_plus=0.2,
            alpha_minus_multiplier=3.0,
            response_lag_days=5.0,
            diffusion_sigma=0.01,
        ),
        "mercenary_config": MercenaryConfig(
            entry_threshold=0.09,
            exit_threshold=0.07,
            max_capital_usd=20_000_000,
        ),
        "weights": LossWeights(
            w_spend=1.0,
            w_spend_waste_penalty=2.0,
            w_apr_variance=1e6,
            w_apr_ceiling=1e8,
            w_tvl_shortfall=5e-7,
            apr_target=0.065,  # TOTAL APR target
            apr_ceiling=0.10,  # TOTAL APR ceiling
            tvl_target=190_000_000,
            apr_stability_on_total=True,
        ),
        "grid": SurfaceGrid.from_ranges(
            B_min=180_000,
            B_max=300_000,
            B_steps=13,
            r_max_min=0.04,
            r_max_max=0.10,
            r_max_steps=13,
            dt_days=0.25,
            horizon_days=28,
            base_apy=BASE_APY,
        ),
    }


VENUES = {
    "morpho_pyusd": morpho_pyusd_config,
    "euler_rlusd": euler_rlusd_config,
}


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pre-compute campaign surface")
    parser.add_argument(
        "--venue",
        default=None,
        choices=list(VENUES.keys()),
        help="Static venue configuration (if not using --live or --multi)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live on-chain data instead of static config",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Multi-venue allocation mode",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Multi-venue config JSON file (required with --multi). "
        "Searched in: current dir, configs/, relative to script.",
    )
    parser.add_argument(
        "--vault-address",
        default=None,
        help="Vault address (required with --live)",
    )
    parser.add_argument(
        "--vault-type",
        default="morpho",
        choices=["euler", "morpho", "aave", "curve", "kamino"],
        help="Vault type (for --live)",
    )
    parser.add_argument(
        "--asset-symbol",
        default="PYUSD",
        help="Asset symbol for competitor lookup (for --live)",
    )
    parser.add_argument(
        "--budget-min",
        type=float,
        default=100_000,
    )
    parser.add_argument(
        "--budget-max",
        type=float,
        default=250_000,
    )
    parser.add_argument(
        "--tvl-target",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output",
        default="results/latest",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if args.multi:
        if not args.config:
            parser.error("--config is required with --multi")
        compute_multi_venue(args.config, args.output, n_paths=args.n_paths)

    elif args.live:
        if not args.vault_address:
            parser.error("--vault-address is required with --live")
        compute_from_live_data(
            vault_address=args.vault_address,
            vault_type=args.vault_type,
            asset_symbol=args.asset_symbol,
            budget_min=args.budget_min,
            budget_max=args.budget_max,
            output_dir=args.output,
            n_paths=args.n_paths,
            tvl_target=args.tvl_target,
        )

    else:
        venue = args.venue or "morpho_pyusd"
        cfg = VENUES[venue]()
        print(f"Computing surface for: {cfg['name']}")
        print(f"Grid: {cfg['grid'].shape}")
        print(f"Base APY: {cfg['grid'].base_apy:.2%}")
        print(f"Paths per point: {args.n_paths}")
        total_sims = cfg["grid"].shape[0] * cfg["grid"].shape[1] * args.n_paths
        print(f"Total simulations: {total_sims:,}")
        print()

        t0 = time.time()

        result = optimize_surface(
            grid=cfg["grid"],
            env=cfg["env"],
            initial_tvl=cfg["initial_tvl"],
            whale_profiles=cfg["whales"],
            weights=cfg["weights"],
            n_paths=args.n_paths,
            retail_config=cfg["retail_config"],
            mercenary_config=cfg["mercenary_config"],
            base_seed=args.seed,
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"\nComputation time: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
        print(f"Saving to: {args.output}")

        save_surface_result(result, args.output)
        print("Done.")


if __name__ == "__main__":
    main()

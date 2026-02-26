"""
Campaign Optimizer — Optimal Merkl campaign parameter selection.

Finds the optimal (B, r_max) on the 2D campaign parameter surface
via agent-based Monte Carlo simulation with strategic whale actors.

Production configuration:
- RPC URLs from environment (.env): ALCHEMY_ETH_RPC_URL, HELIUS_SOLANA_RPC_URL
- Kamino vault pubkeys from environment: KAMINO_PYUSD_EARN_VAULT_PUBKEY, etc.
- Aave Horizon as separate pool (0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8)
- Whale fetching via Alchemy (EVM) and Helius (Solana) — no Transfer log scanning
- Zero fallbacks to static/empty whale data

Core workflow:
    >>> from campaign import (
    ...     CampaignConfig, CampaignEnvironment,
    ...     WhaleProfile, LossWeights, run_monte_carlo,
    ...     SurfaceGrid, optimize_surface,
    ... )
    >>>
    >>> config = CampaignConfig(weekly_budget=170_000, apr_cap=0.07)
    >>> env = CampaignEnvironment(r_threshold=0.045)
    >>> whales = [WhaleProfile(whale_id="w1", position_usd=24_000_000)]
    >>> weights = LossWeights(tvl_target=150_000_000)
    >>>
    >>> mc = run_monte_carlo(config, env, 160_000_000, whales, weights, n_paths=100)
    >>> print(f"Mean APR: {mc.mean_apr:.2%}, Budget util: {mc.mean_budget_util:.1%}")

Whale fetching example (production):
    >>> from campaign.evm_data import fetch_aave_whales, fetch_euler_whales
    >>> # Requires ALCHEMY_ETH_RPC_URL in environment
    >>> aave_whales = fetch_aave_whales("PYUSD", market="core", r_threshold=0.045)
    >>> euler_whales = fetch_euler_whales("PYUSD", r_threshold=0.045)

    >>> from campaign.kamino_data import fetch_kamino_vault_whales
    >>> # Requires HELIUS_SOLANA_RPC_URL in environment
    >>> kamino_whales = fetch_kamino_vault_whales("VAULT_PUBKEY_HERE")
"""

from .agents import (
    APYSensitiveAgent,
    APYSensitiveConfig,
    CampaignAgent,
    MercenaryAgent,
    MercenaryConfig,
    RetailDepositorAgent,
    RetailDepositorConfig,
    WhaleAgent,
    WhaleProfile,
    resolve_cascades,
)
from .base_apy import (
    BaseAPYResult,
    fetch_aave_base_apy,
    fetch_all_base_apys,
    fetch_base_apy,
    fetch_defillama_base_apy,
    fetch_euler_base_apy,
    fetch_kamino_base_apy,
    fetch_morpho_base_apy,
)
from .data import (
    CalibratedVenueParams,
    CompetitorRate,
    VaultSnapshot,
    calibrate_from_snapshot,
    compute_r_threshold,
    fetch_aave_vault_snapshot,
    fetch_and_calibrate,
    fetch_competitor_rates,
    fetch_curve_pool_snapshot,
    fetch_euler_vault_snapshot,
    fetch_kamino_snapshot,
    fetch_morpho_vault_snapshot,
    fetch_usdc_benchmark,
)
from .engine import (
    CampaignLossEvaluator,
    CampaignSimulationEngine,
    LossResult,
    LossWeights,
    MonteCarloResult,
    run_monte_carlo,
)
from .evm_data import (
    AaveReserveData,
    EulerVaultData,
    build_whale_profiles_from_holders,
    fetch_aave_atoken_top_holders,
    fetch_aave_base_apy_onchain,
    fetch_aave_reserve_data,
    # Production whale fetchers (end-to-end)
    fetch_aave_whales,
    fetch_erc20_top_holders,
    fetch_euler_base_apy_onchain,
    fetch_euler_vault_data,
    fetch_euler_vault_top_holders,
    fetch_euler_whales,
)
from .kamino_data import (
    KaminoReserveMetrics,
    KaminoStrategyMetrics,
    KaminoVaultMetrics,
    KaminoVaultState,
    fetch_kamino_earn_snapshot,
    fetch_kamino_lend_snapshot,
    fetch_kamino_market_reserves,
    fetch_kamino_reserve_for_asset,
    fetch_kamino_strategy_metrics,
    fetch_kamino_strategy_whales,
    fetch_kamino_top_share_holders,
    fetch_kamino_vault_metrics,
    fetch_kamino_vault_state,
    # Production whale fetcher (end-to-end)
    fetch_kamino_vault_whales,
)
from .multi_venue import (
    MultiVenueResult,
    VenueAllocation,
    VenueSpec,
    allocate_budget,
    compute_venue_surface,
)
from .optimizer import (
    SurfaceGrid,
    SurfaceResult,
    optimize_surface,
)
from .serialize import (
    load_mc_diagnostics,
    load_metadata,
    load_surface_result,
    save_surface_result,
)
from .state import (
    CampaignConfig,
    CampaignEnvironment,
    CampaignState,
)

__all__ = [
    # State
    "CampaignConfig",
    "CampaignEnvironment",
    "CampaignState",
    # Agents
    "CampaignAgent",
    "RetailDepositorAgent",
    "RetailDepositorConfig",
    "WhaleAgent",
    "WhaleProfile",
    "MercenaryAgent",
    "MercenaryConfig",
    "APYSensitiveAgent",
    "APYSensitiveConfig",
    "resolve_cascades",
    # Engine
    "CampaignSimulationEngine",
    "CampaignLossEvaluator",
    "LossWeights",
    "LossResult",
    "MonteCarloResult",
    "run_monte_carlo",
    # Optimizer
    "SurfaceGrid",
    "SurfaceResult",
    "optimize_surface",
    # Base APY
    "BaseAPYResult",
    "fetch_base_apy",
    "fetch_all_base_apys",
    "fetch_morpho_base_apy",
    "fetch_aave_base_apy",
    "fetch_euler_base_apy",
    "fetch_kamino_base_apy",
    "fetch_defillama_base_apy",
    # EVM Data (Aave/Euler on-chain + Alchemy whale fetching)
    "AaveReserveData",
    "EulerVaultData",
    "fetch_aave_reserve_data",
    "fetch_aave_base_apy_onchain",
    "fetch_euler_vault_data",
    "fetch_euler_base_apy_onchain",
    "fetch_aave_atoken_top_holders",
    "fetch_euler_vault_top_holders",
    "fetch_erc20_top_holders",
    "build_whale_profiles_from_holders",
    "fetch_aave_whales",
    "fetch_euler_whales",
    # Kamino Data (Solana API + Helius whale fetching)
    "KaminoReserveMetrics",
    "KaminoVaultMetrics",
    "KaminoVaultState",
    "KaminoStrategyMetrics",
    "fetch_kamino_vault_state",
    "fetch_kamino_vault_metrics",
    "fetch_kamino_strategy_metrics",
    "fetch_kamino_strategy_whales",
    "fetch_kamino_market_reserves",
    "fetch_kamino_reserve_for_asset",
    "fetch_kamino_top_share_holders",
    "fetch_kamino_lend_snapshot",
    "fetch_kamino_earn_snapshot",
    "fetch_kamino_vault_whales",
    # Legacy Data (data.py)
    "VaultSnapshot",
    "CompetitorRate",
    "CalibratedVenueParams",
    "fetch_euler_vault_snapshot",
    "fetch_morpho_vault_snapshot",
    "fetch_aave_vault_snapshot",
    "fetch_curve_pool_snapshot",
    "fetch_kamino_snapshot",
    "fetch_competitor_rates",
    "compute_r_threshold",
    "fetch_usdc_benchmark",
    "calibrate_from_snapshot",
    "fetch_and_calibrate",
    # Multi-venue
    "VenueSpec",
    "VenueAllocation",
    "MultiVenueResult",
    "allocate_budget",
    "compute_venue_surface",
    # Serialization
    "save_surface_result",
    "load_surface_result",
    "load_metadata",
    "load_mc_diagnostics",
]

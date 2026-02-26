"""
Simulation engine for campaign optimization.

Inner loop: run one TVL simulation path for a given (B, r_max) configuration.
Loss evaluator: compute the multi-objective loss functional on a completed path.

Key APR distinction:
- r_total = base_apy + incentive_apr: what depositors see, drives drift/whale behavior
- r_incentive: what Merkl pays, drives spend cost
- APR stability targets can reference either (configurable via apr_stability_on_total)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .agents import (
    APYSensitiveAgent,
    APYSensitiveConfig,
    MercenaryAgent,
    MercenaryConfig,
    RetailDepositorAgent,
    RetailDepositorConfig,
    WhaleAgent,
    WhaleProfile,
    resolve_cascades,
)
from .state import CampaignConfig, CampaignEnvironment, CampaignState

# ============================================================================
# SIMULATION ENGINE (inner loop — single path)
# ============================================================================


@dataclass
class CampaignSimulationEngine:
    """
    Runs a single TVL simulation path for a given campaign configuration.

    Simulation loop per timestep:
    1. Record state
    2. RetailDepositorAgent: compute flow, update TVL
    3. Add diffusion noise (already in retail agent)
    4. WhaleAgents: observe APR, decide stay/exit/re-enter
    5. Resolve cascades (if any whale exited)
    6. MercenaryAgent: observe APR, enter/exit
    7. Update spend accounting
    8. Advance time
    9. Handle epoch boundaries
    """

    config: CampaignConfig
    env: CampaignEnvironment
    retail_agent: RetailDepositorAgent
    whale_agents: list[WhaleAgent]
    mercenary_agent: MercenaryAgent
    apy_sensitive_agent: APYSensitiveAgent

    def run(self, initial_state: CampaignState) -> CampaignState:
        """
        Run simulation and return final state with full history.

        Args:
            initial_state: Starting campaign state (will be cloned)

        Returns:
            CampaignState with populated history arrays
        """
        state = initial_state.fast_clone()
        state.reset_epoch_budget(self.config)

        # Register whale positions in state
        for whale in self.whale_agents:
            state.whale_positions[whale.profile.whale_id] = whale.profile.position_usd
            state.whale_exited[whale.profile.whale_id] = False

        env = self.env.copy()

        for step in range(self.config.num_steps):
            state.current_step = step

            # 1. Record state
            state.record(self.config, env)

            # 2-3. Retail depositor flow (includes drift + diffusion)
            # Note: retail agent calls config.realized_apr() which now returns
            # total APR (base + incentive). This is correct — depositors respond
            # to total yield vs competitor total yield (r_threshold).
            self.retail_agent.act(state, self.config, env)

            # 4. Whale actions
            # Whales also use config.realized_apr() — they compare total yield
            # at this venue against their alternative total yields.
            any_exit = False
            for whale in self.whale_agents:
                was_exited = whale.has_exited
                whale.act(state, self.config, env)
                if whale.has_exited and not was_exited:
                    any_exit = True

            # 5. Resolve cascades if any whale exited this step
            if any_exit:
                resolve_cascades(self.whale_agents, state, self.config)

            # 5.5 APY-sensitive agent — floor-sensitive depositors (loopers, yield chasers)
            # Placed after whale cascades (sensitive depositors react to post-cascade APR)
            # and before mercenary (sensitive exits can spike APR → mercenary entry)
            self.apy_sensitive_agent.act(state, self.config, env)

            # 6. Mercenary agent
            self.mercenary_agent.act(state, self.config, env)

            # 7. Update spend accounting (incentive spend only)
            state.update_spend(self.config, self.config.dt_days)

            # 8. Advance time
            env.step(self.config.dt_days)

            # 9. Epoch boundary: reset budget
            epoch_step = int(self.config.epoch_duration_days / self.config.dt_days)
            if epoch_step > 0 and (step + 1) % epoch_step == 0:
                state.reset_epoch_budget(self.config)

        return state

    @classmethod
    def from_params(
        cls,
        config: CampaignConfig,
        env: CampaignEnvironment,
        whale_profiles: list[WhaleProfile] | None = None,
        retail_config: RetailDepositorConfig | None = None,
        mercenary_config: MercenaryConfig | None = None,
        apy_sensitive_config: APYSensitiveConfig | None = None,
        seed: int = 42,
    ) -> CampaignSimulationEngine:
        """
        Factory method: build engine from parameters.

        Args:
            config: Campaign configuration (B, r_max)
            env: Environment (competitor rates, etc.)
            whale_profiles: List of whale depositor profiles
            retail_config: Retail depositor behavior config
            mercenary_config: Mercenary capital config
            apy_sensitive_config: APY-sensitive depositor config (floor sensitivity)
            seed: Base random seed (each agent gets seed + offset)
        """
        retail = RetailDepositorAgent(
            config=retail_config or RetailDepositorConfig(),
            seed=seed,
        )

        whales = []
        if whale_profiles:
            for i, profile in enumerate(whale_profiles):
                whales.append(WhaleAgent(profile=profile, seed=seed + i + 1))

        mercenary = MercenaryAgent(
            config=mercenary_config or MercenaryConfig(),
            seed=seed + 100,
        )

        apy_sensitive = APYSensitiveAgent(
            config=apy_sensitive_config or APYSensitiveConfig(),
            seed=seed + 200,
        )

        return cls(
            config=config,
            env=env,
            retail_agent=retail,
            whale_agents=whales,
            mercenary_agent=mercenary,
            apy_sensitive_agent=apy_sensitive,
        )

    def reset_agents(self, seed: int | None = None) -> None:
        """Reset all agents for a new Monte Carlo path."""
        self.retail_agent.reset()
        if seed is not None:
            self.retail_agent.rng = np.random.RandomState(seed)

        for i, whale in enumerate(self.whale_agents):
            whale.reset()
            if seed is not None:
                whale.rng = np.random.RandomState(seed + i + 1)

        self.mercenary_agent.reset()
        if seed is not None:
            self.mercenary_agent.rng = np.random.RandomState(seed + 100)

        self.apy_sensitive_agent.reset()
        if seed is not None:
            self.apy_sensitive_agent.rng = np.random.RandomState(seed + 200)


# ============================================================================
# LOSS FUNCTIONAL EVALUATOR
# ============================================================================


@dataclass
class LossWeights:
    """
    Weights for the multi-objective loss functional.

    NORMALIZED (Feb 2026): All loss components are self-normalized to ~O(1)
    before weighting, so weights live in a human-readable 0.1-20.0 range.
    This replaces the old system where weights spanned 12 orders of magnitude
    (0.5 to 1e9) and were impossible to tune.

    Priority Order:
    1. Hit TVL target (primary)       → w_tvl_shortfall = 8.0
    2. Avoid mercenary / whale risk   → w_mercenary = 6.0, w_whale_proximity = 6.0
    3. Maintain APR stability         → w_apr_variance = 3.0, w_apr_ceiling = 5.0
    4. APR floor (APY-sensitive protection) → w_apr_floor = 7.0 (active when floor_apr > 0)
    5. Minimize spend (weakest)       → w_spend = 1.0 (tie-breaker only)
    """

    w_spend: float = 1.0
    """Weight on normalized spend. Spend is normalized by total budget
    so it's always in [0, 1]. Tie-breaker only — two configs hitting
    TVL target? Pick the cheaper one."""

    w_spend_waste_penalty: float = 0.0
    """DEPRECATED: TVL overshoot is good, not bad. Set to 0.0."""

    w_apr_variance: float = 3.0
    """Weight on APR variance. Component normalized by apr_target^2
    so a 1% deviation from 5% target ≈ 0.04 raw → ~O(1) after norm."""

    w_apr_ceiling: float = 5.0
    """Weight on APR ceiling breaches. Normalized by apr_ceiling^2.
    Higher than variance — ceiling is a harder constraint."""

    w_tvl_shortfall: float = 8.0
    """Weight on TVL shortfall. Normalized by tvl_target^2.
    Primary objective — highest weight. A 10% TVL miss ≈ 0.01 raw →
    multiplied by 8.0 = 0.08 contribution."""

    w_budget_waste: float = 0.0
    """Disabled. Underspending is valid (MAX regime)."""

    w_mercenary: float = 6.0
    """Weight on mercenary capital fraction. Already in [0, 1]^2.
    High priority — mercenary capital is 'very not wanted'."""

    w_whale_proximity: float = 6.0
    """Weight on whale proximity risk. Normalized by (num_whales × num_steps).
    Penalizes APR being close to whale exit thresholds."""

    w_apr_floor: float = 7.0
    """Weight on APR floor breach penalty. Active when apr_floor > 0.
    Normalized by apr_floor^2. Penalizes any timestep where total APR
    drops below the floor — critical for APY-sensitive venues."""

    # Reference values
    apr_target: float = 0.055
    """Target APR for stability calculation (center of desired band)."""

    apr_ceiling: float = 0.10
    """Hard APR ceiling (above this, quant strategies break)."""

    tvl_target: float = 150_000_000.0
    """TVL target (USD)."""

    apr_floor: float = 0.0
    """APR floor for APY-sensitive depositor protection. 0.0 = disabled.
    When > 0, any timestep where total APR < apr_floor is penalized
    proportionally to the breach depth and apr_floor_sensitivity."""

    apr_floor_sensitivity: float = 0.0
    """How critical staying above apr_floor is (0.0 to 1.0).
    0.0 = no floor penalty (disabled).
    0.5 = moderate — soft preference for staying above floor.
    1.0 = maximum — hard constraint, optimizer strongly avoids breach.
    Scales the floor penalty: higher sensitivity = steeper penalty."""

    spend_reference_budget: float = 0.0
    """If > 0, normalize spend by this FIXED reference instead of the
    config's own total budget. Used in Set APR mode where budget varies
    across grid points: normalizing by each config's own budget makes
    spend_cost ≈ utilization (same for all configs when cap binds).
    Setting this to the budget ceiling ensures absolute spend is compared."""

    apr_stability_on_total: bool = True
    """If True: APR stability costs computed on total APR (base + incentive)."""


@dataclass
class LossResult:
    """Detailed breakdown of loss computation."""

    total_loss: float
    spend_cost: float
    apr_variance_cost: float
    apr_ceiling_cost: float
    tvl_shortfall_cost: float
    merkl_fee_cost: float
    budget_waste_cost: float
    mercenary_cost: float = 0.0
    whale_proximity_cost: float = 0.0
    floor_breach_cost: float = 0.0  # APR floor penalty (APY-sensitive protection)

    # Diagnostic metrics (not part of loss, but useful for analysis)
    avg_apr: float = 0.0  # TOTAL APR (base + incentive)
    avg_incentive_apr: float = 0.0  # Incentive APR only
    apr_std: float = 0.0
    apr_p5: float = 0.0
    apr_p95: float = 0.0
    avg_tvl: float = 0.0
    tvl_min: float = 0.0
    tvl_max: float = 0.0
    total_spend: float = 0.0
    budget_utilization: float = 0.0  # fraction of budget actually spent
    max_cascade_depth: int = 0
    mercenary_fraction: float = 0.0  # peak mercenary TVL / peak total TVL
    sensitive_fraction: float = 0.0  # APY-sensitive TVL / total TVL at end
    time_cap_binding: float = 0.0  # fraction of time APR cap was binding
    time_below_floor: float = 0.0  # fraction of time APR was below floor


class CampaignLossEvaluator:
    """
    Evaluates the multi-objective loss functional on a completed simulation path.
    """

    def __init__(self, weights: LossWeights | None = None):
        self.weights = weights or LossWeights()

    def evaluate(
        self,
        state: CampaignState,
        config: CampaignConfig,
    ) -> LossResult:
        """
        Compute loss on a completed simulation path.

        Args:
            state: CampaignState with populated history
            config: CampaignConfig used for the simulation

        Returns:
            LossResult with total loss and component breakdown
        """
        w = self.weights
        arrays = state.to_arrays()
        tvl = arrays["tvl"]
        apr_total = arrays["apr"]  # TOTAL APR (base + incentive)
        spend = arrays["spend"]  # Incentive spend only
        dt = config.dt_days
        n = len(tvl)

        if n == 0:
            return LossResult(
                total_loss=float("inf"),
                spend_cost=0,
                apr_variance_cost=0,
                apr_ceiling_cost=0,
                tvl_shortfall_cost=0,
                merkl_fee_cost=0,
                budget_waste_cost=0,
                mercenary_cost=0,
                whale_proximity_cost=0,
                floor_breach_cost=0,
                avg_apr=0,
                avg_incentive_apr=0,
                apr_std=0,
                apr_p5=0,
                apr_p95=0,
                avg_tvl=0,
                tvl_min=0,
                tvl_max=0,
                total_spend=0,
                budget_utilization=0,
                max_cascade_depth=0,
                mercenary_fraction=0,
                sensitive_fraction=0,
                time_cap_binding=0,
                time_below_floor=0,
            )

        # Incentive APR array
        if "incentive_apr" in arrays:
            apr_incentive = arrays["incentive_apr"]
        else:
            apr_incentive = apr_total - config.base_apy

        # Choose which APR series to use for stability costs
        if w.apr_stability_on_total:
            apr_for_stability = apr_total
        else:
            apr_for_stability = apr_incentive

        # Normalization constants
        total_budget = config.weekly_budget * config.num_epochs
        T = n * dt  # Total simulation time in days

        # ── Spend cost (NORMALIZED) ──
        # For Set APR mode (spend_reference_budget > 0): normalize by the
        # fixed reference so absolute spend is compared across grid points.
        # For Set Budget mode: normalize by config's own budget → utilization.
        normalization_budget = (
            w.spend_reference_budget if w.spend_reference_budget > 0 else total_budget
        )
        below_target = tvl < w.tvl_target
        spend_multiplier = np.where(below_target, 1.0 + w.w_spend_waste_penalty, 1.0)
        raw_spend = np.sum(spend * spend_multiplier * dt)
        spend_normalized = raw_spend / max(normalization_budget, 1.0)
        spend_cost = w.w_spend * spend_normalized

        # ── APR variance cost (NORMALIZED) ──
        # (r - r_target)^2 / r_target^2, time-averaged → ~O(1) for typical deviations
        apr_dev = apr_for_stability - w.apr_target
        apr_norm_sq = max(w.apr_target**2, 1e-6)
        apr_variance_raw = np.mean(apr_dev**2) / apr_norm_sq
        apr_variance_cost = w.w_apr_variance * apr_variance_raw

        # ── APR ceiling cost (NORMALIZED) ──
        # max(0, r - r_ceiling)^2 / r_ceiling^2, time-averaged
        apr_excess = np.maximum(0, apr_for_stability - w.apr_ceiling)
        apr_ceil_norm_sq = max(w.apr_ceiling**2, 1e-6)
        apr_ceiling_raw = np.mean(apr_excess**2) / apr_ceil_norm_sq
        apr_ceiling_cost = w.w_apr_ceiling * apr_ceiling_raw

        # ── TVL shortfall cost (NORMALIZED) ──
        # max(0, T* - TVL)^2 / T*^2, time-averaged → 10% miss ≈ 0.01
        tvl_shortfall = np.maximum(0, w.tvl_target - tvl)
        tvl_norm_sq = max(w.tvl_target**2, 1.0)
        tvl_shortfall_raw = np.mean(tvl_shortfall**2) / tvl_norm_sq
        tvl_shortfall_cost = w.w_tvl_shortfall * tvl_shortfall_raw

        # ── Merkl fee cost (normalized by total budget) ──
        total_spent = state.budget_spent_total
        _unspent = max(0, total_budget - total_spent)
        # Normalize: fee as fraction of total budget (0 to fee_rate)
        merkl_fee_cost = (
            config.merkl_fee_rate * total_spent / total_budget if total_budget > 0 else 0.0
        )

        # ── Budget waste cost (NORMALIZED) ──
        budget_util = total_spent / total_budget if total_budget > 0 else 1.0
        waste_fraction = max(0.0, 1.0 - budget_util)
        budget_waste_cost = w.w_budget_waste * waste_fraction  # Already [0, 1]

        # ── Mercenary capital penalty (NORMALIZED) ──
        # (merc_tvl / max_tvl)^2 — already fractional, in [0, 1]
        merc_tvl_frac = state.mercenary_tvl / max(np.max(tvl), 1.0)
        mercenary_cost = w.w_mercenary * (merc_tvl_frac**2)

        # ── Whale proximity risk penalty (NORMALIZED) ──
        # Normalize by (num_whales × num_steps) to keep scale independent of
        # simulation length and whale count.
        whale_proximity_cost = 0.0
        num_whales = len(config.whale_profiles) if config.whale_profiles else 0
        if config.whale_profiles and num_whales > 0:
            raw_proximity = 0.0
            for t_idx, apr_t in enumerate(apr_total):
                for whale_profile in config.whale_profiles:
                    whale_id = whale_profile.whale_id
                    if whale_id in state.whale_exited and not state.whale_exited[whale_id]:
                        threshold = whale_profile.exit_threshold
                        if apr_t > threshold:
                            buffer = (apr_t - threshold) / max(threshold, 0.01)
                            if buffer < 0.05:
                                proximity_risk = 1.0 / max(buffer, 0.001) ** 2
                                raw_proximity += proximity_risk * dt
            # Normalize: divide by (num_whales × T) to make scale-independent
            whale_proximity_cost = w.w_whale_proximity * raw_proximity / max(num_whales * T, 1.0)

        # ── APR floor breach penalty (NORMALIZED) ──
        # Active only when apr_floor > 0 and apr_floor_sensitivity > 0.
        # Penalizes timesteps where total APR drops below the floor.
        # Uses LINEAR relative breach (|breach| / floor) rather than squared,
        # because squared is too weak for small breaches (a 10% relative breach
        # becomes 1% after squaring — insufficient to outweigh spend savings).
        # Linear gives a proportional penalty that balances well with w_spend.
        # Scaled by sensitivity (0-1) — at 1.0 this is the full weight.
        floor_breach_cost = 0.0
        time_below_floor_frac = 0.0
        if w.apr_floor > 0 and w.apr_floor_sensitivity > 0:
            floor_breach = np.maximum(0, w.apr_floor - apr_total)
            floor_norm = max(w.apr_floor, 1e-6)
            # Time-averaged linear breach, normalized by floor
            floor_linear = np.mean(floor_breach) / floor_norm
            # Also compute squared term for steep penalty on large breaches
            floor_quadratic = np.mean(floor_breach**2) / (floor_norm**2)
            # Combined: linear dominates for small breaches, quadratic for large
            floor_raw = floor_linear + floor_quadratic
            # Scale by sensitivity — at 1.0 this is the full weight,
            # at 0.5 it's half as important
            floor_breach_cost = w.w_apr_floor * w.apr_floor_sensitivity * floor_raw
            # Diagnostic: fraction of time below floor
            time_below_floor_frac = float(np.mean(apr_total < w.apr_floor))

        # ── Total loss ──
        total_loss = (
            spend_cost
            + apr_variance_cost
            + apr_ceiling_cost
            + tvl_shortfall_cost
            + merkl_fee_cost
            + budget_waste_cost
            + mercenary_cost
            + whale_proximity_cost
            + floor_breach_cost
        )

        # ── Diagnostics ──
        cap_binding = np.array([config.is_cap_binding(t) for t in tvl])
        sensitive_frac = state.sensitive_tvl / max(state.tvl, 1.0) if state.tvl > 0 else 0.0

        return LossResult(
            total_loss=total_loss,
            spend_cost=spend_cost,
            apr_variance_cost=apr_variance_cost,
            apr_ceiling_cost=apr_ceiling_cost,
            tvl_shortfall_cost=tvl_shortfall_cost,
            merkl_fee_cost=merkl_fee_cost,
            budget_waste_cost=budget_waste_cost,
            mercenary_cost=mercenary_cost,
            whale_proximity_cost=whale_proximity_cost,
            floor_breach_cost=floor_breach_cost,
            avg_apr=float(np.mean(apr_total)),
            avg_incentive_apr=float(np.mean(apr_incentive)),
            apr_std=float(np.std(apr_total)),
            apr_p5=float(np.percentile(apr_total, 5)),
            apr_p95=float(np.percentile(apr_total, 95)),
            avg_tvl=float(np.mean(tvl)),
            tvl_min=float(np.min(tvl)),
            tvl_max=float(np.max(tvl)),
            total_spend=total_spent,
            budget_utilization=total_spent / total_budget if total_budget > 0 else 0,
            max_cascade_depth=state.max_cascade_depth,
            mercenary_fraction=(max(state.mercenary_tvl, 0) / max(np.max(tvl), 1)),
            sensitive_fraction=sensitive_frac,
            time_cap_binding=float(np.mean(cap_binding)),
            time_below_floor=time_below_floor_frac,
        )


# ============================================================================
# MONTE CARLO RUNNER (M paths for one (B, r_max) point)
# ============================================================================


@dataclass
class MonteCarloResult:
    """Aggregated results from M simulation paths at one (B, r_max) point."""

    B: float
    r_max: float
    t_bind: float
    base_apy: float = 0.0  # For reference

    # Aggregated loss
    mean_loss: float = 0.0
    std_loss: float = 0.0
    loss_components: dict[str, float] = field(default_factory=dict)

    # Aggregated diagnostics
    mean_apr: float = 0.0  # TOTAL APR
    mean_incentive_apr: float = 0.0  # Incentive only
    std_apr: float = 0.0
    apr_p5: float = 0.0
    apr_p95: float = 0.0
    mean_tvl: float = 0.0
    tvl_min_p5: float = 0.0  # 5th percentile of path minimums
    tvl_max_p95: float = 0.0  # 95th percentile of path maximums
    mean_spend: float = 0.0
    mean_budget_util: float = 0.0
    mean_cascade_depth: float = 0.0
    max_cascade_depth: int = 0
    mean_mercenary_fraction: float = 0.0
    mean_time_cap_binding: float = 0.0

    # APR floor diagnostics
    mean_time_below_floor: float = 0.0  # avg fraction of time below floor

    # Feasibility
    is_feasible: bool = True
    infeasibility_reason: str = ""

    # Raw path results (for detailed analysis)
    path_results: list[LossResult] = field(default_factory=list)


def run_monte_carlo(
    config: CampaignConfig,
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
) -> MonteCarloResult:
    """
    Run M Monte Carlo paths for a single (B, r_max) configuration.

    Args:
        config: Campaign configuration
        env: Environment (competitor rates, etc.)
        initial_tvl: Starting TVL
        whale_profiles: Whale depositor profiles
        weights: Loss functional weights
        n_paths: Number of Monte Carlo paths
        retail_config: Retail depositor config (defaults if None)
        mercenary_config: Mercenary capital config (defaults if None)
        base_seed: Base random seed
        cascade_tolerance: Max acceptable mean cascade depth

    Returns:
        MonteCarloResult with aggregated statistics
    """
    evaluator = CampaignLossEvaluator(weights)
    results: list[LossResult] = []

    for m in range(n_paths):
        seed = base_seed + m * 1000

        # Build engine for this path
        engine = CampaignSimulationEngine.from_params(
            config=config,
            env=env,
            whale_profiles=whale_profiles,
            retail_config=retail_config,
            mercenary_config=mercenary_config,
            apy_sensitive_config=apy_sensitive_config,
            seed=seed,
        )

        # Build initial state
        initial_state = CampaignState(
            tvl=initial_tvl,
            budget_remaining_epoch=config.weekly_budget,
        )

        # Run simulation
        final_state = engine.run(initial_state)

        # Evaluate loss
        loss_result = evaluator.evaluate(final_state, config)
        results.append(loss_result)

    # Aggregate
    losses = np.array([r.total_loss for r in results])
    cascade_depths = np.array([r.max_cascade_depth for r in results])

    # Feasibility check
    mean_cascade = float(np.mean(cascade_depths))
    is_feasible = mean_cascade <= cascade_tolerance
    reason = ""
    if not is_feasible:
        reason = f"Mean cascade depth {mean_cascade:.1f} exceeds tolerance {cascade_tolerance}"

    return MonteCarloResult(
        B=config.weekly_budget,
        r_max=config.apr_cap,
        t_bind=config.t_bind,
        base_apy=config.base_apy,
        mean_loss=float(np.mean(losses)),
        std_loss=float(np.std(losses)),
        loss_components={
            "spend": float(np.mean([r.spend_cost for r in results])),
            "apr_variance": float(np.mean([r.apr_variance_cost for r in results])),
            "apr_ceiling": float(np.mean([r.apr_ceiling_cost for r in results])),
            "tvl_shortfall": float(np.mean([r.tvl_shortfall_cost for r in results])),
            "merkl_fee": float(np.mean([r.merkl_fee_cost for r in results])),
            "budget_waste": float(np.mean([r.budget_waste_cost for r in results])),
            "mercenary": float(np.mean([r.mercenary_cost for r in results])),
            "whale_proximity": float(np.mean([r.whale_proximity_cost for r in results])),
            "floor_breach": float(np.mean([r.floor_breach_cost for r in results])),
        },
        mean_apr=float(np.mean([r.avg_apr for r in results])),
        mean_incentive_apr=float(np.mean([r.avg_incentive_apr for r in results])),
        std_apr=float(np.mean([r.apr_std for r in results])),
        apr_p5=float(np.mean([r.apr_p5 for r in results])),
        apr_p95=float(np.mean([r.apr_p95 for r in results])),
        mean_tvl=float(np.mean([r.avg_tvl for r in results])),
        tvl_min_p5=float(np.percentile([r.tvl_min for r in results], 5)),
        tvl_max_p95=float(np.percentile([r.tvl_max for r in results], 95)),
        mean_spend=float(np.mean([r.total_spend for r in results])),
        mean_budget_util=float(np.mean([r.budget_utilization for r in results])),
        mean_cascade_depth=mean_cascade,
        max_cascade_depth=int(np.max(cascade_depths)),
        mean_mercenary_fraction=float(np.mean([r.mercenary_fraction for r in results])),
        mean_time_cap_binding=float(np.mean([r.time_cap_binding for r in results])),
        mean_time_below_floor=float(np.mean([r.time_below_floor for r in results])),
        is_feasible=is_feasible,
        infeasibility_reason=reason,
        path_results=results,
    )

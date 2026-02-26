"""
Agent classes for campaign simulation.

Three agent types following observe → decide → execute:
1. RetailDepositorAgent: aggregate continuous flow (drift + diffusion)
2. WhaleAgent: individual strategic actors (Stackelberg followers)
3. MercenaryAgent: reactive APR-chasing capital

All agents use seeded RNG for deterministic replay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .state import CampaignConfig, CampaignEnvironment, CampaignState


# ============================================================================
# BASE AGENT
# ============================================================================


class CampaignAgent(ABC):
    """Base class for campaign simulation agents."""

    def __init__(self, agent_id: str, seed: int | None = None):
        self.agent_id = agent_id
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def act(
        self,
        state: CampaignState,
        config: CampaignConfig,
        env: CampaignEnvironment,
    ) -> None:
        """
        Observe → decide → execute in one call.

        Mutates state in-place (consistent with lending sim pattern).
        """
        pass


# ============================================================================
# RETAIL DEPOSITOR AGENT (aggregate, continuous)
# ============================================================================


@dataclass
class RetailDepositorConfig:
    """Configuration for retail depositor behavior."""

    alpha_plus: float = 0.15
    """
    Inflow elasticity: fractional TVL growth per unit APR excess per day.
    e.g., 0.15 means if APR exceeds threshold by 1% (0.01), retail TVL grows
    at 0.15 * 0.01 = 0.15% per day.

    Calibration note: 0.15 is conservative. On a $600M pool with 1% APR
    excess, this gives ~$900k/day inflow. Institutional stablecoin pools
    grow more slowly than DeFi native pools.
    """

    alpha_minus_multiplier: float = 3.0
    """
    Outflow elasticity as multiple of alpha_plus.
    Conservative assumption: capital exits 3x faster than it enters.
    """

    response_lag_days: float = 5.0
    """
    Delay between APR change and depositor response.
    Retail depositors check rates weekly, not per-block.
    """

    diffusion_sigma: float = 0.008
    """
    Daily volatility of TVL (annualized ~15%).
    Captures routine noise from small depositors entering/exiting.

    Calibration note: 0.008 is conservative for institutional stablecoin
    pools. DeFi-native pools may warrant 0.015-0.025.
    """

    @property
    def alpha_minus(self) -> float:
        return self.alpha_plus * self.alpha_minus_multiplier


class RetailDepositorAgent(CampaignAgent):
    """
    Aggregate retail depositor behavior.

    Models the mass of small depositors as continuous drift + diffusion.
    Not modeled individually — their effect is the SDE:

    dTVL_retail = μ(t) · TVL_retail · dt + σ · TVL_retail · dW

    where μ depends on lagged APR relative to competitor threshold.
    """

    def __init__(
        self,
        config: RetailDepositorConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__("retail_depositors", seed)
        self.config = config or RetailDepositorConfig()
        self._apr_buffer: list[float] = []  # ring buffer for lag

    def act(
        self,
        state: CampaignState,
        config: CampaignConfig,
        env: CampaignEnvironment,
    ) -> None:
        """
        Apply retail depositor flow to TVL.

        Uses lagged APR for drift, current noise for diffusion.
        Only operates on the retail portion of TVL.
        """
        dt = config.dt_days

        # Record current APR for future lagged reference
        current_apr = config.realized_apr(state.tvl)
        self._apr_buffer.append(current_apr)

        # Get lagged APR (or current if not enough history)
        lag_steps = int(self.config.response_lag_days / dt)
        if len(self._apr_buffer) > lag_steps:
            lagged_apr = self._apr_buffer[-lag_steps - 1]
        else:
            lagged_apr = current_apr

        # Compute drift (asymmetric response)
        apr_gap = lagged_apr - env.r_threshold
        if apr_gap > 0:
            mu = self.config.alpha_plus * apr_gap
        else:
            mu = -self.config.alpha_minus * abs(apr_gap)

        # Apply to retail TVL only
        retail_tvl = state.retail_tvl
        if retail_tvl <= 0:
            return

        # Drift component
        drift = mu * retail_tvl * dt

        # Diffusion component (Brownian noise)
        noise = self.config.diffusion_sigma * retail_tvl * np.sqrt(dt) * self.rng.randn()

        # Apply change (respect supply cap if set)
        state.apply_tvl_change(drift + noise, supply_cap=config.supply_cap)

    def reset(self) -> None:
        """Reset lag buffer for new simulation path."""
        self._apr_buffer.clear()


# ============================================================================
# WHALE AGENT (individual, strategic, Stackelberg follower)
# ============================================================================


@dataclass
class WhaleProfile:
    """
    Profile for a single whale depositor.

    Calibrated from on-chain data and behavioral analysis.
    """

    whale_id: str
    position_usd: float  # Current deposit size

    # Exit threshold components
    alt_rate: float = 0.05  # Best alternative APR available
    risk_premium: float = 0.005  # Perceived risk of this venue
    switching_cost_usd: float = 500.0  # Gas + slippage to move position

    # Behavioral parameters
    exit_delay_days: float = 2.0  # Reaction time before exiting
    reentry_delay_days: float = 7.0  # Longer delay to re-enter
    hysteresis_band: float = 0.005  # APR must exceed threshold + band to re-enter

    # Classification (affects behavior modeling)
    whale_type: str = "institutional"
    # "institutional" = sticky, high switching cost, long delays
    # "quant_desk" = strategic, medium switching cost, responds to precise thresholds
    # "opportunistic" = less sticky, low switching cost, fast response

    @property
    def exit_threshold(self) -> float:
        """
        APR below which this whale exits.

        r_k^threshold = r_k^alt + c_k^risk - c_k^switch / d_k
        """
        per_dollar_switching = self.switching_cost_usd / max(self.position_usd, 1.0)
        return self.alt_rate + self.risk_premium - per_dollar_switching

    @property
    def reentry_threshold(self) -> float:
        """APR above which this whale re-enters (exit_threshold + hysteresis)."""
        return self.exit_threshold + self.hysteresis_band


class WhaleAgent(CampaignAgent):
    """
    Individual whale depositor (Stackelberg follower).

    Each whale has an exit threshold derived from their utility function.
    Whales observe APR and exit/re-enter based on threshold comparison,
    with stochastic delays modeling reaction time.

    Cascade dynamics: when one whale exits and APR changes, other whales
    are immediately re-evaluated within the same timestep.
    """

    def __init__(self, profile: WhaleProfile, seed: int | None = None):
        super().__init__(f"whale_{profile.whale_id}", seed)
        self.profile = profile
        self._time_below_threshold: float = 0.0  # accumulator for exit delay
        self._time_above_threshold: float = 0.0  # accumulator for re-entry delay
        self._has_exited: bool = False

    @property
    def has_exited(self) -> bool:
        return self._has_exited

    @property
    def position(self) -> float:
        return self.profile.position_usd

    def act(
        self,
        state: CampaignState,
        config: CampaignConfig,
        env: CampaignEnvironment,
    ) -> None:
        """
        Whale observes APR and decides whether to exit or re-enter.

        Exit logic:
        - If APR < exit_threshold, accumulate time below threshold
        - Once accumulated time exceeds exit_delay, execute exit
        - Add stochastic jitter to delay (±20%)

        Re-entry logic:
        - If exited and APR > reentry_threshold, accumulate time above
        - Once accumulated time exceeds reentry_delay, execute re-entry
        - Re-entry delay is longer than exit delay (asymmetric)
        """
        dt = config.dt_days
        current_apr = config.realized_apr(state.tvl)

        if not self._has_exited:
            self._evaluate_exit(current_apr, dt, state)
        else:
            self._evaluate_reentry(current_apr, dt, state, config)

    def _evaluate_exit(self, current_apr: float, dt: float, state: CampaignState) -> None:
        """Check if whale should exit."""
        if current_apr < self.profile.exit_threshold:
            self._time_below_threshold += dt
            # Stochastic delay: base delay ± 20%
            jitter = 1.0 + 0.2 * (self.rng.rand() - 0.5)
            effective_delay = self.profile.exit_delay_days * jitter

            if self._time_below_threshold >= effective_delay:
                # Execute exit
                state.apply_whale_exit(self.profile.whale_id, self.profile.position_usd)
                self._has_exited = True
                self._time_below_threshold = 0.0
                self._time_above_threshold = 0.0
        else:
            # Reset accumulator (APR recovered before delay elapsed)
            self._time_below_threshold = max(0.0, self._time_below_threshold - dt * 0.5)
            # Decay rather than hard reset — captures "partial concern"

    def _evaluate_reentry(
        self,
        current_apr: float,
        dt: float,
        state: CampaignState,
        config: CampaignConfig | None = None,
    ) -> None:
        """Check if whale should re-enter."""
        if current_apr > self.profile.reentry_threshold:
            self._time_above_threshold += dt
            jitter = 1.0 + 0.2 * (self.rng.rand() - 0.5)
            effective_delay = self.profile.reentry_delay_days * jitter

            if self._time_above_threshold >= effective_delay:
                # Execute re-entry (respect supply cap)
                cap = config.supply_cap if config else 0.0
                state.apply_whale_reentry(
                    self.profile.whale_id,
                    self.profile.position_usd,
                    supply_cap=cap,
                )
                self._has_exited = False
                self._time_above_threshold = 0.0
                self._time_below_threshold = 0.0
        else:
            self._time_above_threshold = max(0.0, self._time_above_threshold - dt * 0.5)

    def force_cascade_check(self, state: CampaignState, config: CampaignConfig) -> bool:
        """
        Immediate cascade evaluation (no delay).

        Called when another whale exits and APR changes.
        Returns True if this whale also exits.

        Cascade exits happen faster than normal exits — whales seeing
        a peer exit accelerate their own decision.
        """
        if self._has_exited:
            return False

        current_apr = config.realized_apr(state.tvl)
        if current_apr < self.profile.exit_threshold:
            # Cascade: exit with reduced delay (immediate if already accumulating)
            cascade_threshold = self.profile.exit_delay_days * 0.3
            if self._time_below_threshold >= cascade_threshold:
                state.apply_whale_exit(self.profile.whale_id, self.profile.position_usd)
                self._has_exited = True
                self._time_below_threshold = 0.0
                return True
            else:
                # Accelerate accumulation (seeing peer exit increases urgency)
                self._time_below_threshold += self.profile.exit_delay_days * 0.5
        return False

    def reset(self) -> None:
        """Reset for new simulation path."""
        self._has_exited = False
        self._time_below_threshold = 0.0
        self._time_above_threshold = 0.0


# ============================================================================
# MERCENARY AGENT (reactive, APR-chasing)
# ============================================================================


@dataclass
class MercenaryConfig:
    """Configuration for mercenary capital behavior."""

    entry_threshold: float = 0.08
    """APR above which mercenary capital enters (they want outsized yields)."""

    exit_threshold: float = 0.06
    """APR below which mercenary capital exits (tight band)."""

    max_capital_usd: float = 20_000_000.0
    """Maximum mercenary capital that can enter (market depth constraint)."""

    entry_rate_per_day: float = 0.3
    """Fraction of max_capital that enters per day when conditions are met."""

    exit_rate_per_day: float = 0.5
    """Fraction of current mercenary TVL that exits per day (faster than entry)."""

    entry_delay_days: float = 1.0
    """Minimum time APR must be above entry threshold before capital flows in."""

    noise_fraction: float = 0.1
    """Random variation in entry/exit amounts (±10%)."""


class MercenaryAgent(CampaignAgent):
    """
    Opportunistic capital that enters for outsized APR and exits quickly.

    Captures the "mercenary TVL" risk that the client wants to avoid.
    Configurations that frequently trigger mercenary entry/exit score
    poorly on the loss functional.
    """

    def __init__(
        self,
        config: MercenaryConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__("mercenary_capital", seed)
        self.config = config or MercenaryConfig()
        self._time_above_entry: float = 0.0

    def act(
        self,
        state: CampaignState,
        config: CampaignConfig,
        env: CampaignEnvironment,
    ) -> None:
        """
        Mercenary capital enters when APR spikes, exits when it normalizes.
        """
        dt = config.dt_days
        current_apr = config.realized_apr(state.tvl)
        merc_cfg = self.config

        if current_apr > merc_cfg.entry_threshold:
            # Accumulate time above entry threshold
            self._time_above_entry += dt

            if self._time_above_entry >= merc_cfg.entry_delay_days:
                # Enter: fraction of remaining capacity per day
                remaining_capacity = merc_cfg.max_capital_usd - state.mercenary_tvl
                if remaining_capacity > 0:
                    entry_amount = remaining_capacity * merc_cfg.entry_rate_per_day * dt
                    # Add noise
                    noise = 1.0 + merc_cfg.noise_fraction * (self.rng.rand() - 0.5)
                    entry_amount *= max(0, noise)
                    entry_amount = min(entry_amount, remaining_capacity)
                    if entry_amount > 0:
                        state.apply_mercenary_entry(entry_amount, supply_cap=config.supply_cap)

        elif current_apr < merc_cfg.exit_threshold and state.mercenary_tvl > 0:
            # Exit: fraction of current mercenary TVL per day
            exit_amount = state.mercenary_tvl * merc_cfg.exit_rate_per_day * dt
            noise = 1.0 + merc_cfg.noise_fraction * (self.rng.rand() - 0.5)
            exit_amount *= max(0, noise)
            if exit_amount > 0:
                state.apply_mercenary_exit(exit_amount)
            self._time_above_entry = 0.0

        else:
            # In between thresholds — slow decay of accumulator
            self._time_above_entry = max(0.0, self._time_above_entry - dt * 0.3)

    def reset(self) -> None:
        """Reset for new simulation path."""
        self._time_above_entry = 0.0


# ============================================================================
# WHALE CASCADE RESOLVER
# ============================================================================


def resolve_cascades(
    whale_agents: list[WhaleAgent],
    state: CampaignState,
    config: CampaignConfig,
    max_cascade_depth: int = 10,
) -> int:
    """
    Resolve whale exit cascades within a single timestep.

    When one whale exits:
    1. TVL drops
    2. APR recalculates (may rise under Float, capped under Hybrid)
    3. Other whales re-evaluate — if new APR crosses their threshold, they may exit too
    4. Repeat until no more exits or max depth reached

    Args:
        whale_agents: List of all whale agents
        state: Current campaign state (mutated in place)
        config: Campaign configuration
        max_cascade_depth: Safety limit to prevent infinite loops

    Returns:
        Cascade depth (0 = no cascade, 1 = one additional exit, etc.)
    """
    depth = 0
    for _ in range(max_cascade_depth):
        triggered = False
        # Shuffle order to avoid bias (whales don't have perfect ordering)
        indices = list(range(len(whale_agents)))
        np.random.shuffle(indices)

        for idx in indices:
            whale = whale_agents[idx]
            if whale.force_cascade_check(state, config):
                triggered = True
                depth += 1

        if not triggered:
            break

    state.max_cascade_depth = max(state.max_cascade_depth, depth)
    return depth


# ============================================================================
# APY-SENSITIVE AGENT (floor-sensitive depositors: loopers, yield chasers, etc.)
# ============================================================================


@dataclass
class APYSensitiveConfig:
    """
    Configuration for APY-sensitive depositors in a vault.

    Models any depositor class with a hard APR floor — including leveraged
    loopers, yield-chasing funds, and institutional allocations with
    minimum return requirements. When APR drops below their floor,
    these depositors exit (potentially unwinding leveraged positions).

    For loopers specifically: unwind removes leverage_multiple × real_capital
    from headline TVL (e.g., 3x leverage → $10M real = $30M headline).
    For non-leveraged sensitive depositors, set leverage_multiple = 1.0.
    """

    floor_apr: float = 0.0
    """
    Minimum total APR (base + incentive) below which sensitive positions
    become unprofitable and start exiting. Set by user based on
    knowledge of vault composition.
    0.0 = disabled (no APY sensitivity).
    """

    sensitivity: float = 0.0
    """
    How aggressively depositors exit when APR drops below floor.
    0.0 = disabled (no sensitive behavior).
    0.5 = moderate — 1.5 day delay before exit starts.
    1.0 = maximum — immediate exit on floor breach.

    Controls exit delay: effective_delay = (1 - sensitivity) * max_delay_days.
    At sensitivity=1.0, delay is 0 (instant exit on breach).
    """

    leverage_multiple: float = 3.0
    """
    Average leverage of sensitive positions.
    3.0 means $10M real capital → $30M headline supply, $20M borrow.
    Unwind removes the full $30M from TVL (supply side).
    Set to 1.0 for non-leveraged APY-sensitive depositors.
    """

    max_sensitive_tvl: float = 0.0
    """
    Total headline TVL attributable to sensitive positions (USD).
    e.g., if 3 loopers each have $10M real capital at 3x leverage,
    max_sensitive_tvl = 3 × $10M × 3 = $90M headline.
    0.0 = auto-estimate as 10% of initial TVL.
    """

    unwind_rate_per_day: float = 0.4
    """
    Fraction of remaining sensitive TVL that exits per day once triggered.
    0.4 = 40%/day — faster than retail (0.15), slower than mercenary exit (0.5).
    Leveraged unwinds are mechanical (closing leverage), not discretionary.
    """

    reentry_rate_per_day: float = 0.1
    """
    Fraction of max capacity that re-enters per day once APR recovers.
    0.1 = 10%/day — cautious. Re-leveraging requires borrowing, gas, slippage.
    Much slower than exit (asymmetric, like whale re-entry).
    """

    max_delay_days: float = 3.0
    """
    Maximum delay before exit begins (at sensitivity=0).
    Actual delay = max_delay_days × (1 - sensitivity).
    At sensitivity=0.5: 1.5 day delay.
    At sensitivity=1.0: 0 day delay (immediate).
    """

    hysteresis_band: float = 0.005
    """
    APR must exceed floor_apr + hysteresis_band before depositors re-enter.
    Prevents oscillation at the floor boundary.
    """


class APYSensitiveAgent(CampaignAgent):
    """
    APY-floor-sensitive depositor agent — models positions that exit
    when total APR drops below a configurable floor.

    Covers leveraged loopers (recursive borrow/supply), yield-chasing
    funds, and any depositor with a hard minimum return requirement.

    Key dynamics:
    - TVL grows → APR dilutes → margin compresses → exit
    - Exit removes leverage_multiple × real_capital from headline TVL
    - TVL drop after exit → APR spikes → attracts mercenary/retail
    - New TVL → APR drops again → more exits (oscillation)

    Under MAX campaigns:
    - APR spike after exit is capped → less mercenary attraction
    - Less reactive TVL overshoot → fewer subsequent exits
    - Dampened oscillation → rate stability → depositors stay

    This is why MAX is optimal for APY-sensitive venues.
    """

    def __init__(
        self,
        config: APYSensitiveConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__("apy_sensitive_positions", seed)
        self.config = config or APYSensitiveConfig()
        self._time_below_floor: float = 0.0
        self._time_above_floor: float = 0.0
        self._current_sensitive_tvl: float = 0.0  # Tracks active sensitive headline TVL
        self._initialized: bool = False

    def act(
        self,
        state: CampaignState,
        config: CampaignConfig,
        env: CampaignEnvironment,
    ) -> None:
        """
        APY-sensitive agent observes total APR and exits if below floor.

        Exit: when total APR < floor_apr for effective_delay days,
        sensitive positions exit at unwind_rate_per_day.

        Re-entry: when total APR > floor_apr + hysteresis for sustained
        period, depositors slowly re-enter.

        TVL impact: exit removes headline TVL (leveraged amount),
        which is leverage_multiple × real capital.
        """
        lc = self.config

        # Skip if disabled
        if lc.sensitivity <= 0 or lc.floor_apr <= 0:
            return

        dt = config.dt_days

        # Initialize sensitive TVL on first step
        if not self._initialized:
            if lc.max_sensitive_tvl > 0:
                self._current_sensitive_tvl = lc.max_sensitive_tvl
            else:
                # Auto-estimate: 10% of current TVL is sensitive headline
                self._current_sensitive_tvl = state.tvl * 0.10
            # Register in state — this TVL already exists within current deposits
            state.sensitive_tvl = self._current_sensitive_tvl
            self._initialized = True

        current_apr = config.realized_apr(state.tvl)

        if current_apr < lc.floor_apr:
            # APR is below profitability floor
            self._time_below_floor += dt
            self._time_above_floor = 0.0

            # Effective delay: shorter at higher sensitivity
            effective_delay = lc.max_delay_days * (1.0 - lc.sensitivity)

            if self._time_below_floor >= effective_delay and self._current_sensitive_tvl > 0:
                # Exit: remove fraction of remaining sensitive TVL
                unwind_amount = self._current_sensitive_tvl * lc.unwind_rate_per_day * dt

                # Add noise (±15%)
                noise = 1.0 + 0.15 * (self.rng.rand() - 0.5)
                unwind_amount *= max(0, noise)
                unwind_amount = min(unwind_amount, self._current_sensitive_tvl)

                if unwind_amount > 0:
                    # Apply as sensitive-position TVL removal
                    state.apply_sensitive_unwind(unwind_amount)
                    self._current_sensitive_tvl = max(
                        0, self._current_sensitive_tvl - unwind_amount
                    )

        elif current_apr > lc.floor_apr + lc.hysteresis_band:
            # APR recovered above floor + hysteresis — depositors may re-enter
            self._time_above_floor += dt
            self._time_below_floor = max(0, self._time_below_floor - dt * 0.3)

            # Re-entry delay: 2 days minimum (cautious)
            if self._time_above_floor >= 2.0:
                max_tvl = lc.max_sensitive_tvl if lc.max_sensitive_tvl > 0 else state.tvl * 0.10
                capacity = max_tvl - self._current_sensitive_tvl
                if capacity > 0:
                    reentry_amount = capacity * lc.reentry_rate_per_day * dt
                    noise = 1.0 + 0.10 * (self.rng.rand() - 0.5)
                    reentry_amount *= max(0, noise)
                    reentry_amount = min(reentry_amount, capacity)

                    if reentry_amount > 0:
                        state.apply_sensitive_entry(reentry_amount, supply_cap=config.supply_cap)
                        self._current_sensitive_tvl += reentry_amount
        else:
            # In the hysteresis band — slow decay of accumulators
            self._time_below_floor = max(0, self._time_below_floor - dt * 0.5)
            self._time_above_floor = max(0, self._time_above_floor - dt * 0.3)

    def reset(self) -> None:
        """Reset for new simulation path."""
        self._time_below_floor = 0.0
        self._time_above_floor = 0.0
        self._current_sensitive_tvl = 0.0
        self._initialized = False

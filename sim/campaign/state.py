"""
State objects for campaign simulation.

CampaignState: mutable state of a running campaign (TVL, spend, APR history)
CampaignConfig: immutable campaign parameters (B, r_max, epoch duration)
CampaignEnvironment: exogenous conditions (competitor rates, time)

Key distinction:
- incentive_apr(t): the Merkl incentive rate = min(r_max, B/TVL * 365/7)
- base_apy: the venue's organic/base yield (supply rate, trading fees, vault APY)
- realized_apr(t) = base_apy + incentive_apr(t): what depositors actually see

Depositor drift responds to realized_apr vs competitor r_threshold (both total).
Spend cost is computed on incentive_apr only (that's what Merkl disburses).
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# CAMPAIGN CONFIGURATION (immutable per simulation run)
# ============================================================================


@dataclass(frozen=True)
class CampaignConfig:
    """
    Immutable campaign parameters for a single simulation run.

    Represents one point on the (B, r_max) surface.
    """

    # Core campaign parameters (the 2D surface)
    weekly_budget: float  # B (USD/week)
    apr_cap: float  # r_max (decimal, e.g. 0.07 for 7%)

    # Base / organic yield for this venue (NOT from Merkl incentives)
    # Sources: Aave supply rate, Morpho allocation yield, Curve fees, etc.
    base_apy: float = 0.0  # decimal (0.03 = 3%)

    # Epoch structure
    epoch_duration_days: int = 7
    merkl_fee_rate: float = 0.015  # 1.5% on SPENT budget (unspent is refunded, no fee)

    # Simulation parameters
    dt_days: float = 0.25  # timestep (6 hours default — sub-daily for cascade resolution)
    horizon_days: int = 28  # 4-week simulation horizon

    # Supply cap (0 = unlimited)
    supply_cap: float = 0.0  # USD — max TVL the venue can accept

    # Whale profiles for proximity penalty calculation
    whale_profiles: tuple = ()  # tuple of WhaleProfile (immutable)

    @property
    def daily_budget(self) -> float:
        """Budget per day (USD/day)."""
        return self.weekly_budget / 7.0

    @property
    def t_bind(self) -> float:
        """
        TVL threshold where regime switches from Float-like to MAX-like.

        T_bind = B * (365/7) / r_max

        Note: defined on the INCENTIVE rate only, not total.
        """
        if self.apr_cap <= 0:
            return float("inf")
        return self.weekly_budget * (365.0 / 7.0) / self.apr_cap

    @property
    def num_steps(self) -> int:
        """Total simulation steps."""
        return int(self.horizon_days / self.dt_days)

    @property
    def num_epochs(self) -> int:
        """Number of complete epochs in simulation horizon."""
        return self.horizon_days // self.epoch_duration_days

    def incentive_apr(self, tvl: float) -> float:
        """
        Compute Merkl incentive APR given current TVL.

        r_incentive(t) = min(r_max, B / TVL(t) * 365/7)

        This is what Merkl pays out. Capped at r_max.
        """
        if tvl <= 0:
            return self.apr_cap
        float_apr = self.weekly_budget / tvl * (365.0 / 7.0)
        return min(self.apr_cap, float_apr)

    def realized_apr(self, tvl: float) -> float:
        """
        Total APR seen by depositors = base_apy + incentive_apr.

        This is what drives depositor behavior (drift model) and
        whale exit decisions. Depositors compare this total against
        competitor total yields.
        """
        return self.base_apy + self.incentive_apr(tvl)

    def instantaneous_spend_rate(self, tvl: float) -> float:
        """
        Instantaneous Merkl spend rate (USD/day) at given TVL.

        Spend is based on INCENTIVE rate only (not base).
        spend/day = TVL * r_incentive(t) / 365
        """
        r = self.incentive_apr(tvl)
        return tvl * r / 365.0

    def is_cap_binding(self, tvl: float) -> bool:
        """Whether the incentive APR cap is currently binding (TVL < T_bind)."""
        return tvl < self.t_bind


# ============================================================================
# CAMPAIGN ENVIRONMENT (exogenous, read-only for agents)
# ============================================================================


@dataclass
class CampaignEnvironment:
    """
    Exogenous market conditions.

    Agents read from this but don't modify it directly.
    The simulation engine advances time and can update competitor rates.

    IMPORTANT: r_threshold should be the TOTAL yield (base + incentive)
    at competing venues. Depositors compare total-to-total.
    """

    # Competitor landscape
    competitor_rates: dict[str, float] = field(default_factory=dict)
    # e.g. {"aave_pyusd": 0.0332, "euler_pyusd": 0.065, "kamino_earn": 0.0406}

    r_threshold: float = 0.045
    """
    Opportunity cost / indifference rate (decimal).
    Derived from weighted average of competitor incentive rates.
    Treated as range [r_threshold_lo, r_threshold_hi] for robustness.
    """

    r_threshold_lo: float = 0.04
    r_threshold_hi: float = 0.055

    # Time
    current_time_days: float = 0.0

    # Network conditions (for whale switching cost estimation)
    gas_price_gwei: float = 30.0
    eth_price_usd: float = 2500.0

    def step(self, dt_days: float) -> None:
        """Advance time (in-place mutation)."""
        self.current_time_days += dt_days

    def switching_cost_usd(self, gas_limit: int = 500_000) -> float:
        """Estimated switching cost in USD for a position change."""
        gas_cost_eth = (self.gas_price_gwei * gas_limit) / 1e9
        return gas_cost_eth * self.eth_price_usd

    def copy(self) -> CampaignEnvironment:
        """Shallow copy for parallel simulation paths."""
        new = copy(self)
        new.competitor_rates = dict(self.competitor_rates)
        return new


# ============================================================================
# CAMPAIGN STATE (mutable, tracks simulation evolution)
# ============================================================================


@dataclass
class CampaignState:
    """
    Mutable state of a running campaign at time t.

    Analogous to MarketState in the lending simulation.
    Tracks TVL, spend, APR, and whale positions over time.

    History arrays track TOTAL APR (base + incentive) since that's what
    depositors see. Spend tracks incentive cost only.
    """

    # Current state
    tvl: float  # Current TVL (USD)
    budget_remaining_epoch: float  # Unspent budget in current epoch
    budget_spent_total: float = 0.0  # Cumulative spend
    current_step: int = 0

    # Whale tracking
    whale_positions: dict[str, float] = field(default_factory=dict)
    # {whale_id: position_usd}
    whale_exited: dict[str, bool] = field(default_factory=dict)
    # {whale_id: has_exited}

    # History (recorded at each timestep)
    tvl_history: list[float] = field(default_factory=list)
    apr_history: list[float] = field(default_factory=list)  # TOTAL APR (base + incentive)
    incentive_apr_history: list[float] = field(default_factory=list)  # Incentive only
    spend_history: list[float] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)

    # Cascade tracking (per simulation)
    max_cascade_depth: int = 0
    total_whale_exits: int = 0
    total_whale_reentries: int = 0

    # Mercenary tracking
    mercenary_tvl: float = 0.0
    mercenary_entries: int = 0
    mercenary_exits: int = 0

    # APY-sensitive depositor tracking (loopers, yield chasers, etc.)
    sensitive_tvl: float = 0.0  # Current headline TVL from sensitive positions
    sensitive_unwinds: int = 0  # Count of unwind/exit events
    sensitive_entries: int = 0  # Count of re-entry events

    def record(self, config: CampaignConfig, env: CampaignEnvironment) -> None:
        """Record current state to history."""
        r_total = config.realized_apr(self.tvl)
        r_incentive = config.incentive_apr(self.tvl)
        spend_rate = config.instantaneous_spend_rate(self.tvl)

        self.tvl_history.append(self.tvl)
        self.apr_history.append(r_total)
        self.incentive_apr_history.append(r_incentive)
        self.spend_history.append(spend_rate)
        self.time_history.append(env.current_time_days)

    def apply_tvl_change(self, delta: float, supply_cap: float = 0.0) -> None:
        """
        Apply TVL change (positive = inflow, negative = outflow).

        Enforces TVL >= 0 and TVL <= supply_cap (if cap > 0).
        The supply cap bounds reflexive inflows: when a whale exits and APR spikes,
        inflows are bounded by the protocol's supply cap.
        """
        new_tvl = self.tvl + delta
        new_tvl = max(0.0, new_tvl)
        if supply_cap > 0:
            new_tvl = min(new_tvl, supply_cap)
        self.tvl = new_tvl

    def apply_whale_exit(self, whale_id: str, position: float) -> None:
        """Remove whale position from TVL."""
        self.tvl = max(0.0, self.tvl - position)
        self.whale_exited[whale_id] = True
        self.total_whale_exits += 1

    def apply_whale_reentry(self, whale_id: str, position: float, supply_cap: float = 0.0) -> None:
        """Re-add whale position to TVL (respects supply cap)."""
        new_tvl = self.tvl + position
        if supply_cap > 0:
            new_tvl = min(new_tvl, supply_cap)
        self.tvl = new_tvl
        self.whale_exited[whale_id] = False
        self.total_whale_reentries += 1

    def apply_mercenary_entry(self, amount: float, supply_cap: float = 0.0) -> None:
        """Add mercenary capital (respects supply cap)."""
        if supply_cap > 0:
            headroom = max(0.0, supply_cap - self.tvl)
            amount = min(amount, headroom)
        if amount <= 0:
            return
        self.tvl += amount
        self.mercenary_tvl += amount
        self.mercenary_entries += 1

    def apply_mercenary_exit(self, amount: float) -> None:
        """Remove mercenary capital."""
        actual = min(amount, self.mercenary_tvl)
        self.tvl = max(0.0, self.tvl - actual)
        self.mercenary_tvl = max(0.0, self.mercenary_tvl - actual)
        self.mercenary_exits += 1

    def apply_sensitive_unwind(self, headline_amount: float) -> None:
        """
        Remove APY-sensitive headline TVL (leveraged amount).

        An unwind removes the full position from supply.
        e.g., 3x leverage on $10M real = $30M headline removed from TVL.
        For non-leveraged sensitive depositors (leverage=1.0), removes 1:1.
        """
        actual = min(headline_amount, self.sensitive_tvl)
        self.tvl = max(0.0, self.tvl - actual)
        self.sensitive_tvl = max(0.0, self.sensitive_tvl - actual)
        self.sensitive_unwinds += 1

    def apply_sensitive_entry(self, headline_amount: float, supply_cap: float = 0.0) -> None:
        """
        Add APY-sensitive headline TVL (re-entry / re-leverage).
        Respects supply cap.
        """
        if supply_cap > 0:
            headroom = max(0.0, supply_cap - self.tvl)
            headline_amount = min(headline_amount, headroom)
        if headline_amount <= 0:
            return
        self.tvl += headline_amount
        self.sensitive_tvl += headline_amount
        self.sensitive_entries += 1

    def update_spend(self, config: CampaignConfig, dt_days: float) -> None:
        """
        Update budget accounting for this timestep.

        Spend = TVL * r_incentive(t) / 365 * dt_days
        (incentive spend only, not base)
        """
        spend = config.instantaneous_spend_rate(self.tvl) * dt_days
        self.budget_spent_total += spend
        self.budget_remaining_epoch -= spend

    def reset_epoch_budget(self, config: CampaignConfig) -> None:
        """Reset epoch budget (called at epoch boundaries)."""
        self.budget_remaining_epoch = config.weekly_budget

    @property
    def retail_tvl(self) -> float:
        """TVL attributable to retail (non-whale, non-mercenary)."""
        whale_total = sum(
            pos
            for wid, pos in self.whale_positions.items()
            if not self.whale_exited.get(wid, False)
        )
        return max(0.0, self.tvl - whale_total - self.mercenary_tvl - self.sensitive_tvl)

    def get_active_whale_tvl(self) -> float:
        """Total TVL from whales still in the venue."""
        return sum(
            pos
            for wid, pos in self.whale_positions.items()
            if not self.whale_exited.get(wid, False)
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert histories to numpy arrays for analysis."""
        result = {
            "time": np.array(self.time_history),
            "tvl": np.array(self.tvl_history),
            "apr": np.array(self.apr_history),
            "spend": np.array(self.spend_history),
        }
        if self.incentive_apr_history:
            result["incentive_apr"] = np.array(self.incentive_apr_history)
        return result

    def fast_clone(self) -> CampaignState:
        """
        Fast clone for Monte Carlo paths.

        Shares whale_positions dict (read-only reference data).
        Clones mutable tracking state.
        """
        clone = CampaignState(
            tvl=self.tvl,
            budget_remaining_epoch=self.budget_remaining_epoch,
            budget_spent_total=0.0,
            current_step=0,
            whale_positions=dict(self.whale_positions),  # copy — whales may exit
            whale_exited={wid: False for wid in self.whale_positions},
            tvl_history=[],
            apr_history=[],
            incentive_apr_history=[],
            spend_history=[],
            time_history=[],
            max_cascade_depth=0,
            total_whale_exits=0,
            total_whale_reentries=0,
            mercenary_tvl=0.0,
            mercenary_entries=0,
            mercenary_exits=0,
            sensitive_tvl=0.0,
            sensitive_unwinds=0,
            sensitive_entries=0,
        )
        return clone

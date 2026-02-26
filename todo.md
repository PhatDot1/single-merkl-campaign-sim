# Data Sync & Simulation Enhancement Plan

## Overview

Final ToDo's for Optimizer. Three pillars:

1. **Data Ingestion** — automated sync of whale behavior, competitor landscapes, TVL stickiness
2. **Simulation Realism** — caps, swap costs, liquidity constraints, empirical calibration
3. **Actionable Outputs** — regime-aware campaign suggestions that don't waste budget

---

## 1. Data Sync: What We Have vs What We Need

### 1.1 Currently Working
- [x] Live TVL + utilization from Aave V3 (Core/Horizon) and Euler V2 via RPC
- [x] Base APY from on-chain interest rate models
- [x] Whale positions from aToken/eToken holder scanning (Alchemy or eth_getLogs)
- [x] r_threshold from DeFiLlama stablecoin class benchmark (37+ pools)
- [x] Synthetic whale exit thresholds (based on position size tiers)
- [x] Supply/borrow caps from on-chain config words

###  1.15 Caps data pull not working

actually pulling caps data - currently might be not pulling properly and just getting "no cap unlimited" for everything, need to ensure we are actually getting caps properly for each venue and unit test properly before integrating in a seamless and working way.

For example, euler currently has a cap, aave pyusd has a cap of 500m right now, and morpho i think are currently unlimited. Lets ensure we are getting these caps peoperly!

### 1.2 Missing: Dune Data Sync

**Priority: HIGH — whale stickiness is the #1 blind spot**

#### 1.2.1 Whale Behavioral History (Dune)

Currently `build_whale_profiles_from_holders()` uses synthetic exit thresholds.
We need empirical data on how each whale actually behaves.

**Query: per-whale deposit/withdrawal history**
```
For each whale address found via on-chain scanning:
- Number of deposits in last 90/180/365 days
- Number of withdrawals (full or partial)
- Average hold duration between deposit and withdrawal
- Largest single withdrawal as % of their position
- Did they exit during any APR drop events? (correlate with rate history)
- Where did withdrawn funds go? (DEX swap → competitor venue?)
```

**Dune tables needed:**
- `aave_v3_ethereum.Pool_evt_Supply` — deposits by address
- `aave_v3_ethereum.Pool_evt_Withdraw` — withdrawals by address
- `erc20_ethereum.evt_Transfer` — Euler vault share transfers
- `dex.trades` — post-withdrawal swap routing (did they go to Morpho? another Aave market?)

**Output: `whale_history.json` per venue**
```json
{
  "0xabc...": {
    "n_deposits": 12,
    "n_withdrawals": 3,
    "avg_hold_days": 45.2,
    "max_withdrawal_pct": 0.65,
    "exits_during_rate_drops": 2,
    "destinations": ["morpho_steakhouse_usdc", "aave_core_usdc"],
    "last_seen": "2025-12-15",
    "stickiness_score": 0.72
  }
}
```

**Stickiness score derivation:**
- `base = 1.0 - (n_withdrawals / max(n_deposits, 1))`
- `hold_bonus = min(avg_hold_days / 90, 0.3)` — long holders get credit
- `exit_penalty = exits_during_rate_drops * 0.15` — rate-sensitive exits penalized
- `score = clamp(base + hold_bonus - exit_penalty, 0.1, 1.0)`
- Score → maps to `exit_delay_days` and `hysteresis_band` in WhaleProfile

#### 1.2.2 TVL Stickiness / Organic vs Incentivized (Dune)

**Priority: HIGH — determines how much TVL stays when incentives end**

**Query: TVL decomposition over time**
```
For each venue, over 90-day windows:
- Daily TVL snapshots
- Daily incentive rate (from Merkl distribution events or known campaign data)
- Correlation: when incentives dropped, how much TVL left? How fast?
- Residual TVL after incentive campaigns ended
- Organic growth rate (TVL change during periods with zero incentives)
```

**Dune tables:**
- `aave_v3_ethereum.Pool_evt_Supply` / `Withdraw` aggregated daily
- `merkl.RewardDistributed` or equivalent Merkl events
- Cross-reference with `lending.deposits` for daily TVL

**Output: `tvl_stickiness.json` per venue**
```json
{
  "venue": "euler_sentora_rlusd",
  "organic_tvl_estimate_usd": 45000000,
  "incentive_elasticity": 0.35,
  "mean_exit_lag_days": 4.2,
  "tvl_half_life_post_incentive_days": 12.5,
  "sticky_fraction": 0.42
}
```

**How this feeds the simulator:**
- `sticky_fraction` → sets a TVL floor that doesn't respond to rate changes
- `incentive_elasticity` → calibrates `alpha_plus` / `alpha_minus` in RetailDepositorConfig
- `mean_exit_lag_days` → calibrates `response_lag_days`
- `organic_tvl_estimate_usd` → if target TVL < organic, you may not need incentives at all

#### 1.2.3 Historical APR Time Series (Dune + DeFiLlama)

**Priority: MEDIUM — needed for calibration, not blocking**

**What we need:**
- Daily supply APR for every competitor venue over 90/180 days
- Correlation between APR changes and TVL flows (how rate-sensitive is capital?)
- APR volatility per venue (some venues have stable rates, others spike)

**Sources:**
- DeFiLlama `/yields/chart/{pool_id}` — historical APY time series
- Dune: `aave_v3_ethereum.Pool_evt_ReserveDataUpdated` — block-level rate changes
- Morpho: subgraph or direct vault `totalAssets()` historical via archive node

**Output: feeds `historical.py` → `calibrate_retail_params()`**

---

### 1.3 Missing: Competitor Venue Sync

**Priority: HIGH — r_threshold is only as good as the competitor data**

#### 1.3.1 Full Competitor Landscape Scan

Currently `fetch_stablecoin_class_benchmark()` pulls 37+ pools from DeFiLlama.
This is a broad average. We need **targeted competitor intelligence**.

**For each stablecoin (RLUSD, PYUSD, USDC):**

| Protocol | Source | What to Fetch |
|----------|--------|---------------|
| Aave V3 Core | RPC | supply APY, TVL, supply cap, borrow cap, utilization |
| Aave V3 Horizon | RPC | same |
| Euler V2 (all vaults for asset) | RPC | same + vault-specific caps |
| Morpho Blue (all markets for asset) | Subgraph/API | supply APY, TVL, caps, utilization |
| Morpho Vaults (MetaMorpho) | RPC | vault APY, TVL, allocation across markets |
| Spark (MakerDAO) | RPC | DSR/sDAI rate, TVL |
| Compound V3 | RPC | supply APY, TVL, caps |
| Fluid | RPC/API | supply APY, TVL |

**Critical enrichments per competitor:**
```python
@dataclass
class CompetitorVenue:
    protocol: str
    asset: str
    chain: str
    supply_apy: float          # Current base APY (no incentives)
    incentive_apy: float       # Current incentive APY (Merkl, ARB, OP, etc.)
    total_apy: float           # supply_apy + incentive_apy
    tvl_usd: float
    supply_cap_usd: float      # 0 = unlimited
    available_capacity_usd: float  # supply_cap - tvl (how much room)
    utilization: float
    swap_cost_bps: float       # Cost to move capital here (DEX routing)
    bridge_cost_usd: float     # If cross-chain (0 for same-chain)
    gas_cost_usd: float        # Typical deposit gas
    min_viable_position_usd: float  # Position size where costs < 1 week of yield
```

#### 1.3.2 Swap/Bridge Cost Adjustment

**Priority: HIGH — a 5% APR on Morpho means nothing if it costs 50bps to get there**

**r_threshold should be adjusted for switching costs:**
```
effective_competitor_rate = competitor_apy - annualized_switching_cost
annualized_switching_cost = (swap_fee_bps + gas_cost_bps) * (52 / avg_rebalance_weeks)
```

**For same-chain moves (e.g., Aave → Euler on Ethereum):**
- DEX swap cost: 1-5bps for stablecoins (1inch/Uniswap routing)
- Gas: ~$2-10 for withdraw + approve + deposit
- For $10M position: gas is negligible, swap is ~$500-5000

**For cross-chain moves:**
- Bridge cost: 5-20bps (Across, Stargate, Circle CCTP)
- Additional gas on destination chain
- Bridge delay: 10min-7days depending on bridge

**Implementation:**
- Add `swap_cost_discount_bps` to `RThresholdConfig` in `venue_registry.py`
- Apply discount when computing r_threshold: `r_thresh = benchmark - swap_discount`
- Currently hardcoded as `discount=0.50%` — should be per-venue based on actual routing

#### 1.3.3 Capacity-Aware Competitor Filtering

**Priority: HIGH — whales can't exit to a full pool**

**Current problem:** r_threshold treats all competitors equally regardless of capacity.
A Morpho vault at 95% of its supply cap with $2M remaining capacity is NOT a viable
exit for a $50M whale.

**Fix:**
```python
def compute_whale_aware_r_threshold(
    competitors: list[CompetitorVenue],
    whale_profiles: list[WhaleProfile],
) -> float:
    """
    Weight competitor rates by how much whale capital they can actually absorb.
    A capped-out competitor with 5% APY is irrelevant to a $50M whale.
    """
    max_whale = max(w.position_usd for w in whale_profiles)

    weighted_rates = []
    for c in competitors:
        # How much of the largest whale could this venue absorb?
        absorbable = min(c.available_capacity_usd, max_whale)
        if absorbable < max_whale * 0.1:
            continue  # Can't absorb even 10% of whale — irrelevant

        # Weight by absorptive capacity and rate
        effective_rate = c.total_apy - c.swap_cost_bps / 10000
        weight = absorbable / max_whale
        weighted_rates.append((effective_rate, weight))

    if not weighted_rates:
        return fallback_benchmark_rate

    return sum(r * w for r, w in weighted_rates) / sum(w for _, w in weighted_rates)
```

---

## 2. Simulation Realism Enhancements

### 2.1 Cap-Aware Campaign Optimization

**Priority: CRITICAL — the optimizer must understand caps**

**Current gap:** The optimizer searches (B, r_max) but doesn't deeply reason about
how the supply cap constrains the outcome. It should:

#### 2.1.1 Don't waste budget on MAX up to cap

**Problem:** Setting r_max = 7% with a $200k budget on a venue with a $190M cap
means you're paying 7% all the way up to $190M. But the float rate at $190M with
$200k/wk is only 5.5%. The extra 1.5% is wasted money that attracts mercenary
capital and doesn't accelerate TVL growth (already at cap).

**Fix: Cap-proximity taper**
```python
def optimal_r_max_near_cap(
    budget_weekly: float,
    target_tvl: float,
    supply_cap: float,
    base_apy: float,
    floor_incentive: float,
) -> float:
    """
    When target TVL is near the supply cap, set r_max just above the float
    rate at cap. This saves budget and provides better rate intelligence.

    The campaign should:
    1. Use float regime for growth (natural price discovery)
    2. Set r_max to bind just below cap (catch overshoots, save budget)
    3. Report what the "natural" equilibrium rate would be
    """
    float_rate_at_cap = budget_weekly * 52.14 / supply_cap
    float_rate_at_target = budget_weekly * 52.14 / target_tvl

    # If target is >90% of cap, taper r_max
    cap_proximity = target_tvl / supply_cap if supply_cap > 0 else 0
    if cap_proximity > 0.9:
        # r_max should be just above float rate at cap
        # This means: at cap, rate = r_max (tiny premium over float)
        # Below cap, rate = float (higher, natural price discovery)
        r_max = float_rate_at_cap * 1.05  # 5% headroom
    else:
        # Normal: r_max from optimizer
        r_max = max(float_rate_at_target, floor_incentive)

    return r_max
```

#### 2.1.2 Cap as hard constraint in MC paths

**Current:** The cap is somewhat modeled but not enforced as a hard wall.

**Fix:** In `engine.py`, when simulating TVL paths:
```python
# After computing new_tvl from retail/whale/merc dynamics:
if supply_cap > 0:
    new_tvl = min(new_tvl, supply_cap)
    # If TVL hits cap, the ACTUAL rate paid is r_max (not float)
    # This means spend = r_max * cap / 52.14 (could be < budget)
    actual_spend = min(budget, r_max * new_tvl / 52.14)
    # Track budget savings from cap binding
    budget_saved += budget - actual_spend
```

#### 2.1.3 Surface unused budget in results

**When cap binds and budget is underutilized:**
- Show "Budget utilization: 78% — $44k/wk unused due to supply cap"
- Suggest: "Consider reducing weekly budget to $156k (cap-optimal)"
- Or: "Consider raising supply cap to $240M to fully deploy budget"

### 2.2 Whale Exit Destination Modeling

**Priority: MEDIUM — improves exit probability accuracy**

GET INDIVIDUAL WHALES HISTORICAL DATA FOR SENSITIVITY TO APY SHIFTS OR MOVEMENT RATES TO GUAGE THIS

SAME SAME WITH THE VAULTS OVERALL SENSITIVITY OR MOVEMENT RATE!!

**Current:** Whales have `alt_rate` which is a synthetic threshold.

**Enhanced:** Use actual competitor data to model where whales would go:
```python
@dataclass
class WhaleExitRoute:
    destination_venue: str
    destination_apy: float
    available_capacity: float   # Can the destination absorb this whale?
    switching_cost_usd: float   # Gas + swap + bridge
    historical_exits: int       # How many times has this whale gone there before?
    exit_probability: float     # Derived from above

    @property
    def effective_threshold(self) -> float:
        """APR at which this exit becomes rational for the whale."""
        annualized_cost = self.switching_cost_usd * 52.14 / whale_position
        return self.destination_apy - annualized_cost
```

### 2.3 Mercenary Capital Realism

**Priority: MEDIUM**

**Current:** `MercenaryConfig` has flat entry/exit thresholds.

**Enhanced:**
- Entry threshold should be relative to **competitor rates**, not absolute
- Mercs enter when `our_rate > best_competitor_rate * 1.3` (30% premium attracts hot money)
- Mercs exit when `our_rate < best_competitor_rate * 1.1` (10% premium isn't worth the gas)
- Merc TVL should be capped by available liquidity (they can't deposit more than DEX liquidity allows in a single block without slippage)

### 2.4 Utilization Dynamics

**Priority: MEDIUM — affects base APY feedback loop**

**Current:** Utilization is static or loosely modeled.

**Reality:** As TVL grows but borrows stay flat, utilization drops, base APY drops,
total APR drops. This creates a negative feedback loop that the simulator must capture.

**Fix:**
```python
def dynamic_base_apy(
    current_tvl: float,
    current_borrows: float,
    interest_rate_model: dict,  # Aave/Euler IRM parameters
) -> float:
    """
    As TVL grows, utilization = borrows/tvl drops.
    Lower utilization → lower borrow rate → lower supply APY.
    This means incentive rate must compensate for falling base APY.
    """
    util = current_borrows / current_tvl if current_tvl > 0 else 0
    # Aave V3 IRM: linear below optimal, steep above
    if util <= interest_rate_model['optimal_util']:
        borrow_rate = interest_rate_model['base_rate'] + \
            util / interest_rate_model['optimal_util'] * interest_rate_model['slope1']
    else:
        excess = (util - interest_rate_model['optimal_util']) / \
            (1 - interest_rate_model['optimal_util'])
        borrow_rate = interest_rate_model['base_rate'] + \
            interest_rate_model['slope1'] + excess * interest_rate_model['slope2']

    supply_apy = borrow_rate * util * (1 - interest_rate_model['reserve_factor'])
    return supply_apy
```

---

## 3. Optimal Competitor Data Grabbing Strategy

### 3.1 Data Sources Priority

| Source | What | Refresh Rate | Cost |
|--------|------|-------------|------|
| **RPC (Alchemy/Infura)** | TVL, APY, caps, whale positions, IRM params | Real-time | Free tier sufficient |
| **DeFiLlama API** | Competitor APYs, historical yields, TVL | `/yields` endpoint, 15min cache | Free |
| **Dune Analytics** | Whale behavior, TVL flow, deposit/withdraw events | Hourly/daily queries | API key needed |
| **Morpho API/Subgraph** | Market params, vault allocations, reallocator | Real-time | Free |
| **1inch API** | Swap routing/cost for switching cost estimation | Per-query | Free tier |

### 3.2 Sync Schedule

```
┌─────────────────────┬────────────┬──────────────────────────────────────┐
│ Data                │ Frequency  │ Trigger                              │
├─────────────────────┼────────────┼──────────────────────────────────────┤
│ TVL + APY + caps    │ On demand  │ Venue selection in dashboard         │
│ Whale positions     │ On demand  │ Venue selection (cached 1hr)         │
│ Whale history       │ Daily      │ Cron or manual "Sync" button         │
│ Competitor landscape│ Hourly     │ Background job or on r_threshold     │
│ TVL stickiness      │ Weekly     │ Offline analysis, cached             │
│ Swap costs          │ Daily      │ Background job, cached               │
│ IRM parameters      │ On deploy  │ Rarely changes, manual refresh       │
└─────────────────────┴────────────┴──────────────────────────────────────┘
```

### 3.3 Morpho-Specific Fetching

**Morpho Blue markets + MetaMorpho vaults need special handling:**

```python
def fetch_morpho_competitors(asset: str) -> list[CompetitorVenue]:
    """
    Morpho has two layers:
    1. Morpho Blue markets — direct lending, isolated risk
    2. MetaMorpho vaults — curated allocators that spread across markets

    For competitor analysis:
    - MetaMorpho vault APY = blended rate across their allocated markets
    - Supply cap = vault-level cap (not market-level)
    - Available capacity = vault cap - vault TVL
    - Reallocator can shift capital between markets → APY can change fast

    Key vaults to track:
    - Steakhouse USDC, Gauntlet USDC, MEV Capital USDC
    - Any vault that holds the same asset we're incentivizing
    """
```

### 3.4 Cross-Protocol Cap Intelligence

**Build a unified view of where capital CAN go:**

```python
@dataclass
class CapacityMap:
    """
    For a given stablecoin, map all venues and their remaining capacity.
    This tells us: if a whale exits our venue, WHERE can they actually go?
    """
    asset: str
    venues: list[VenueCapacity]

    def absorbable_by(self, position_usd: float) -> list[VenueCapacity]:
        """Which venues can absorb this position without hitting caps?"""
        return [v for v in self.venues
                if v.available_capacity_usd >= position_usd * 0.5]

    def best_alternative_rate(self, position_usd: float) -> float:
        """Best rate among venues that can actually take this capital."""
        viable = self.absorbable_by(position_usd)
        if not viable:
            return 0.0
        return max(v.effective_apy for v in viable)
```

---

## 4. Simulation Tuning Checklist

### 4.1 Things the Simulator MUST Get Right

- [ ] **Cap as hard wall** — TVL never exceeds supply cap, ever
- [ ] **Budget savings when cap binds** — track and report unused budget
- [ ] **r_max taper near cap** — don't overpay when TVL is 95% of cap
- [ ] **Float rate at cap** — always show what the natural float rate is at cap
- [ ] **Base APY feedback** — as TVL grows, utilization drops, base APY drops
- [ ] **Competitor-relative whale exits** — whales exit to real venues with real capacity
- [ ] **Swap cost in r_threshold** — discount competitor rates by switching costs
- [ ] **Capacity-filtered competitors** — ignore capped-out venues for whale exit modeling
- [ ] **Incentive elasticity from Dune** — calibrate retail alpha from empirical TVL flows
- [ ] **Stickiness floor** — organic TVL that doesn't respond to rate changes

### 4.2 Things the Simulator Should Report

- [ ] **Budget efficiency** — $ spent per $1M TVL gained
- [ ] **Cap-binding frequency** — what % of MC paths hit the cap?
- [ ] **Natural equilibrium rate** — what rate would hold TVL at target without cap?
- [ ] **Whale risk concentration** — what % of TVL is in whale-exit-risk positions?
- [ ] **Organic vs incentivized split** — estimated TVL that stays post-incentive
- [ ] **Competitor gap** — how far above/below each competitor's rate are we?
- [ ] **Rate sensitivity score** — how much TVL would we lose per 1% APR drop?

### 4.3 Extreme Cases That Must Be Handled

| Scenario | Expected Behavior |
|----------|-------------------|
| Target TVL = supply cap | Optimizer should use float with r_max just above float-at-cap |
| Target TVL > supply cap | Warn user, clamp target to cap, optimize for cap-fill speed |
| Budget can fund >20% APR | Flag mercenary risk, suggest lower budget or tighter r_max |
| All competitors are capped out | r_threshold should drop (no viable exits for whales) |
| Base APY > target total APR | No incentives needed — report this and skip optimization |
| Single whale = 40% of TVL | Flag concentration risk, show exit-impact scenarios |
| Zero whale data available | Fall back to synthetic profiles with conservative assumptions |
| Venue has no supply cap | Use protocol-default safety cap (e.g., 10× current TVL) |

---

## 5. Implementation Order

### Phase 1: Core Data (Week 1)
1. **Competitor landscape scan** — enrich `data.py` with Morpho, Compound, Spark fetchers
2. **Capacity-aware r_threshold** — filter competitors by available capacity
3. **Swap cost adjustment** — add per-venue switching cost to r_threshold
4. **Cap-proximity r_max taper** — in optimizer, detect near-cap and adjust grid

### Phase 2: Whale Intelligence (Week 2)
5. **Dune whale history sync** — deposit/withdrawal events, hold durations
6. **Stickiness scoring** — derive per-whale stickiness from empirical data
7. **Exit destination modeling** — map whales to capacity-filtered competitor venues
8. **Whale-aware r_threshold** — weight competitor rates by whale-absorptive capacity

### Phase 3: Simulation Realism (Week 3)
9. **Dynamic base APY** — interest rate model feedback as TVL changes utilization
10. **Hard cap enforcement** — TVL clamped, budget savings tracked
11. **Organic TVL floor** — sticky fraction from Dune TVL decomposition
12. **Empirical retail calibration** — alpha/sigma from historical TVL-APR correlation

### Phase 4: Reporting (Week 4)
13. **Budget efficiency metrics** — $/TVL, cap-binding %, natural equilibrium rate
14. **Sensitivity dashboard** — rate sensitivity score, whale concentration risk
15. **Competitor gap visualization** — where we sit vs every viable alternative
16. **Post-campaign projection** — estimated TVL retention after incentives end

---

## 6. Agent Prompt for Implementation

Use the following prompt when starting implementation in the new repo:

---

> Read every file in this repo thoroughly. This is a single-venue DeFi incentive
> campaign optimizer. The core simulation in `sim/campaign/engine.py` runs Monte Carlo
> paths modeling TVL dynamics under different campaign configurations (budget B, max rate
> r_max). The optimizer in `sim/campaign/optimizer.py` searches a grid to find the
> cheapest configuration meeting the user's constraints.
>
> **Your task is to implement the enhancements in `TODO_DATA_SYNC_AND_SIMULATION.md`
> in the specified phase order.** For each item:
>
> 1. Read the relevant existing code first
> 2. Implement the change with full type annotations and docstrings
> 3. Add tests that verify the behavior with synthetic data (no RPC needed)
> 4. Update the dashboard (`app5.py`) to surface new data/metrics
> 5. Run `uv run pytest -v` and ensure all tests pass
>
> **Key constraints:**
> - The simulation must respect supply caps as hard walls (TVL never exceeds cap)
> - r_threshold must discount competitor rates by switching costs
> - Competitors at capacity (supply cap reached) must be excluded from whale exit modeling
> - The optimizer should never suggest r_max significantly above the float rate at cap
>   when target TVL is near the cap — this wastes budget
> - Budget savings from cap-binding must be tracked and reported
> - Base APY should dynamically adjust as TVL changes utilization (IRM feedback)
> - Whale exit thresholds should use empirical Dune data when available, synthetic fallback otherwise
> - All new metrics must have tooltips explaining what they are and how to interpret them
>
> **Testing requirements:**
> - Every new function must have ≥3 tests covering normal, edge, and extreme cases
> - Tests must use synthetic/mock data only (no network calls)
> - Test file naming: `test_{module}.py` in `sim/tests/`
> - Run `uv run ruff check sim/ --fix` after each change
>
> Start with Phase 1 Item 1: Competitor landscape scan. Read `sim/campaign/data.py`
> and `sim/campaign/evm_data.py` to understand the current fetching, then implement
> the enriched `CompetitorVenue` dataclass and multi-protocol fetcher.

---
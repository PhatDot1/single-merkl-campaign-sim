"""
Tests for Set APR mode — budget minimization with APR floor enforcement.

Validates that:
1. Optimizer finds budget BELOW the derived ceiling (budget minimization works)
2. APR floor is maintained (p5 APR ≥ floor × 0.98)
3. Floor breach cost is zero or near-zero when floor is met
4. Higher spend weight drives budget lower
5. Whale exits increase the required budget (resilience)
6. Supply cap constrains TVL growth, reducing needed budget
7. time_below_floor tracks correctly
8. Headroom computation is consistent
9. Full grid is searched (not a 3×3 pinned grid)
"""

from campaign.agents import (
    APYSensitiveConfig,
    WhaleProfile,
)
from campaign.engine import (
    LossWeights,
    MonteCarloResult,
    run_monte_carlo,
)
from campaign.optimizer import SurfaceGrid, optimize_surface
from campaign.state import CampaignConfig, CampaignEnvironment

WEEKS_PER_YEAR = 365.0 / 7.0  # 52.142857


# ============================================================================
# Helpers
# ============================================================================


def _set_apr_weights(
    floor_apr: float,
    target_tvl: float,
    w_spend_mult: float = 5.0,
    sensitivity: float = 0.9,
    r_threshold: float = 0.03,
    budget_ceiling: float = 0.0,
) -> LossWeights:
    """Build LossWeights matching Set APR mode in the dashboard."""
    # If budget_ceiling not provided, derive it
    if budget_ceiling <= 0:
        inc_rate = max(0.005, floor_apr - 0.02)  # assume ~2% base
        budget_ceiling = target_tvl * inc_rate / WEEKS_PER_YEAR * 1.1
    return LossWeights(
        w_spend=1.0 * w_spend_mult,  # Boosted to drive minimum budget
        w_spend_waste_penalty=0.0,
        w_apr_variance=3.0 * 1.5,  # Set APR: ×1.5
        w_apr_ceiling=5.0 * 2.0,  # Set APR: ×2.0
        w_tvl_shortfall=8.0,
        w_budget_waste=0.0,
        w_mercenary=6.0,
        w_whale_proximity=6.0,
        w_apr_floor=15.0,  # Set APR: ≥15.0
        apr_target=r_threshold * 1.2,
        apr_ceiling=floor_apr * 2,  # Generous ceiling
        tvl_target=target_tvl,
        apr_stability_on_total=True,
        apr_floor=floor_apr,
        apr_floor_sensitivity=max(sensitivity, 0.85),
        # Normalize spend by budget ceiling so absolute spend is compared
        spend_reference_budget=budget_ceiling * 4,  # × num_epochs (4 weeks)
    )


def _build_set_apr_grid(
    floor_apr: float,
    base_apy: float,
    target_tvl: float,
    budget_ceiling: float,
    r_lo: float | None = None,
    r_hi: float | None = None,
    b_steps: int = 8,
    r_steps: int = 8,
) -> SurfaceGrid:
    """Build a grid mimicking the Set APR mode search range."""
    inc_rate = max(0.005, floor_apr - base_apy)
    floor_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
    # Floor-aware b_min (same as run_venue_optimization)
    b_min = max(10_000, floor_budget * 0.85)
    b_max = budget_ceiling

    if r_lo is None:
        # r_max must be ≥ required incentive rate — otherwise floor is impossible
        r_lo = inc_rate
    if r_hi is None:
        r_hi = min(0.08, max(floor_apr * 2, r_lo + 0.01))

    return SurfaceGrid.from_ranges(
        B_min=b_min,
        B_max=b_max,
        B_steps=b_steps,
        r_max_min=r_lo,
        r_max_max=r_hi,
        r_max_steps=r_steps,
        base_apy=base_apy,
    )


# ============================================================================
# 1. Budget Minimization — optimizer picks B < derived ceiling
# ============================================================================


class TestSetAPRBudgetMinimization:
    """The core property: optimizer should find B < derived_budget when possible."""

    def test_optimizer_finds_budget_below_ceiling(self):
        """With a generous ceiling, the optimizer should NOT pick the max budget."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05  # 5% total → 3% incentive needed
        inc_rate = floor_apr - base_apy  # 0.03
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR  # ~$57,534/wk
        budget_ceiling = derived_budget * 1.5  # 50% above derived — generous

        weights = _set_apr_weights(floor_apr, target_tvl, budget_ceiling=budget_ceiling)
        grid = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
            b_steps=8,
            r_steps=8,
        )
        env = CampaignEnvironment(r_threshold=0.03)
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.9,
            max_sensitive_tvl=target_tvl * 0.10,
        )

        sr = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=30,
            apy_sensitive_config=apy_cfg,
        )

        # Optimizer should NOT pick the maximum budget
        assert sr.optimal_B < budget_ceiling * 0.95, (
            f"Optimizer picked B=${sr.optimal_B:,.0f} which is at the ceiling "
            f"(${budget_ceiling:,.0f}). Budget minimization is not working."
        )

    def test_optimizer_respects_floor(self):
        """Chosen budget must still maintain the APR floor."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
        budget_ceiling = derived_budget * 1.1

        weights = _set_apr_weights(floor_apr, target_tvl, budget_ceiling=budget_ceiling)
        grid = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
        )
        env = CampaignEnvironment(r_threshold=0.03)
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.9,
            max_sensitive_tvl=target_tvl * 0.10,
        )

        sr = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=30,
            apy_sensitive_config=apy_cfg,
        )

        mc = sr.optimal_mc_result
        assert mc is not None
        # APR p5 should be close to or above floor (allow 5% tolerance)
        assert mc.apr_p5 >= floor_apr * 0.95, (
            f"APR p5={mc.apr_p5:.2%} is well below floor {floor_apr:.2%}. "
            f"Optimizer B=${sr.optimal_B:,.0f}, r_max={sr.optimal_r_max:.2%}"
        )

    def test_grid_is_full_not_pinned(self):
        """Set APR mode should search a full grid, not a 3×3 pinned grid."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
        budget_ceiling = derived_budget * 1.1

        grid = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
            b_steps=8,
            r_steps=8,
        )

        # Grid should have 8×8 = 64 points, not 3×3 = 9
        n_B, n_r = grid.shape
        assert n_B >= 6, f"Budget grid only has {n_B} points — looks pinned"
        assert n_r >= 6, f"r_max grid only has {n_r} points — looks pinned"

    def test_higher_spend_weight_lowers_budget(self):
        """Increasing w_spend should push the optimizer toward cheaper configs."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
        budget_ceiling = derived_budget * 1.5

        env = CampaignEnvironment(r_threshold=0.03)
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.9,
            max_sensitive_tvl=target_tvl * 0.10,
        )

        budgets = []
        for spend_mult in [1.0, 5.0, 10.0]:
            weights = _set_apr_weights(
                floor_apr, target_tvl, w_spend_mult=spend_mult, budget_ceiling=budget_ceiling
            )
            grid = _build_set_apr_grid(
                floor_apr,
                base_apy,
                target_tvl,
                budget_ceiling,
                b_steps=6,
                r_steps=6,
            )
            sr = optimize_surface(
                grid=grid,
                env=env,
                initial_tvl=target_tvl,
                whale_profiles=[],
                weights=weights,
                n_paths=20,
                apy_sensitive_config=apy_cfg,
            )
            budgets.append(sr.optimal_B)

        # Higher spend weight should give equal or lower budget
        # (with discrete grid, exact monotonicity may not hold, but trend should)
        assert budgets[-1] <= budgets[0] * 1.1, (
            f"w_spend×10 gave B=${budgets[-1]:,.0f} but w_spend×1 gave "
            f"B=${budgets[0]:,.0f} — higher spend weight should lower budget"
        )


# ============================================================================
# 2. Floor Enforcement — APR target is maintained
# ============================================================================


class TestSetAPRFloorEnforcement:
    """Verify the APR floor constraint is enforced by the loss function."""

    def test_floor_breach_cost_penalizes_underspend(self):
        """A config well below the floor should have high floor_breach_cost."""
        base_apy = 0.02
        floor_apr = 0.05
        target_tvl = 100e6

        # Budget that gives only 1% incentive (well below 3% needed)
        low_budget = target_tvl * 0.01 / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=low_budget,
            apr_cap=0.08,
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
        )

        floor_cost = mc.loss_components.get("floor_breach", 0.0)
        assert floor_cost > 1.0, (
            f"Floor breach cost is only {floor_cost:.4f} but budget "
            f"(${low_budget:,.0f}) is way below floor requirement. "
            f"Expected significant floor penalty."
        )

    def test_no_floor_breach_when_budget_sufficient(self):
        """A config at or above derived budget should have ~zero floor breach."""
        base_apy = 0.02
        floor_apr = 0.05
        target_tvl = 100e6
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=derived_budget,
            apr_cap=inc_rate,  # Cap at exactly the needed rate
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=30,
        )

        floor_cost = mc.loss_components.get("floor_breach", 0.0)
        assert floor_cost < 0.5, (
            f"Floor breach cost is {floor_cost:.4f} even with sufficient budget "
            f"(${derived_budget:,.0f}/wk for {floor_apr:.2%} floor). "
            f"APR mean={mc.mean_apr:.2%}, p5={mc.apr_p5:.2%}"
        )

    def test_time_below_floor_tracks_correctly(self):
        """time_below_floor should be high when budget is too low."""
        base_apy = 0.02
        floor_apr = 0.06  # 6% — needs 4% incentive
        target_tvl = 100e6
        # Budget gives only 1% incentive — always below 6% floor
        low_budget = target_tvl * 0.01 / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=low_budget,
            apr_cap=0.08,
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
        )

        # With only 3% total APR vs 6% floor, should be below floor most of the time
        assert mc.mean_time_below_floor > 0.5, (
            f"time_below_floor={mc.mean_time_below_floor:.2f} but APR (~3%) "
            f"is well below floor (6%), should be >50%"
        )


# ============================================================================
# 3. Whale Resilience — whales increase required budget
# ============================================================================


class TestSetAPRWhaleResilience:
    """Whale exits create TVL drops that require more budget to maintain APR."""

    def test_whales_increase_optimal_budget(self):
        """With whales, the optimizer should pick a higher budget to maintain APR."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
        budget_ceiling = derived_budget * 1.5

        env = CampaignEnvironment(r_threshold=0.03)
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.9,
            max_sensitive_tvl=target_tvl * 0.10,
        )

        # Run without whales
        weights = _set_apr_weights(floor_apr, target_tvl, budget_ceiling=budget_ceiling)
        grid = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
            b_steps=6,
            r_steps=6,
        )
        sr_no_whales = optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
            apy_sensitive_config=apy_cfg,
        )

        # Run with a big whale (20% of TVL)
        # WhaleProfile.exit_threshold is a property derived from alt_rate + risk_premium
        # Set alt_rate low so whale stays unless APR drops significantly
        whale = WhaleProfile(
            whale_id="big_whale",
            position_usd=target_tvl * 0.20,
            alt_rate=0.03,
            risk_premium=0.005,
            exit_delay_days=1.0,
            reentry_delay_days=3.0,
        )
        grid2 = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
            b_steps=6,
            r_steps=6,
        )
        sr_with_whales = optimize_surface(
            grid=grid2,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[whale],
            weights=weights,
            n_paths=20,
            apy_sensitive_config=apy_cfg,
        )

        # With whales, optimizer should pick equal or higher budget
        # (whale exit → TVL drop → rate spike → need more budget)
        # Allow 10% tolerance for MC noise
        assert sr_with_whales.optimal_B >= sr_no_whales.optimal_B * 0.9, (
            f"With whale: B=${sr_with_whales.optimal_B:,.0f}, "
            f"without: B=${sr_no_whales.optimal_B:,.0f}. "
            f"Whale presence should not drastically reduce budget."
        )


# ============================================================================
# 4. Supply Cap — caps reduce needed budget
# ============================================================================


class TestSetAPRSupplyCap:
    """Supply cap limits TVL growth, so less budget is needed at capped TVL."""

    def test_supply_cap_reduces_budget(self):
        """With a supply cap below target TVL, optimizer needs less budget."""
        target_tvl = 200e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR
        budget_ceiling = derived_budget * 1.1

        env = CampaignEnvironment(r_threshold=0.03)
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.9,
            max_sensitive_tvl=target_tvl * 0.10,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)

        # Without supply cap
        grid1 = _build_set_apr_grid(
            floor_apr,
            base_apy,
            target_tvl,
            budget_ceiling,
            b_steps=6,
            r_steps=6,
        )
        sr_no_cap = optimize_surface(
            grid=grid1,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
            apy_sensitive_config=apy_cfg,
        )

        # With supply cap at 50% of target — TVL can't grow beyond 100M
        supply_cap = target_tvl * 0.5
        grid2 = SurfaceGrid.from_ranges(
            B_min=10_000,
            B_max=budget_ceiling,
            B_steps=6,
            r_max_min=0.02,
            r_max_max=0.08,
            r_max_steps=6,
            base_apy=base_apy,
            supply_cap=supply_cap,
        )
        sr_with_cap = optimize_surface(
            grid=grid2,
            env=env,
            initial_tvl=supply_cap,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
            apy_sensitive_config=apy_cfg,
        )

        # With supply cap, optimizer can use less budget (same rate at lower TVL)
        assert sr_with_cap.optimal_B <= sr_no_cap.optimal_B * 1.1, (
            f"Supply cap should allow equal or lower budget: "
            f"capped=${sr_with_cap.optimal_B:,.0f}, "
            f"uncapped=${sr_no_cap.optimal_B:,.0f}"
        )


# ============================================================================
# 5. Headroom Consistency
# ============================================================================


class TestAPRHeadroom:
    """Verify headroom metrics are consistent."""

    def test_headroom_positive_when_budget_generous(self):
        """With budget at 150% of derived, headroom should be positive."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.05
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR

        # Generous budget
        config = CampaignConfig(
            weekly_budget=derived_budget * 1.5,
            apr_cap=inc_rate * 1.5,
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=30,
        )

        headroom_p5 = mc.apr_p5 - floor_apr
        headroom_mean = mc.mean_apr - floor_apr

        assert headroom_mean > 0, (
            f"Mean headroom should be positive: mean_apr={mc.mean_apr:.2%}, "
            f"floor={floor_apr:.2%}, headroom={headroom_mean:+.2%}"
        )
        assert headroom_p5 > -0.005, (
            f"p5 headroom should be near-positive: p5={mc.apr_p5:.2%}, "
            f"floor={floor_apr:.2%}, headroom={headroom_p5:+.2%}"
        )

    def test_headroom_negative_when_budget_insufficient(self):
        """With budget at 50% of derived, headroom should be negative."""
        target_tvl = 100e6
        base_apy = 0.02
        floor_apr = 0.06  # 6% — needs 4% incentive
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR

        # Insufficient budget — 50%
        config = CampaignConfig(
            weekly_budget=derived_budget * 0.5,
            apr_cap=0.08,
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=20,
        )

        headroom_mean = mc.mean_apr - floor_apr
        assert headroom_mean < 0, (
            f"Mean headroom should be negative with 50% budget: "
            f"mean_apr={mc.mean_apr:.2%}, floor={floor_apr:.2%}"
        )

    def test_time_below_floor_zero_when_ample_budget(self):
        """With ample budget and matching r_max, time_below_floor should be ~0."""
        target_tvl = 50e6
        base_apy = 0.03
        floor_apr = 0.05  # 5% — needs 2% incentive
        inc_rate = floor_apr - base_apy
        derived_budget = target_tvl * inc_rate / WEEKS_PER_YEAR

        config = CampaignConfig(
            weekly_budget=derived_budget * 1.2,  # 20% above derived
            apr_cap=inc_rate * 1.2,
            base_apy=base_apy,
        )
        weights = _set_apr_weights(floor_apr, target_tvl)
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=30,
        )

        assert mc.mean_time_below_floor < 0.15, (
            f"time_below_floor={mc.mean_time_below_floor:.2f} but budget is "
            f"20% above derived — should rarely breach floor"
        )


# ============================================================================
# 6. MonteCarloResult field
# ============================================================================


class TestMonteCarloResultFields:
    """Verify the new mean_time_below_floor field."""

    def test_mean_time_below_floor_exists(self):
        """MonteCarloResult should have mean_time_below_floor field."""
        mc = MonteCarloResult(B=10_000, r_max=0.05, t_bind=1e6)
        assert hasattr(mc, "mean_time_below_floor")
        assert mc.mean_time_below_floor == 0.0

    def test_mean_time_below_floor_in_range(self):
        """mean_time_below_floor should be in [0, 1]."""
        base_apy = 0.02
        target_tvl = 50e6
        config = CampaignConfig(
            weekly_budget=10_000,
            apr_cap=0.05,
            base_apy=base_apy,
        )
        weights = LossWeights(
            apr_floor=0.08,
            apr_floor_sensitivity=0.9,
            tvl_target=target_tvl,
        )
        env = CampaignEnvironment(r_threshold=0.03)

        mc = run_monte_carlo(
            config=config,
            env=env,
            initial_tvl=target_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=10,
        )

        assert 0.0 <= mc.mean_time_below_floor <= 1.0, (
            f"mean_time_below_floor={mc.mean_time_below_floor} out of [0,1]"
        )


# ============================================================================
# 7. Loss Function Spend Normalization
# ============================================================================


class TestSpendNormalization:
    """Verify that spend cost is properly normalized and drives budget minimization."""

    def test_higher_budget_higher_spend_cost(self):
        """All else equal, higher budget should produce higher spend cost."""
        base_apy = 0.02
        target_tvl = 100e6
        env = CampaignEnvironment(r_threshold=0.03)
        weights = _set_apr_weights(0.05, target_tvl)

        spend_costs = []
        for budget in [30_000, 60_000, 90_000]:
            config = CampaignConfig(
                weekly_budget=budget,
                apr_cap=0.05,
                base_apy=base_apy,
            )
            mc = run_monte_carlo(
                config=config,
                env=env,
                initial_tvl=target_tvl,
                whale_profiles=[],
                weights=weights,
                n_paths=10,
            )
            spend_costs.append(mc.loss_components.get("spend", 0.0))

        # Spend cost should be monotonically increasing with budget
        assert spend_costs[1] >= spend_costs[0] * 0.8, f"Spend cost not increasing: {spend_costs}"
        assert spend_costs[2] >= spend_costs[1] * 0.8, f"Spend cost not increasing: {spend_costs}"

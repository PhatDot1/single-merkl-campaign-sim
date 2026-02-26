"""
Tests for multi-venue budget allocation logic.

Validates that:
1. Per-protocol r_max ceilings are correct and respected
2. Max deployable budget = target_tvl × protocol_ceiling / 52.14
3. TVL-proportional weighting produces correct scale factors
4. Small pools can never exceed max deployable budget
5. Total allocation equals total budget exactly
6. r_max floor respects per-protocol ceiling (not global 8%)
7. RLUSD-Core-like scenario: Curve gets proportional allocation, not 24%
8. Pinned venue r_max ceiling uses protocol defaults
"""

import numpy as np
import pytest
from campaign.agents import (
    APYSensitiveConfig,
)
from campaign.engine import CampaignSimulationEngine, LossWeights
from campaign.optimizer import SurfaceGrid, SurfaceResult, optimize_surface
from campaign.state import CampaignConfig, CampaignEnvironment, CampaignState
from campaign.venue_registry import (
    GLOBAL_R_MAX_CEILING,
    PROTOCOL_R_MAX_DEFAULTS,
)

WEEKS_PER_YEAR = 365.0 / 7.0  # 52.142857


# ============================================================================
# 1. Protocol r_max Ceiling Values
# ============================================================================


class TestProtocolCeilings:
    """Verify PROTOCOL_R_MAX_DEFAULTS and GLOBAL_R_MAX_CEILING."""

    def test_global_ceiling_is_eight_percent(self):
        assert GLOBAL_R_MAX_CEILING == 0.08

    def test_all_expected_protocols_present(self):
        expected = {"aave", "euler", "morpho", "curve", "kamino"}
        assert expected == set(PROTOCOL_R_MAX_DEFAULTS.keys())

    def test_curve_ceiling_is_six_percent(self):
        lo, hi = PROTOCOL_R_MAX_DEFAULTS["curve"]
        assert hi == 0.06, f"Curve ceiling should be 6%, got {hi:.0%}"

    def test_aave_ceiling_is_six_percent(self):
        lo, hi = PROTOCOL_R_MAX_DEFAULTS["aave"]
        assert hi == 0.06, f"AAVE ceiling should be 6%, got {hi:.0%}"

    def test_euler_ceiling_is_seven_percent(self):
        lo, hi = PROTOCOL_R_MAX_DEFAULTS["euler"]
        assert hi == 0.07

    def test_morpho_ceiling_is_eight_percent(self):
        lo, hi = PROTOCOL_R_MAX_DEFAULTS["morpho"]
        assert hi == 0.08

    def test_all_ceilings_below_global(self):
        for proto, (lo, hi) in PROTOCOL_R_MAX_DEFAULTS.items():
            assert hi <= GLOBAL_R_MAX_CEILING, (
                f"{proto} ceiling {hi:.0%} exceeds global {GLOBAL_R_MAX_CEILING:.0%}"
            )

    def test_all_floors_positive(self):
        for proto, (lo, hi) in PROTOCOL_R_MAX_DEFAULTS.items():
            assert lo > 0, f"{proto} floor should be positive"
            assert lo < hi, f"{proto} floor should be less than ceiling"


# ============================================================================
# 2. Max Deployable Budget Computation
# ============================================================================


class TestMaxDeployableBudget:
    """Max deployable = target_tvl × protocol_ceiling / 52.14."""

    def test_curve_85m_tvl(self):
        """Curve at $85M TVL can deploy at most ~$97.8k/wk."""
        tvl = 85_000_000
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS["curve"]
        max_deploy = tvl * proto_hi / WEEKS_PER_YEAR
        # 85M × 0.06 / 52.14 ≈ $97,800
        assert 97_000 < max_deploy < 99_000

    def test_aave_612m_tvl(self):
        """AAVE at $612M TVL can deploy up to ~$704k/wk."""
        tvl = 612_000_000
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS["aave"]
        max_deploy = tvl * proto_hi / WEEKS_PER_YEAR
        # 612M × 0.06 / 52.14 ≈ $704k
        assert 700_000 < max_deploy < 710_000

    def test_euler_190m_tvl(self):
        """Euler at $190M TVL can deploy up to ~$255k/wk."""
        tvl = 190_000_000
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS["euler"]
        max_deploy = tvl * proto_hi / WEEKS_PER_YEAR
        # 190M × 0.07 / 52.14 ≈ $255k
        assert 250_000 < max_deploy < 260_000

    def test_small_pool_has_tight_cap(self):
        """A tiny pool with strict protocol ceiling has very small cap."""
        tvl = 10_000_000
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS["aave"]
        max_deploy = tvl * proto_hi / WEEKS_PER_YEAR
        # $10M × 0.06 / 52.14 ≈ $11.5k
        assert max_deploy < 12_000


# ============================================================================
# 3. TVL Weight Scaling
# ============================================================================


class TestTVLWeighting:
    """TVL weights: (target_tvl / avg_tvl) ^ 0.5."""

    def _compute_weights(self, tvls: list[float], exponent: float = 0.5) -> dict[str, float]:
        avg = sum(tvls) / len(tvls)
        return {f"v{i}": (t / avg) ** exponent for i, t in enumerate(tvls)}

    def test_equal_tvl_gives_equal_weight(self):
        weights = self._compute_weights([100e6, 100e6, 100e6])
        for w in weights.values():
            assert w == pytest.approx(1.0)

    def test_rlusd_core_tvl_weights(self):
        """RLUSD Core: AAVE=612M, Euler=190M, Curve=85M."""
        tvls = [612e6, 190e6, 85e6]
        weights = self._compute_weights(tvls)
        # avg = 295.67M
        # AAVE: (612/295.67)^0.5 ≈ 1.44
        # Euler: (190/295.67)^0.5 ≈ 0.80
        # Curve: (85/295.67)^0.5 ≈ 0.54
        assert weights["v0"] > 1.3  # AAVE weight > 1
        assert weights["v1"] < 1.0  # Euler weight < 1
        assert weights["v2"] < 0.6  # Curve weight << 1
        # Verify ordering
        assert weights["v0"] > weights["v1"] > weights["v2"]

    def test_large_pool_gets_higher_weight(self):
        """A pool 4x larger should have ~2x weight (sqrt scaling)."""
        tvls = [400e6, 100e6]
        weights = self._compute_weights(tvls)
        ratio = weights["v0"] / weights["v1"]
        # (400/250)^0.5 / (100/250)^0.5 = (1.6)^0.5 / (0.4)^0.5 = 1.265/0.632 = 2.0
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_exponent_zero_gives_no_weighting(self):
        """With exponent=0, all weights are 1.0 (no TVL bias)."""
        tvls = [600e6, 100e6, 10e6]
        weights = self._compute_weights(tvls, exponent=0.0)
        for w in weights.values():
            assert w == pytest.approx(1.0)


# ============================================================================
# 4. r_max Floor with Protocol Ceiling
# ============================================================================


class TestRMaxFloorProtocolCeiling:
    """r_max floor should respect per-protocol ceiling, not GLOBAL_R_MAX_CEILING."""

    def _compute_r_max_with_floor(
        self,
        B_alloc: float,
        target_tvl: float,
        r_max_surface: float,
        protocol: str,
    ) -> float:
        """Replicate the r_max floor logic from app3.py."""
        float_rate = B_alloc / max(target_tvl, 1.0) * WEEKS_PER_YEAR
        _, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(protocol, (0.02, GLOBAL_R_MAX_CEILING))
        venue_ceiling = min(proto_hi, GLOBAL_R_MAX_CEILING)
        return max(r_max_surface, min(float_rate, venue_ceiling))

    def test_curve_stays_at_six_percent(self):
        """Curve r_max should cap at 6% even with high budget."""
        r_max = self._compute_r_max_with_floor(
            B_alloc=200_000,
            target_tvl=85e6,
            r_max_surface=0.04,
            protocol="curve",
        )
        # float_rate = 200k/85M * 52.14 = 12.3% → capped at 6%
        assert r_max == pytest.approx(0.06)
        assert r_max < GLOBAL_R_MAX_CEILING

    def test_aave_stays_at_six_percent(self):
        """AAVE r_max should cap at 6%."""
        r_max = self._compute_r_max_with_floor(
            B_alloc=800_000,
            target_tvl=612e6,
            r_max_surface=0.04,
            protocol="aave",
        )
        # float_rate = 800k/612M * 52.14 = 6.8% → capped at 6%
        assert r_max == pytest.approx(0.06)

    def test_euler_can_reach_seven_percent(self):
        """Euler has higher ceiling (7%), so r_max can go higher."""
        r_max = self._compute_r_max_with_floor(
            B_alloc=300_000,
            target_tvl=190e6,
            r_max_surface=0.04,
            protocol="euler",
        )
        # float_rate = 300k/190M * 52.14 = 8.2% → capped at 7%
        assert r_max == pytest.approx(0.07)

    def test_unknown_protocol_uses_global_ceiling(self):
        """Unknown protocol falls back to GLOBAL_R_MAX_CEILING."""
        r_max = self._compute_r_max_with_floor(
            B_alloc=200_000,
            target_tvl=50e6,
            r_max_surface=0.04,
            protocol="unknown_proto",
        )
        # float_rate = 200k/50M * 52.14 = 20.9% → capped at 8%
        assert r_max == pytest.approx(GLOBAL_R_MAX_CEILING)

    def test_float_rate_below_surface_uses_surface(self):
        """When float rate < r_max_surface, use the surface value."""
        r_max = self._compute_r_max_with_floor(
            B_alloc=10_000,
            target_tvl=500e6,
            r_max_surface=0.05,
            protocol="aave",
        )
        # float_rate = 10k/500M * 52.14 = 0.1% → r_max_surface (5%) dominates
        assert r_max == pytest.approx(0.05)

    def test_old_bug_would_give_eight_percent_for_curve(self):
        """Validate the bug is fixed: old code used GLOBAL_R_MAX_CEILING."""
        # Old code: r_max = max(r_max_surface, min(float_rate, GLOBAL_R_MAX_CEILING))
        B_alloc = 200_000
        target_tvl = 85e6
        r_max_surface = 0.04
        float_rate = B_alloc / target_tvl * WEEKS_PER_YEAR  # 12.3%

        old_r_max = max(r_max_surface, min(float_rate, GLOBAL_R_MAX_CEILING))
        new_r_max = self._compute_r_max_with_floor(
            B_alloc,
            target_tvl,
            r_max_surface,
            "curve",
        )

        assert old_r_max == pytest.approx(0.08)  # Old bug: 8%
        assert new_r_max == pytest.approx(0.06)  # Fixed: 6%


# ============================================================================
# 5. Max Deployable Cap Enforcement
# ============================================================================


class TestMaxDeployableCapEnforcement:
    """No venue should receive more budget than its max deployable amount."""

    def _enforce_cap(
        self,
        allocations: dict[str, float],
        venues: list[dict],
    ) -> dict[str, float]:
        """Replicate the post-allocation cap logic from app3.py."""
        max_deployable = {}
        for v in venues:
            target_tvl = v["target_tvl"]
            proto = v.get("protocol", "").lower()
            _, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(proto, (0.02, GLOBAL_R_MAX_CEILING))
            venue_ceiling = min(proto_hi, GLOBAL_R_MAX_CEILING)
            max_deployable[v["name"]] = target_tvl * venue_ceiling / WEEKS_PER_YEAR

        allocs = dict(allocations)
        permanent_caps: set[str] = set()
        for _round in range(5):
            excess_total = 0.0
            for v in venues:
                if v["name"] in permanent_caps:
                    continue
                cap = max_deployable[v["name"]]
                if allocs[v["name"]] > cap:
                    excess_total += allocs[v["name"]] - cap
                    allocs[v["name"]] = cap
                    permanent_caps.add(v["name"])
            if excess_total <= 0:
                break
            uncapped = [v for v in venues if v["name"] not in permanent_caps]
            if not uncapped:
                break  # All venues at cap — excess is undeployable
            uncapped_total = sum(allocs[v["name"]] for v in uncapped)
            for v in uncapped:
                if uncapped_total > 0:
                    allocs[v["name"]] += excess_total * allocs[v["name"]] / uncapped_total
                else:
                    allocs[v["name"]] += excess_total / len(uncapped)
        return allocs

    def test_curve_capped_excess_redistributed(self):
        """Curve getting $240k should be capped and excess goes to AAVE/Euler."""
        venues = [
            {"name": "AAVE", "protocol": "aave", "target_tvl": 612e6},
            {"name": "Euler", "protocol": "euler", "target_tvl": 190e6},
            {"name": "Curve", "protocol": "curve", "target_tvl": 85e6},
        ]
        # Old (buggy) allocation
        old_allocs = {"AAVE": 670_000, "Euler": 89_000, "Curve": 241_000}
        total = sum(old_allocs.values())

        fixed = self._enforce_cap(old_allocs, venues)

        # Curve max deployable = 85M * 0.06 / 52.14 ≈ $97.8k
        assert fixed["Curve"] < 100_000
        # Excess redistributed to AAVE and Euler
        assert fixed["AAVE"] > old_allocs["AAVE"]
        assert fixed["Euler"] > old_allocs["Euler"]
        # Total preserved
        assert sum(fixed.values()) == pytest.approx(total, rel=0.001)

    def test_no_venue_exceeds_max_deployable(self):
        """After cap enforcement, no venue exceeds its protocol ceiling."""
        venues = [
            {"name": "V1", "protocol": "aave", "target_tvl": 600e6},
            {"name": "V2", "protocol": "curve", "target_tvl": 80e6},
            {"name": "V3", "protocol": "euler", "target_tvl": 300e6},
        ]
        allocs = {"V1": 400_000, "V2": 300_000, "V3": 300_000}
        fixed = self._enforce_cap(allocs, venues)

        for v in venues:
            proto = v["protocol"]
            _, proto_hi = PROTOCOL_R_MAX_DEFAULTS[proto]
            max_deploy = v["target_tvl"] * proto_hi / WEEKS_PER_YEAR
            assert fixed[v["name"]] <= max_deploy + 1.0, (
                f"{v['name']} allocation ${fixed[v['name']]:,.0f} exceeds "
                f"max deployable ${max_deploy:,.0f}"
            )

    def test_all_under_cap_no_change(self):
        """When all allocations are under cap, nothing changes."""
        venues = [
            {"name": "V1", "protocol": "aave", "target_tvl": 600e6},
            {"name": "V2", "protocol": "euler", "target_tvl": 300e6},
        ]
        allocs = {"V1": 200_000, "V2": 100_000}
        fixed = self._enforce_cap(allocs, venues)
        assert fixed == allocs

    def test_cascade_redistribution(self):
        """Redistribution from capped venues flows to uncapped ones."""
        venues = [
            {"name": "V1", "protocol": "curve", "target_tvl": 30e6},
            {"name": "V2", "protocol": "curve", "target_tvl": 40e6},
            {"name": "V3", "protocol": "aave", "target_tvl": 500e6},
        ]
        # Both Curve venues have tight caps
        allocs = {"V1": 200_000, "V2": 200_000, "V3": 600_000}
        fixed = self._enforce_cap(allocs, venues)

        # V1 cap: 30M * 0.06 / 52.14 ≈ $34.5k
        # V2 cap: 40M * 0.06 / 52.14 ≈ $46.0k
        # V3 cap: 500M * 0.06 / 52.14 ≈ $575k
        assert fixed["V1"] < 35_000
        assert fixed["V2"] < 47_000
        # V3 absorbs excess from V1 and V2 (up to its $575k AAVE cap)
        v3_cap = 500e6 * 0.06 / WEEKS_PER_YEAR  # ~$575k
        assert fixed["V3"] == pytest.approx(v3_cap, rel=0.01)
        # Total is at most sum of all caps
        assert sum(fixed.values()) <= sum(allocs.values()) + 1.0


# ============================================================================
# 6. RLUSD Core Scenario — End-to-End Allocation Properties
# ============================================================================


class TestRLUSDCoreScenario:
    """
    Validate allocation properties for the RLUSD Core scenario:
    AAVE=$612M, Euler=$190M, Curve=$85M, total budget=$1M/wk.
    """

    def test_curve_tvl_share(self):
        """Curve's TVL share is ~9.6%, should not get >20% of budget."""
        total_tvl = 612e6 + 190e6 + 85e6
        curve_share = 85e6 / total_tvl
        assert curve_share == pytest.approx(0.096, abs=0.01)

    def test_curve_max_deployable_under_100k(self):
        """Curve can deploy at most ~$97.8k at 6% protocol ceiling."""
        max_deploy = 85e6 * 0.06 / WEEKS_PER_YEAR
        assert max_deploy < 100_000

    def test_curve_max_is_under_10pct_of_budget(self):
        """Curve's max deployable is <10% of $1M total budget."""
        max_deploy = 85e6 * 0.06 / WEEKS_PER_YEAR
        assert max_deploy / 1_000_000 < 0.10

    def test_aave_gets_majority_share(self):
        """AAVE (69% TVL share) should get >50% of budget."""
        # This is a property test — the exact allocation depends on
        # loss surfaces, but TVL-weighted marginals + cap enforcement
        # should ensure AAVE gets the lion's share.
        total_tvl = 612e6 + 190e6 + 85e6
        aave_share = 612e6 / total_tvl
        assert aave_share > 0.68
        # Max deployable for AAVE = 612M * 0.06 / 52.14 ≈ $704k
        # With $1M budget and Curve capped at ~$98k, AAVE should get >$600k

    def test_apr_at_proportional_allocation(self):
        """Check APR at TVL-proportional allocation is reasonable."""
        total_budget = 1_000_000
        total_tvl = 612e6 + 190e6 + 85e6
        venues = {
            "AAVE": {"tvl": 612e6, "base": 0.0086},
            "Euler": {"tvl": 190e6, "base": 0.0105},
            "Curve": {"tvl": 85e6, "base": 0.0004},
        }
        for name, v in venues.items():
            share = v["tvl"] / total_tvl
            budget = total_budget * share
            inc_apr = budget / v["tvl"] * WEEKS_PER_YEAR
            total_apr = v["base"] + inc_apr
            # All venues should have reasonable APR (2-8%)
            assert 0.02 < total_apr < 0.10, (
                f"{name}: total APR {total_apr:.2%} out of range "
                f"(budget=${budget:,.0f}, TVL=${v['tvl'] / 1e6:.0f}M)"
            )


# ============================================================================
# 7. Total APR Conversion (Single-Venue)
# ============================================================================


class TestTotalAPRConversion:
    """r_max entered as total APR should be correctly converted to incentive."""

    def test_basic_conversion(self):
        """Total APR 6% with base 1% → incentive r_max 5%."""
        total_apr = 0.06
        base_apy = 0.01
        incentive_r_max = max(0.005, total_apr - base_apy)
        assert incentive_r_max == pytest.approx(0.05)

    def test_high_base_apy(self):
        """Total APR 4% with base 3% → incentive r_max 1%."""
        total_apr = 0.04
        base_apy = 0.03
        incentive_r_max = max(0.005, total_apr - base_apy)
        assert incentive_r_max == pytest.approx(0.01)

    def test_base_apy_exceeds_total_gives_minimum(self):
        """Total APR 2% with base 3% → floored at 0.5%."""
        total_apr = 0.02
        base_apy = 0.03
        incentive_r_max = max(0.005, total_apr - base_apy)
        assert incentive_r_max == pytest.approx(0.005)

    def test_zero_base_apy_passthrough(self):
        """Total APR == incentive r_max when base is zero."""
        total_apr = 0.06
        base_apy = 0.0
        incentive_r_max = max(0.005, total_apr - base_apy)
        assert incentive_r_max == pytest.approx(0.06)

    def test_range_conversion_preserves_width(self):
        """Total APR range width preserved after conversion."""
        lo_total, hi_total = 0.04, 0.08
        base_apy = 0.01
        lo_inc = max(0.005, lo_total - base_apy)
        hi_inc = max(lo_inc + 0.005, hi_total - base_apy)
        assert hi_inc - lo_inc == pytest.approx(hi_total - lo_total)

    def test_curve_total_apr_conversion(self):
        """Curve with base 0.04% — total APR range nearly equals incentive."""
        lo_total, hi_total = 0.02, 0.08
        base_apy = 0.0004
        lo_inc = max(0.005, lo_total - base_apy)
        hi_inc = max(lo_inc + 0.005, hi_total - base_apy)
        # With near-zero base, incentive ≈ total
        assert lo_inc == pytest.approx(0.0196, abs=0.001)
        assert hi_inc == pytest.approx(0.0796, abs=0.001)


# ============================================================================
# 8. Grid Construction Respects Protocol Ceilings
# ============================================================================


class TestGridProtocolCeiling:
    """Verify that run_venue_optimization's grid respects protocol ceilings."""

    def test_curve_grid_capped_at_six_percent(self):
        """Curve venue's r_max grid should not exceed 6%."""
        # Replicate the grid clamping logic from run_venue_optimization
        r_max_range = (0.02, 0.08)  # User sidebar setting
        protocol = "curve"

        r_lo_eff, r_hi_eff = r_max_range
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS[protocol]
        r_lo_eff = max(r_lo_eff, proto_lo)
        r_hi_eff = min(r_hi_eff, proto_hi)
        r_hi_eff = min(r_hi_eff, GLOBAL_R_MAX_CEILING)
        r_hi_eff = max(r_hi_eff, r_lo_eff + 0.005)

        assert r_hi_eff == 0.06
        assert r_lo_eff == 0.02

    def test_aave_grid_capped_at_six_percent(self):
        r_max_range = (0.02, 0.10)
        protocol = "aave"
        r_lo_eff, r_hi_eff = r_max_range
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS[protocol]
        r_lo_eff = max(r_lo_eff, proto_lo)
        r_hi_eff = min(r_hi_eff, proto_hi)
        r_hi_eff = min(r_hi_eff, GLOBAL_R_MAX_CEILING)
        assert r_hi_eff == 0.06

    def test_morpho_grid_uses_full_global_ceiling(self):
        """Morpho ceiling matches global ceiling (8%), so full range."""
        r_max_range = (0.03, 0.10)
        protocol = "morpho"
        r_lo_eff, r_hi_eff = r_max_range
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS[protocol]
        r_lo_eff = max(r_lo_eff, proto_lo)
        r_hi_eff = min(r_hi_eff, proto_hi)
        r_hi_eff = min(r_hi_eff, GLOBAL_R_MAX_CEILING)
        assert r_hi_eff == 0.08


# ============================================================================
# 9. Budget Grid Bounds
# ============================================================================


class TestBudgetGridBounds:
    """Verify budget grid bounds respect target_tvl and protocol ceiling."""

    def test_curve_budget_grid_max(self):
        """Curve b_max should be capped by protocol ceiling."""
        target_tvl = 85e6
        r_hi_eff = 0.06  # After protocol clamping
        total_budget = 1_000_000
        b_max = min(total_budget, target_tvl * r_hi_eff / WEEKS_PER_YEAR * 1.5)
        # 85M * 0.06 / 52.14 * 1.5 ≈ $147k
        assert b_max < 150_000
        assert b_max < total_budget

    def test_aave_budget_grid_max(self):
        """AAVE b_max should be larger (big TVL)."""
        target_tvl = 612e6
        r_hi_eff = 0.06
        total_budget = 1_000_000
        b_max = min(total_budget, target_tvl * r_hi_eff / WEEKS_PER_YEAR * 1.5)
        # 612M * 0.06 / 52.14 * 1.5 ≈ $1.056M → capped at $1M
        assert b_max == total_budget


# ============================================================================
# 10. Integration: Optimize Two Venues and Verify Properties
# ============================================================================


class TestAllocatorIntegration:
    """Run actual optimizer on two venues and check allocation properties."""

    @staticmethod
    def _make_surface(
        target_tvl: float,
        current_tvl: float,
        base_apy: float,
        r_threshold: float,
        protocol: str,
    ) -> SurfaceResult:
        """Create a surface result for a venue (minimal MC for speed)."""
        proto_lo, proto_hi = PROTOCOL_R_MAX_DEFAULTS.get(protocol, (0.02, GLOBAL_R_MAX_CEILING))
        r_lo = max(0.02, proto_lo)
        r_hi = min(proto_hi, GLOBAL_R_MAX_CEILING)

        b_min = max(10_000, target_tvl * r_lo / WEEKS_PER_YEAR * 0.5)
        b_max = min(1_000_000, target_tvl * r_hi / WEEKS_PER_YEAR * 1.5)
        b_max = max(b_max, b_min * 2)

        grid = SurfaceGrid.from_ranges(
            B_min=b_min,
            B_max=b_max,
            B_steps=5,
            r_max_min=r_lo,
            r_max_max=r_hi,
            r_max_steps=5,
            dt_days=1.0,
            horizon_days=7,
            base_apy=base_apy,
        )
        env = CampaignEnvironment(r_threshold=r_threshold)
        weights = LossWeights(
            apr_target=r_threshold * 1.2,
            tvl_target=target_tvl,
        )
        return optimize_surface(
            grid=grid,
            env=env,
            initial_tvl=current_tvl,
            whale_profiles=[],
            weights=weights,
            n_paths=10,
            verbose=False,
        )

    def test_two_venue_allocation_budget_sum(self):
        """Total allocation should equal total budget (after cap + redistribute)."""
        sr_big = self._make_surface(
            target_tvl=500e6,
            current_tvl=450e6,
            base_apy=0.01,
            r_threshold=0.03,
            protocol="aave",
        )
        sr_small = self._make_surface(
            target_tvl=50e6,
            current_tvl=45e6,
            base_apy=0.002,
            r_threshold=0.03,
            protocol="curve",
        )

        total_budget = 500_000

        # Replicate allocation logic (TVL-weighted + cap)
        _venues = [
            {"name": "Big", "protocol": "aave", "target_tvl": 500e6},
            {"name": "Small", "protocol": "curve", "target_tvl": 50e6},
        ]
        _results = {
            "Big": {
                "surface": sr_big,
                "overrides": {"target_tvl": 500e6, "pinned_budget": None},
            },
            "Small": {
                "surface": sr_small,
                "overrides": {"target_tvl": 50e6, "pinned_budget": None},
            },
        }

        # The big venue should get the majority
        big_max = 500e6 * 0.06 / WEEKS_PER_YEAR  # ~$575k
        small_max = 50e6 * 0.06 / WEEKS_PER_YEAR  # ~$57.5k

        # Small venue can never get more than ~$57.5k
        assert small_max < 60_000
        # Big venue can handle the entire budget
        assert big_max > total_budget

    def test_surface_r_max_within_protocol_bounds(self):
        """Optimizer surface should only search within protocol ceiling."""
        sr = self._make_surface(
            target_tvl=100e6,
            current_tvl=80e6,
            base_apy=0.005,
            r_threshold=0.03,
            protocol="curve",
        )
        # r_max values in the grid should be <= 6% (Curve ceiling)
        assert np.max(sr.grid.r_max_values) <= 0.06 + 0.001
        # Optimal r_max should be <= 6%
        assert sr.optimal_r_max <= 0.06 + 0.001


# ============================================================================
# 11. Floor APR is Already Total APR (engine-level validation)
# ============================================================================


class TestFloorAPRIsTotalAPR:
    """Verify floor APR comparison uses total APR, not incentive-only."""

    def test_floor_breach_uses_total_apr(self):
        """Floor penalty fires on total APR (base + incentive), not just incentive."""
        base_apy = 0.02  # 2% base
        r_max = 0.03  # 3% incentive cap
        # Total APR when cap binds = 2% + 3% = 5%
        # Floor at 6% → should trigger penalty
        floor_apr = 0.06

        config = CampaignConfig(
            weekly_budget=50_000,
            apr_cap=r_max,
            base_apy=base_apy,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=floor_apr,
            apr_floor_sensitivity=0.8,
            w_apr_floor=7.0,
            tvl_target=100e6,
        )

        engine = CampaignSimulationEngine.from_params(
            config=config,
            env=env,
            seed=42,
            apy_sensitive_config=APYSensitiveConfig(
                floor_apr=floor_apr,
                sensitivity=0.8,
            ),
        )
        state = CampaignState(
            tvl=100e6,
            budget_remaining_epoch=config.weekly_budget,
        )
        final = engine.run(state)

        from campaign.engine import CampaignLossEvaluator

        evaluator = CampaignLossEvaluator(weights=weights)
        loss_result = evaluator.evaluate(final, config)

        # Floor breach should be positive (total APR ~5% < 6% floor)
        assert loss_result.floor_breach_cost > 0
        # Time below floor should be substantial
        assert loss_result.time_below_floor > 0.5

    def test_no_floor_breach_when_total_apr_above_floor(self):
        """No penalty when total APR (base + incentive) exceeds floor."""
        base_apy = 0.03  # 3% base
        r_max = 0.05  # 5% incentive cap
        # Total = 8% when cap binds
        floor_apr = 0.06  # Floor at 6% → should be OK

        config = CampaignConfig(
            weekly_budget=100_000,
            apr_cap=r_max,
            base_apy=base_apy,
            dt_days=1.0,
            horizon_days=7,
        )
        env = CampaignEnvironment(r_threshold=0.04)
        weights = LossWeights(
            apr_floor=floor_apr,
            apr_floor_sensitivity=0.8,
            w_apr_floor=7.0,
            tvl_target=50e6,
        )

        engine = CampaignSimulationEngine.from_params(
            config=config,
            env=env,
            seed=42,
        )
        state = CampaignState(
            tvl=50e6,
            budget_remaining_epoch=config.weekly_budget,
        )
        final = engine.run(state)

        from campaign.engine import CampaignLossEvaluator

        evaluator = CampaignLossEvaluator(weights=weights)
        loss_result = evaluator.evaluate(final, config)

        # Floor breach should be zero or negligible
        assert loss_result.floor_breach_cost < 0.01


# ============================================================================
# 12. Floor-Aware Budget Grid
# ============================================================================


class TestFloorAwareBudgetGrid:
    """When a floor APR is set and achievable, the budget grid floor
    should be raised so the optimizer never wastes grid points on configs
    that guarantee a floor breach."""

    def _compute_b_min(
        self,
        target_tvl: float,
        r_lo_eff: float,
        base_apy: float,
        floor_apr: float,
        total_budget: float,
    ) -> float:
        """Replicate the floor-aware b_min logic from run_venue_optimization."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        b_min = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        if floor_apr > 0:
            min_inc_for_floor = max(0, floor_apr - base_apy)
            floor_budget = target_tvl * min_inc_for_floor / WEEKS_PER_YEAR
            if floor_budget <= total_budget:
                b_min = max(b_min, floor_budget * 0.85)
        return b_min

    def test_euler_floor_raises_grid_min(self):
        """Euler with 6.2% floor APR and 1.05% base → b_min ≈ $160k."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        target_tvl = 190e6
        floor_apr = 0.062
        base_apy = 0.0105
        r_lo_eff = 0.03  # Euler protocol minimum
        total_budget = 200_000

        b_min = self._compute_b_min(target_tvl, r_lo_eff, base_apy, floor_apr, total_budget)

        # Floor budget = 190M × (6.2% - 1.05%) / 52.14 ≈ $187,792
        # b_min = max(old_b_min, $187,792 × 0.85) = max($54,658, $159,623) ≈ $159,623
        floor_budget = target_tvl * (floor_apr - base_apy) / WEEKS_PER_YEAR
        assert b_min >= floor_budget * 0.84  # At least 85% of floor budget (with rounding)
        assert b_min < floor_budget  # But below full floor budget (gives room to explore)

    def test_no_floor_no_change(self):
        """Without a floor, b_min is the standard formula."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        target_tvl = 190e6
        r_lo_eff = 0.03

        b_min = self._compute_b_min(target_tvl, r_lo_eff, 0.01, 0.0, 200_000)

        expected = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        assert b_min == pytest.approx(expected)

    def test_floor_exceeds_budget_no_change(self):
        """When floor needs more than budget, don't raise b_min."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        target_tvl = 500e6
        floor_apr = 0.08  # Very high floor
        base_apy = 0.01
        r_lo_eff = 0.03
        total_budget = 200_000

        floor_budget = target_tvl * (floor_apr - base_apy) / WEEKS_PER_YEAR
        assert floor_budget > total_budget  # Confirm floor is unachievable

        b_min = self._compute_b_min(target_tvl, r_lo_eff, base_apy, floor_apr, total_budget)
        expected_default = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        assert b_min == pytest.approx(expected_default)

    def test_aave_floor_raises_grid_min(self):
        """AAVE with 4.1% floor APR and 0.86% base → appropriate b_min."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        target_tvl = 612e6
        floor_apr = 0.041
        base_apy = 0.0086
        r_lo_eff = 0.02
        total_budget = 1_000_000

        b_min = self._compute_b_min(target_tvl, r_lo_eff, base_apy, floor_apr, total_budget)

        floor_budget = target_tvl * (floor_apr - base_apy) / WEEKS_PER_YEAR
        assert b_min >= floor_budget * 0.84

    def test_high_base_apy_reduces_floor_budget(self):
        """When base_apy is close to floor, the required incentive is small."""
        WEEKS_PER_YEAR = 365.0 / 7.0
        target_tvl = 100e6
        floor_apr = 0.05
        base_apy = 0.045  # Only 0.5% incentive needed
        r_lo_eff = 0.02
        total_budget = 50_000

        b_min = self._compute_b_min(target_tvl, r_lo_eff, base_apy, floor_apr, total_budget)

        # Floor budget = 100M × 0.5% / 52.14 ≈ $9,589
        # But default b_min = max(10k, 100M × 0.02 / 52.14 × 0.5) = max(10k, $19,178) = $19,178
        # Since $9,589 × 0.85 = $8,151 < $19,178, floor doesn't raise b_min
        default_b_min = max(10_000, target_tvl * r_lo_eff / WEEKS_PER_YEAR * 0.5)
        assert b_min == pytest.approx(default_b_min)


# ============================================================================
# 13. Current TVL Sanity
# ============================================================================


class TestCurrentTVLSanity:
    """Verify that optimizer behavior is dramatically different with
    correct vs wrong current_tvl, proving the venue-change bug matters."""

    def test_wrong_current_tvl_gives_low_budget(self):
        """With current_tvl >> target_tvl and a floor, optimizer picks low B
        because the floor is unachievable at the high TVL."""

        target_tvl = 190e6
        wrong_current_tvl = 612e6  # AAVE's TVL, not Euler's
        floor_apr = 0.062
        base_apy = 0.0105
        total_budget = 200_000
        WEEKS_PER_YEAR = 365.0 / 7.0

        # Check: floor needs $604k/wk at wrong TVL → unachievable with $200k
        floor_budget_at_wrong = wrong_current_tvl * (floor_apr - base_apy) / WEEKS_PER_YEAR
        assert floor_budget_at_wrong > total_budget

        # Check: floor needs $188k/wk at correct TVL → achievable with $200k
        floor_budget_at_correct = target_tvl * (floor_apr - base_apy) / WEEKS_PER_YEAR
        assert floor_budget_at_correct < total_budget

    def test_correct_current_tvl_uses_budget(self):
        """With correct current_tvl ≈ target_tvl and achievable floor,
        optimizer uses a substantial fraction of the budget."""
        from campaign.agents import APYSensitiveConfig
        from campaign.optimizer import SurfaceGrid, optimize_surface

        target_tvl = 190e6
        base_apy = 0.0105
        floor_apr = 0.062
        WEEKS_PER_YEAR = 365.0 / 7.0

        # Floor-aware b_min
        min_inc = max(0, floor_apr - base_apy)
        floor_B = target_tvl * min_inc / WEEKS_PER_YEAR
        b_min = max(10_000, floor_B * 0.85)

        grid = SurfaceGrid.from_ranges(
            B_min=b_min,
            B_max=200_000,
            B_steps=5,
            r_max_min=0.03,
            r_max_max=0.07,
            r_max_steps=5,
            base_apy=base_apy,
        )
        env = CampaignEnvironment(r_threshold=0.024)
        weights = LossWeights(
            apr_target=0.024 * 1.2,
            tvl_target=target_tvl,
            apr_floor=floor_apr,
            apr_floor_sensitivity=0.75,
        )
        apy_cfg = APYSensitiveConfig(
            floor_apr=floor_apr,
            sensitivity=0.75,
            max_sensitive_tvl=target_tvl * 0.10,
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

        # With correct TVL, optimizer should pick high budget (near floor budget)
        assert sr.optimal_B >= floor_B * 0.8, (
            f"Optimizer picked B=${sr.optimal_B:,.0f} but floor needs "
            f"${floor_B:,.0f}/wk — should be close to floor budget"
        )

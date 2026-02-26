"""
Tests for Phase 1.4: cap-proximity r_max taper utilities.

Covers:
  - cap_proximity_r_max_ceiling() arithmetic
  - apply_cap_proximity_taper() trigger / no-trigger conditions
  - SurfaceResult.cap_taper_info default
  - CAP_PROXIMITY_THRESHOLD constant value
"""

from __future__ import annotations

import numpy as np
import pytest
from campaign.optimizer import (
    CAP_PROXIMITY_THRESHOLD,
    SurfaceGrid,
    SurfaceResult,
    apply_cap_proximity_taper,
    cap_proximity_r_max_ceiling,
)

_WEEKS_PER_YEAR = 365.0 / 7.0


# ============================================================================
# cap_proximity_r_max_ceiling
# ============================================================================


class TestCapProximityRMaxCeiling:
    def test_basic_math(self):
        """Ceiling = (B * 52.14 / cap) * 1.05."""
        b = 10_000.0       # $10 k/week
        cap = 1_000_000.0  # $1 M supply cap
        result = cap_proximity_r_max_ceiling(b, cap, buffer=0.05)
        expected = (b * _WEEKS_PER_YEAR / cap) * 1.05
        assert abs(result - expected) < 1e-9

    def test_no_buffer(self):
        """buffer=0 gives exact breakeven."""
        b = 5_000.0
        cap = 500_000.0
        result = cap_proximity_r_max_ceiling(b, cap, buffer=0.0)
        expected = b * _WEEKS_PER_YEAR / cap
        assert abs(result - expected) < 1e-9

    def test_larger_buffer(self):
        """Larger buffer raises ceiling proportionally."""
        b = 10_000.0
        cap = 1_000_000.0
        r0 = cap_proximity_r_max_ceiling(b, cap, buffer=0.0)
        r10 = cap_proximity_r_max_ceiling(b, cap, buffer=0.10)
        assert abs(r10 / r0 - 1.10) < 1e-9

    def test_raises_on_zero_cap(self):
        with pytest.raises(ValueError, match="supply_cap_usd must be positive"):
            cap_proximity_r_max_ceiling(10_000.0, 0.0)

    def test_raises_on_negative_cap(self):
        with pytest.raises(ValueError, match="supply_cap_usd must be positive"):
            cap_proximity_r_max_ceiling(10_000.0, -1.0)

    def test_proportional_to_budget(self):
        """Doubling budget doubles the ceiling."""
        cap = 2_000_000.0
        r1 = cap_proximity_r_max_ceiling(10_000.0, cap)
        r2 = cap_proximity_r_max_ceiling(20_000.0, cap)
        assert abs(r2 / r1 - 2.0) < 1e-9

    def test_inversely_proportional_to_cap(self):
        """Doubling cap halves the ceiling."""
        b = 10_000.0
        r1 = cap_proximity_r_max_ceiling(b, 1_000_000.0)
        r2 = cap_proximity_r_max_ceiling(b, 2_000_000.0)
        assert abs(r2 / r1 - 0.5) < 1e-9


# ============================================================================
# apply_cap_proximity_taper
# ============================================================================


class TestApplyCapProximityTaper:
    def test_taper_triggered_above_threshold(self):
        """If target_tvl/supply_cap > 0.9 and ceiling < r_max_max, taper fires."""
        supply_cap = 1_000_000.0
        target_tvl = 950_000.0   # 95% of cap → above threshold
        b_max = 5_000.0           # ceiling ≈ 5000*52.14/1M * 1.05 ≈ 0.274
        r_max_max = 0.50          # well above ceiling

        clipped, applied = apply_cap_proximity_taper(
            r_max_max=r_max_max,
            b_max=b_max,
            target_tvl=target_tvl,
            supply_cap_usd=supply_cap,
        )
        assert applied is True
        assert clipped < r_max_max
        expected_ceil = cap_proximity_r_max_ceiling(b_max, supply_cap)
        assert abs(clipped - expected_ceil) < 1e-9

    def test_no_taper_at_threshold_boundary(self):
        """Exactly at threshold (== 0.9) should NOT trigger."""
        supply_cap = 1_000_000.0
        target_tvl = 900_000.0  # exactly 90%
        _, applied = apply_cap_proximity_taper(
            r_max_max=0.50,
            b_max=5_000.0,
            target_tvl=target_tvl,
            supply_cap_usd=supply_cap,
        )
        assert applied is False

    def test_no_taper_below_threshold(self):
        """Below threshold → no taper."""
        supply_cap = 1_000_000.0
        target_tvl = 500_000.0  # 50%
        _, applied = apply_cap_proximity_taper(
            r_max_max=0.50,
            b_max=5_000.0,
            target_tvl=target_tvl,
            supply_cap_usd=supply_cap,
        )
        assert applied is False

    def test_no_taper_when_supply_cap_zero(self):
        """supply_cap=0 (unlimited) never triggers taper."""
        clipped, applied = apply_cap_proximity_taper(
            r_max_max=0.50,
            b_max=5_000.0,
            target_tvl=999_999.0,
            supply_cap_usd=0.0,
        )
        assert applied is False
        assert clipped == 0.50

    def test_no_taper_when_ceiling_already_above_r_max(self):
        """If ceiling >= r_max_max the grid is already conservative — no clip."""
        supply_cap = 1_000_000.0
        target_tvl = 950_000.0
        # Very large budget → ceiling >> r_max_max
        b_max = 1_000_000.0
        r_max_max = 0.10
        clipped, applied = apply_cap_proximity_taper(
            r_max_max=r_max_max,
            b_max=b_max,
            target_tvl=target_tvl,
            supply_cap_usd=supply_cap,
        )
        assert applied is False
        assert clipped == r_max_max

    def test_custom_threshold(self):
        """Custom threshold parameter respected."""
        supply_cap = 1_000_000.0
        target_tvl = 850_000.0  # 85%
        # Should not trigger at threshold=0.9 but should at threshold=0.8
        _, applied_09 = apply_cap_proximity_taper(
            r_max_max=0.50, b_max=5_000.0,
            target_tvl=target_tvl, supply_cap_usd=supply_cap, threshold=0.9,
        )
        _, applied_08 = apply_cap_proximity_taper(
            r_max_max=0.50, b_max=5_000.0,
            target_tvl=target_tvl, supply_cap_usd=supply_cap, threshold=0.8,
        )
        assert applied_09 is False
        assert applied_08 is True

    def test_clipped_value_is_strictly_lower(self):
        """Clipped r_max is strictly below original r_max_max."""
        clipped, applied = apply_cap_proximity_taper(
            r_max_max=0.40,
            b_max=5_000.0,
            target_tvl=980_000.0,
            supply_cap_usd=1_000_000.0,
        )
        assert applied is True
        assert clipped < 0.40

    def test_no_taper_when_target_tvl_zero(self):
        """target_tvl=0 → guard clause fires, no taper."""
        _, applied = apply_cap_proximity_taper(
            r_max_max=0.50,
            b_max=5_000.0,
            target_tvl=0.0,
            supply_cap_usd=1_000_000.0,
        )
        assert applied is False


# ============================================================================
# CAP_PROXIMITY_THRESHOLD constant
# ============================================================================


class TestCapProximityThreshold:
    def test_threshold_is_09(self):
        assert CAP_PROXIMITY_THRESHOLD == 0.9


# ============================================================================
# SurfaceResult.cap_taper_info default
# ============================================================================


class TestSurfaceResultCapTaperInfo:
    def _make_minimal_result(self) -> SurfaceResult:
        grid = SurfaceGrid.from_ranges(
            B_min=1000, B_max=2000, B_steps=2,
            r_max_min=0.02, r_max_max=0.10, r_max_steps=2,
        )
        n_b, n_r = grid.shape
        zeros = np.zeros((n_b, n_r))
        ones_bool = np.ones((n_b, n_r), dtype=bool)
        return SurfaceResult(
            grid=grid,
            loss_surface=zeros,
            loss_std_surface=zeros,
            feasibility_mask=ones_bool,
        )

    def test_cap_taper_info_defaults_to_none(self):
        sr = self._make_minimal_result()
        assert sr.cap_taper_info is None

    def test_cap_taper_info_can_be_set(self):
        sr = self._make_minimal_result()
        sr.cap_taper_info = "test taper message"
        assert sr.cap_taper_info == "test taper message"

    def test_cap_taper_info_can_be_cleared(self):
        sr = self._make_minimal_result()
        sr.cap_taper_info = "set"
        sr.cap_taper_info = None
        assert sr.cap_taper_info is None

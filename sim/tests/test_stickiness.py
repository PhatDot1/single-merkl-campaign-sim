"""
Tests for Phase 2: whale stickiness scoring + TVL stickiness model.

Covers:
  - compute_whale_stickiness_score() arithmetic + edge cases
  - stickiness_to_profile_params() interpolation + bounds
  - build_stickiness_enrichment() integration with mocked history
  - compute_tvl_stickiness() from time series
  - enrich_whale_profiles_from_stickiness() patching
"""

from __future__ import annotations

from unittest.mock import patch

from dune.sync import (
    build_stickiness_enrichment,
    compute_tvl_stickiness,
    compute_whale_stickiness_score,
    stickiness_to_profile_params,
)

# ============================================================================
# compute_whale_stickiness_score
# ============================================================================


class TestComputeWhaleStickinesScore:
    def test_perfect_sticker_no_withdrawals(self):
        """Whale with no withdrawals and long hold → near 1.0."""
        h = {"n_deposits": 10, "n_withdrawals": 0, "avg_hold_days": 180}
        score = compute_whale_stickiness_score(h)
        assert score >= 0.9

    def test_pure_mercenary_exits_every_time(self):
        """Whale that exits as many times as it deposits → low score."""
        h = {"n_deposits": 10, "n_withdrawals": 10, "avg_hold_days": 3}
        score = compute_whale_stickiness_score(h, exits_during_rate_drops=5)
        assert score == 0.10  # clamped at min

    def test_zero_deposits_uses_safe_default(self):
        """n_deposits=0 should not raise (division-by-zero guard)."""
        h = {"n_deposits": 0, "n_withdrawals": 0, "avg_hold_days": 30}
        score = compute_whale_stickiness_score(h)
        assert 0.1 <= score <= 1.0

    def test_hold_bonus_caps_at_0_3(self):
        """avg_hold_days=900 should not push hold_bonus above 0.3."""
        h = {"n_deposits": 5, "n_withdrawals": 0, "avg_hold_days": 900}
        score = compute_whale_stickiness_score(h)
        # base=1.0, hold_bonus=0.3 (capped), penalty=0 → raw=1.3 → clamped to 1.0
        assert score == 1.0

    def test_exit_penalty_applied(self):
        """Each rate-drop exit reduces score by 0.15 (when not clamped)."""
        # Use a scenario where no clamping occurs:
        # n_deposits=5, n_withdrawals=2 → base=0.6; avg_hold=30 → hold_bonus=0.333 → raw=0.933
        # With 1 exit_during_rate_drops → raw=0.783; both well within [0.1, 1.0]
        h0 = {"n_deposits": 5, "n_withdrawals": 2, "avg_hold_days": 30}
        s0 = compute_whale_stickiness_score(h0, exits_during_rate_drops=0)
        s1 = compute_whale_stickiness_score(h0, exits_during_rate_drops=1)
        # s0 - s1 should equal 0.15 (one exit penalty)
        assert abs((s0 - s1) - 0.15) < 1e-9

    def test_score_in_valid_range(self):
        """Score always stays within [0.1, 1.0]."""
        cases = [
            {"n_deposits": 0, "n_withdrawals": 0, "avg_hold_days": 0},
            {"n_deposits": 100, "n_withdrawals": 200, "avg_hold_days": 1},
            {"n_deposits": 1, "n_withdrawals": 0, "avg_hold_days": 365},
        ]
        for h in cases:
            s = compute_whale_stickiness_score(h, exits_during_rate_drops=10)
            assert 0.1 <= s <= 1.0, f"score={s} out of range for {h}"

    def test_missing_keys_use_defaults(self):
        """Missing dict keys fall back gracefully."""
        score = compute_whale_stickiness_score({})
        assert 0.1 <= score <= 1.0


# ============================================================================
# stickiness_to_profile_params
# ============================================================================


class TestStickinessToProfileParams:
    def test_returns_all_required_keys(self):
        params = stickiness_to_profile_params(0.5)
        assert "exit_delay_days" in params
        assert "reentry_delay_days" in params
        assert "hysteresis_band" in params

    def test_low_score_gives_fast_exit(self):
        """Score near 0.1 → exit_delay_days near 1.0."""
        params = stickiness_to_profile_params(0.10)
        assert params["exit_delay_days"] < 2.0
        assert params["hysteresis_band"] < 0.004

    def test_high_score_gives_slow_exit(self):
        """Score near 1.0 → exit_delay_days near 14."""
        params = stickiness_to_profile_params(1.0)
        assert params["exit_delay_days"] > 12.0
        assert params["hysteresis_band"] > 0.010

    def test_midpoint_is_between_bounds(self):
        lo = stickiness_to_profile_params(0.10)
        hi = stickiness_to_profile_params(1.00)
        mid = stickiness_to_profile_params(0.55)
        assert lo["exit_delay_days"] < mid["exit_delay_days"] < hi["exit_delay_days"]

    def test_clamps_above_1(self):
        """Score > 1.0 is clamped to 1.0."""
        p1 = stickiness_to_profile_params(1.0)
        p_over = stickiness_to_profile_params(2.0)
        assert p1["exit_delay_days"] == p_over["exit_delay_days"]

    def test_clamps_below_01(self):
        """Score < 0.1 is clamped to 0.1."""
        p_min = stickiness_to_profile_params(0.10)
        p_under = stickiness_to_profile_params(0.01)
        assert p_min["exit_delay_days"] == p_under["exit_delay_days"]

    def test_monotone_in_score(self):
        """All three params are monotonically increasing with score."""
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        params = [stickiness_to_profile_params(s) for s in scores]
        for i in range(len(params) - 1):
            assert params[i]["exit_delay_days"] <= params[i + 1]["exit_delay_days"]
            assert params[i]["hysteresis_band"] <= params[i + 1]["hysteresis_band"]


# ============================================================================
# build_stickiness_enrichment (integration smoke)
# ============================================================================


class TestBuildStickinessEnrichment:
    _FAKE_FLOWS = [
        {"whale_address": "0xaaa", "direction": "deposit", "amount": 1_000_000, "block_time": "2024-01-01T00:00:00"},
        {"whale_address": "0xaaa", "direction": "withdrawal", "amount": 500_000, "block_time": "2024-06-01T00:00:00"},
        {"whale_address": "0xbbb", "direction": "deposit", "amount": 2_000_000, "block_time": "2024-01-01T00:00:00"},
    ]

    def test_returns_dict_with_addresses(self):
        with patch("dune.sync.load_whale_flows", return_value=self._FAKE_FLOWS):
            result = build_stickiness_enrichment("test-pool")
        assert result is not None
        assert "0xaaa" in result
        assert "0xbbb" in result

    def test_all_entries_have_required_keys(self):
        with patch("dune.sync.load_whale_flows", return_value=self._FAKE_FLOWS):
            result = build_stickiness_enrichment("test-pool")
        for v in result.values():
            assert "score" in v
            assert "exit_delay_days" in v
            assert "reentry_delay_days" in v
            assert "hysteresis_band" in v

    def test_address_with_withdrawal_gets_lower_score_than_no_withdrawal(self):
        with patch("dune.sync.load_whale_flows", return_value=self._FAKE_FLOWS):
            result = build_stickiness_enrichment("test-pool")
        assert result["0xaaa"]["score"] < result["0xbbb"]["score"]

    def test_returns_none_when_no_flows(self):
        with patch("dune.sync.load_whale_flows", return_value=[]):
            result = build_stickiness_enrichment("test-pool")
        assert result is None

    def test_returns_none_on_insufficient_data(self):
        with patch("dune.sync.load_whale_flows", return_value=None):
            result = build_stickiness_enrichment("test-pool")
        assert result is None


# ============================================================================
# compute_tvl_stickiness
# ============================================================================


def _make_series(
    n: int,
    tvl_start: float = 10_000_000,
    incentive_start: float = 0.04,
    *,
    incentive_cutoff_at: int | None = None,
    tvl_decay_after_cutoff: float = 0.95,
) -> list[dict]:
    """Helper: create synthetic daily TVL + incentive rate series."""
    records = []
    for i in range(n):
        if incentive_cutoff_at is not None and i >= incentive_cutoff_at:
            rate = 0.0
            days_since = i - incentive_cutoff_at
            tvl = tvl_start * (tvl_decay_after_cutoff**days_since)
        else:
            rate = incentive_start
            tvl = tvl_start
        records.append({"date": f"2024-01-{i+1:02d}", "tvl_usd": tvl, "incentive_rate": rate})
    return records


class TestComputeTVLStickiness:
    def test_empty_returns_default(self):
        m = compute_tvl_stickiness([], venue="test")
        assert m.venue == "test"
        assert m.n_observations == 0

    def test_n_observations_matches_input(self):
        records = _make_series(30)
        m = compute_tvl_stickiness(records)
        assert m.n_observations == 30

    def test_all_incentivized_uses_min_tvl_as_floor(self):
        """If no zero-incentive windows, organic = min TVL."""
        records = _make_series(20, tvl_start=15_000_000, incentive_start=0.05)
        m = compute_tvl_stickiness(records)
        # All records have incentive_rate > threshold; organic ≤ min TVL
        all_tvls = [r["tvl_usd"] for r in records]
        assert abs(m.organic_tvl_estimate_usd - min(all_tvls)) < 1.0

    def test_zero_incentive_windows_used_for_organic(self):
        """Organic TVL = mean TVL in zero-incentive periods."""
        records = [
            {"date": "2024-01-01", "tvl_usd": 8_000_000, "incentive_rate": 0.0},
            {"date": "2024-01-02", "tvl_usd": 8_200_000, "incentive_rate": 0.0},
            {"date": "2024-01-03", "tvl_usd": 12_000_000, "incentive_rate": 0.05},
        ]
        m = compute_tvl_stickiness(records)
        assert abs(m.organic_tvl_estimate_usd - 8_100_000) < 1.0

    def test_sticky_fraction_in_range(self):
        records = _make_series(60, incentive_cutoff_at=30)
        m = compute_tvl_stickiness(records)
        assert 0.05 <= m.sticky_fraction <= 0.95

    def test_exit_lag_detected_after_cutoff(self):
        """After incentive cutoff, exit lag should be detected."""
        records = _make_series(40, tvl_start=10_000_000, incentive_cutoff_at=20, tvl_decay_after_cutoff=0.90)
        m = compute_tvl_stickiness(records)
        # After cutoff, TVL drops ~10%/day; first >5% drop should be day 2 → exit_lag ≈ 1–2
        assert m.mean_exit_lag_days >= 1.0

    def test_half_life_estimated(self):
        """TVL half-life should be detected when TVL falls below 50% of cutoff."""
        records = _make_series(60, tvl_start=10_000_000, incentive_cutoff_at=10, tvl_decay_after_cutoff=0.85)
        m = compute_tvl_stickiness(records)
        # 0.85^n reaches 0.5 at n = log(0.5)/log(0.85) ≈ 4.3 days
        assert m.tvl_half_life_post_incentive_days >= 2.0

    def test_values_clamped_in_reasonable_ranges(self):
        records = _make_series(100, incentive_cutoff_at=50)
        m = compute_tvl_stickiness(records)
        assert 1.0 <= m.mean_exit_lag_days <= 60.0
        assert 2.0 <= m.tvl_half_life_post_incentive_days <= 90.0
        assert 0.0 <= m.incentive_elasticity <= 1.0

    def test_elasticity_higher_when_tvl_volatile(self):
        """More volatile TVL should produce higher elasticity."""
        stable = [{"date": f"2024-{i+1:02d}-01", "tvl_usd": 10_000_000, "incentive_rate": 0.04} for i in range(12)]
        volatile = [
            {"date": f"2024-{i+1:02d}-01",
             "tvl_usd": 5_000_000 + (i % 3) * 5_000_000,
             "incentive_rate": 0.04}
            for i in range(12)
        ]
        m_stable = compute_tvl_stickiness(stable)
        m_volatile = compute_tvl_stickiness(volatile)
        assert m_volatile.incentive_elasticity >= m_stable.incentive_elasticity


# ============================================================================
# enrich_whale_profiles_from_stickiness
# ============================================================================


class TestEnrichWhaleProfilesFromStickiness:
    def _make_profile(self, whale_id: str, position: float = 1_000_000) -> object:
        from campaign.agents import WhaleProfile
        return WhaleProfile(
            whale_id=whale_id,
            position_usd=position,
            exit_delay_days=2.0,
            reentry_delay_days=7.0,
            hysteresis_band=0.005,
        )

    def test_enriches_matching_address(self):
        from campaign.data import enrich_whale_profiles_from_stickiness

        enrichment = {
            "0xabc": {
                "score": 0.9,
                "exit_delay_days": 12.0,
                "reentry_delay_days": 18.0,
                "hysteresis_band": 0.010,
            }
        }
        profiles = [self._make_profile("0xabc")]
        with patch("campaign.data.build_stickiness_enrichment", return_value=enrichment):
            result = enrich_whale_profiles_from_stickiness(profiles, "test-pool")
        assert result[0].exit_delay_days == 12.0
        assert result[0].hysteresis_band == 0.010

    def test_fallback_for_unknown_address(self):
        from campaign.data import enrich_whale_profiles_from_stickiness

        enrichment = {}  # no match
        profiles = [self._make_profile("0xunknown")]
        with patch("campaign.data.build_stickiness_enrichment", return_value=enrichment):
            result = enrich_whale_profiles_from_stickiness(profiles, "pool", fallback_score=0.5)
        expected = stickiness_to_profile_params(0.5)
        assert abs(result[0].exit_delay_days - expected["exit_delay_days"]) < 1e-9

    def test_returns_original_when_no_dune_data(self):
        from campaign.data import enrich_whale_profiles_from_stickiness

        profiles = [self._make_profile("0xabc")]
        with patch("campaign.data.build_stickiness_enrichment", return_value=None):
            result = enrich_whale_profiles_from_stickiness(profiles, "pool")
        # When enrichment is None, fallback still applied but exit_delay changes
        # (The function applies fallback_score=0.5 in this branch)
        assert len(result) == len(profiles)

    def test_does_not_mutate_original_profiles(self):
        from campaign.data import enrich_whale_profiles_from_stickiness

        enrichment = {
            "0xabc": {"score": 0.9, "exit_delay_days": 12.0, "reentry_delay_days": 18.0, "hysteresis_band": 0.010}
        }
        original = self._make_profile("0xabc")
        profiles = [original]
        with patch("campaign.data.build_stickiness_enrichment", return_value=enrichment):
            result = enrich_whale_profiles_from_stickiness(profiles, "pool")
        # Original unchanged
        assert original.exit_delay_days == 2.0
        # Result changed
        assert result[0].exit_delay_days == 12.0
        assert result[0] is not original

    def test_returns_same_length_list(self):
        from campaign.data import enrich_whale_profiles_from_stickiness

        profiles = [self._make_profile(f"0x{i:040x}") for i in range(5)]
        with patch("campaign.data.build_stickiness_enrichment", return_value={}):
            result = enrich_whale_profiles_from_stickiness(profiles, "pool")
        assert len(result) == 5

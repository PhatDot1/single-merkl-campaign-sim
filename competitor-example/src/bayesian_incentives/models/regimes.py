"""
Regime analysis: classify periods into volatility/utilization/stress
regimes and estimate regime-conditional elasticities.

Volatility regimes:
    High:    sigma_t > Q75
    Extreme: sigma_t > Q90

Utilization regimes:
    High:    U_t > 0.85
    Extreme: U_t > 0.95

Stress regime:
    High vol AND (high util OR util increasing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bayesian_incentives.models.elasticity import ElasticityModel, OLSResult


@dataclass
class RegimeDefinition:
    """Threshold-based regime definition."""

    name: str
    column: str
    threshold: float
    direction: str = "above"  # "above" or "below"

    def classify(self, data: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where the regime condition holds."""
        if self.direction == "above":
            return data > self.threshold
        return data < self.threshold


@dataclass
class RegimeElasticityResult:
    """Regime-conditional elasticity comparison."""

    regime_name: str
    n_in: int
    n_out: int
    ols_in: OLSResult | None
    ols_out: OLSResult | None
    beta_diff: float | None = None
    bootstrap_ci: tuple[float, float] | None = None

    def summary(self) -> str:
        lines = [f"Regime: {self.regime_name}"]
        lines.append(f"  In-regime: n={self.n_in}")
        if self.ols_in:
            lines.append(f"    beta={self.ols_in.beta:,.2f}, R2={self.ols_in.r_squared:.4%}")
        lines.append(f"  Out-of-regime: n={self.n_out}")
        if self.ols_out:
            lines.append(f"    beta={self.ols_out.beta:,.2f}, R2={self.ols_out.r_squared:.4%}")
        if self.beta_diff is not None:
            lines.append(f"  beta_diff = {self.beta_diff:,.2f}")
        if self.bootstrap_ci:
            lines.append(f"  95% CI: [{self.bootstrap_ci[0]:,.2f}, {self.bootstrap_ci[1]:,.2f}]")
        return "\n".join(lines)


class RegimeAnalyzer:
    """Classify market regimes and compute regime-conditional elasticities.

    Parameters
    ----------
    vol_high_q : float
        Quantile threshold for high-volatility regime (default 0.75).
    vol_extreme_q : float
        Quantile threshold for extreme-volatility regime (default 0.90).
    util_high : float
        Utilization threshold for high utilization (default 0.85).
    util_extreme : float
        Utilization threshold for extreme utilization (default 0.95).
    """

    def __init__(
        self,
        vol_high_q: float = 0.75,
        vol_extreme_q: float = 0.90,
        util_high: float = 0.85,
        util_extreme: float = 0.95,
    ) -> None:
        self.vol_high_q = vol_high_q
        self.vol_extreme_q = vol_extreme_q
        self.util_high = util_high
        self.util_extreme = util_extreme

    def classify_volatility(
        self, realized_vol: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Classify periods by volatility quantile.

        Returns dict of boolean masks for each regime.
        """
        vol = np.asarray(realized_vol, dtype=float)
        q75 = np.nanquantile(vol, self.vol_high_q)
        q90 = np.nanquantile(vol, self.vol_extreme_q)
        return {
            "high_vol": vol > q75,
            "extreme_vol": vol > q90,
            "vol_q75": q75,
            "vol_q90": q90,
        }

    def classify_utilization(
        self, utilization: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Classify periods by utilization level."""
        u = np.asarray(utilization, dtype=float)
        return {
            "high_util": u > self.util_high,
            "extreme_util": u > self.util_extreme,
            "low_headroom": (1 - u) < 0.10,
        }

    def classify_stress(
        self,
        realized_vol: np.ndarray,
        utilization: np.ndarray,
    ) -> np.ndarray:
        """Stress regime: high vol AND (high util OR util increasing).

        Returns boolean mask.
        """
        vol_regimes = self.classify_volatility(realized_vol)
        util_regimes = self.classify_utilization(utilization)

        u = np.asarray(utilization, dtype=float)
        util_increasing = np.zeros(len(u), dtype=bool)
        util_increasing[1:] = np.diff(u) > 0

        stress = vol_regimes["high_vol"] & (
            util_regimes["high_util"] | util_increasing
        )
        return stress

    def fit_conditional_elasticity(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
        regime_mask: np.ndarray,
        regime_name: str = "regime",
        n_bootstrap: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> RegimeElasticityResult:
        """Estimate elasticity separately in- and out-of-regime, with
        bootstrap confidence interval for the difference.

        Parameters
        ----------
        d_incentive, d_tvl : array-like
            First differences of incentive and TVL.
        regime_mask : array-like of bool
            True for in-regime periods.
        regime_name : str
            Label for the regime.
        n_bootstrap : int
            Number of bootstrap resamples for CI.
        rng : numpy Generator, optional

        Returns
        -------
        RegimeElasticityResult
        """
        rng = rng or np.random.default_rng(42)
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.asarray(regime_mask, dtype=bool)

        # Align lengths
        min_len = min(len(x), len(y), len(mask))
        x, y, mask = x[:min_len], y[:min_len], mask[:min_len]

        # Valid observations
        valid = np.isfinite(x) & np.isfinite(y)
        x_in, y_in = x[valid & mask], y[valid & mask]
        x_out, y_out = x[valid & ~mask], y[valid & ~mask]

        ols_in = ols_out = None
        if len(x_in) >= 3:
            m_in = ElasticityModel()
            ols_in = m_in.fit_ols(x_in, y_in)
        if len(x_out) >= 3:
            m_out = ElasticityModel()
            ols_out = m_out.fit_ols(x_out, y_out)

        beta_diff = None
        bootstrap_ci = None
        if ols_in is not None and ols_out is not None:
            beta_diff = ols_in.beta - ols_out.beta

            # Bootstrap the difference
            diffs = []
            indices = np.arange(min_len)[valid]
            for _ in range(n_bootstrap):
                boot_idx = rng.choice(indices, size=len(indices), replace=True)
                bx, by, bm = x[boot_idx], y[boot_idx], mask[boot_idx]
                bx_in, by_in = bx[bm], by[bm]
                bx_out, by_out = bx[~bm], by[~bm]
                if len(bx_in) >= 3 and len(bx_out) >= 3:
                    m1 = ElasticityModel()
                    m2 = ElasticityModel()
                    r1 = m1.fit_ols(bx_in, by_in)
                    r2 = m2.fit_ols(bx_out, by_out)
                    diffs.append(r1.beta - r2.beta)

            if len(diffs) > 10:
                bootstrap_ci = (
                    float(np.percentile(diffs, 2.5)),
                    float(np.percentile(diffs, 97.5)),
                )

        return RegimeElasticityResult(
            regime_name=regime_name,
            n_in=int(np.sum(valid & mask)),
            n_out=int(np.sum(valid & ~mask)),
            ols_in=ols_in,
            ols_out=ols_out,
            beta_diff=beta_diff,
            bootstrap_ci=bootstrap_ci,
        )

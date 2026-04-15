"""
Chi-square independence test for directional association between
incentive changes and TVL changes.

H0: TVL direction (up/down) is independent of incentive direction (up/down)
H1: TVL direction depends on incentive direction

A high p-value (fail to reject H0) confirms that incentives do not
systematically predict TVL direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class IndependenceResult:
    """Result of chi-square independence test."""

    chi2: float
    p_value: float
    df: int
    contingency: np.ndarray  # 2x2 contingency table
    expected: np.ndarray
    independent: bool  # True if we fail to reject H0 at alpha=0.05

    def summary(self) -> str:
        labels = [
            "           TVL_up  TVL_down",
            f"  Inc_up    {self.contingency[0,0]:5d}     {self.contingency[0,1]:5d}",
            f"  Inc_down  {self.contingency[1,0]:5d}     {self.contingency[1,1]:5d}",
        ]
        verdict = "INDEPENDENT (fail to reject H0)" if self.independent else "DEPENDENT (reject H0)"
        return (
            "Contingency table:\n"
            + "\n".join(labels)
            + f"\n\nchi2 = {self.chi2:.3f}, p = {self.p_value:.4f}, df = {self.df}\n"
            f"Conclusion: {verdict}"
        )


class IndependenceTest:
    """Test whether TVL direction is independent of incentive direction.

    Parameters
    ----------
    alpha : float
        Significance level (default 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
    ) -> IndependenceResult:
        """Run chi-square test on the 2x2 contingency table of directions.

        Parameters
        ----------
        d_incentive : array-like
            First differences of incentive.
        d_tvl : array-like
            First differences of TVL.

        Returns
        -------
        IndependenceResult
        """
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Directions: True = up, False = down
        inc_up = x > 0
        tvl_up = y > 0

        # Build contingency table
        n11 = np.sum(inc_up & tvl_up)
        n12 = np.sum(inc_up & ~tvl_up)
        n21 = np.sum(~inc_up & tvl_up)
        n22 = np.sum(~inc_up & ~tvl_up)

        observed = np.array([[n11, n12], [n21, n22]])

        # Expected under independence
        n = observed.sum()
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / n

        # Chi-square
        chi2 = np.sum((observed - expected) ** 2 / expected)
        df = 1  # (2-1)(2-1)
        p_value = 1 - stats.chi2.cdf(chi2, df=df)

        return IndependenceResult(
            chi2=float(chi2),
            p_value=float(p_value),
            df=df,
            contingency=observed,
            expected=expected,
            independent=p_value > self.alpha,
        )

    def test_counter_examples(
        self,
        incentive: np.ndarray,
        tvl: np.ndarray,
        inc_threshold: float = 0.10,
        tvl_horizon: int = 30,
    ) -> dict:
        """Analyze counter-examples: cases where incentives went up
        but TVL went down within a horizon.

        A counter-example is defined as:
            delta_I_t > threshold * I_{t-1}  AND  S_{t+horizon} < S_t

        Parameters
        ----------
        incentive : array-like
            Incentive level series.
        tvl : array-like
            TVL level series.
        inc_threshold : float
            Minimum fractional incentive increase to qualify (default 10%).
        tvl_horizon : int
            Days to look ahead for TVL decline (default 30).

        Returns
        -------
        dict with keys:
            n_increases: number of qualifying incentive increases
            n_counter: number of counter-examples
            p_counter: fraction that are counter-examples
            p_decline_given_decrease: P(TVL down | incentive down) for comparison
        """
        inc = np.asarray(incentive, dtype=float)
        s = np.asarray(tvl, dtype=float)

        d_inc = np.diff(inc)
        # Fractional change
        frac_change = d_inc / np.where(inc[:-1] > 0, inc[:-1], 1)

        n_increases = 0
        n_counter = 0
        n_decreases = 0
        n_decline_on_decrease = 0

        max_t = len(s) - tvl_horizon - 1
        for t in range(max_t):
            if frac_change[t] > inc_threshold:
                n_increases += 1
                if s[t + tvl_horizon] < s[t]:
                    n_counter += 1
            elif frac_change[t] < -inc_threshold:
                n_decreases += 1
                if s[t + tvl_horizon] < s[t]:
                    n_decline_on_decrease += 1

        return {
            "n_increases": n_increases,
            "n_counter_examples": n_counter,
            "p_counter": n_counter / n_increases if n_increases > 0 else None,
            "n_decreases": n_decreases,
            "p_decline_given_decrease": (
                n_decline_on_decrease / n_decreases if n_decreases > 0 else None
            ),
        }

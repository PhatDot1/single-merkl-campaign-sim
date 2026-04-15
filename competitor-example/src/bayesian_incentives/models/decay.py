"""
AR(1) decay model for TVL persistence analysis.

Model:
    S_t = c + rho * S_{t-1} + eta_t

Derived quantities:
    decay rate   delta = 1 - rho
    half-life    t_{1/2} = -ln(2) / ln(rho)
    equilibrium  mu = c / (1 - rho)

A high rho (close to 1) means TVL is very persistent: shocks take a long
time to decay. A lower rho means faster mean-reversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DecayResult:
    """Estimated AR(1) parameters and derived quantities."""

    rho: float          # persistence coefficient
    c: float            # intercept
    se_rho: float       # standard error of rho
    t_stat: float
    p_value: float
    r_squared: float
    n_obs: int

    @property
    def decay_rate(self) -> float:
        """Daily decay rate delta = 1 - rho."""
        return 1 - self.rho

    @property
    def half_life(self) -> float | None:
        """Half-life in days. None if rho <= 0 or rho >= 1."""
        if 0 < self.rho < 1:
            return -np.log(2) / np.log(self.rho)
        return None

    @property
    def equilibrium(self) -> float | None:
        """Long-run equilibrium mu = c / (1 - rho). None if rho >= 1."""
        if self.rho < 1:
            return self.c / (1 - self.rho)
        return None

    def impulse_response(self, horizons: int = 60) -> np.ndarray:
        """Impulse response function: effect of a unit shock at time 0.

        Returns rho^t for t = 0, 1, ..., horizons-1.
        """
        return self.rho ** np.arange(horizons)

    def summary(self) -> str:
        hl = f"{self.half_life:.1f} days" if self.half_life else "N/A"
        eq = f"${self.equilibrium:,.0f}" if self.equilibrium else "N/A"
        return (
            f"rho = {self.rho:.6f}  (SE={self.se_rho:.6f})\n"
            f"decay rate = {self.decay_rate:.4%}/day\n"
            f"half-life = {hl}\n"
            f"equilibrium = {eq}\n"
            f"R^2 = {self.r_squared:.4%}, n = {self.n_obs}"
        )


class DecayModel:
    """Estimate AR(1) persistence / decay of a target metric (TVL).

    Fits:  S_t = c + rho * S_{t-1} + eta_t
    """

    def __init__(self) -> None:
        self._result: DecayResult | None = None

    def fit(self, series: np.ndarray) -> DecayResult:
        """Fit the AR(1) model via OLS on lagged values.

        Parameters
        ----------
        series : array-like
            Level series (e.g. daily TVL).

        Returns
        -------
        DecayResult
        """
        s = np.asarray(series, dtype=float)
        mask = np.isfinite(s)
        s = s[mask]
        n = len(s) - 1
        if n < 3:
            raise ValueError(f"Need >= 4 observations, got {len(s)}")

        y = s[1:]       # S_t
        x = s[:-1]      # S_{t-1}

        x_mean = x.mean()
        y_mean = y.mean()

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)

        rho = ss_xy / ss_xx if ss_xx > 0 else 0.0
        c = y_mean - rho * x_mean

        y_hat = c + rho * x
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)

        r_squared = 1 - ss_res / ss_yy if ss_yy > 0 else 0.0
        sigma = np.sqrt(ss_res / (n - 2)) if n > 2 else 0.0
        se_rho = sigma / np.sqrt(ss_xx) if ss_xx > 0 else np.inf

        t_stat = rho / se_rho if se_rho > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        self._result = DecayResult(
            rho=rho,
            c=c,
            se_rho=se_rho,
            t_stat=t_stat,
            p_value=p_value,
            r_squared=r_squared,
            n_obs=n,
        )
        return self._result

    @property
    def result(self) -> DecayResult | None:
        return self._result

    def decay_prior_params(self, n_markets: int = 1) -> tuple[float, float]:
        """Compute Beta distribution shape parameters for the decay rate
        prior, calibrated from fitted rho.

        Uses moment matching:
            alpha = mu * kappa
            beta  = (1 - mu) * kappa
        where kappa controls concentration (higher = tighter prior).

        Returns
        -------
        (alpha, beta) : tuple of float
            Shape parameters for Beta(alpha, beta) prior on delta in [0, 1].
        """
        if self._result is None:
            raise RuntimeError("Must fit model before computing prior params.")

        delta = self._result.decay_rate
        delta = np.clip(delta, 0.001, 0.999)

        # Concentration scales with data: more data -> tighter prior
        kappa = 10 + 5 * n_markets
        alpha = delta * kappa
        beta = (1 - delta) * kappa

        return float(alpha), float(beta)

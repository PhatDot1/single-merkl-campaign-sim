"""
Elasticity analysis: OLS regression on first differences with Bayesian
posterior interpretation.

Model:
    delta_TVL_t = alpha + beta * delta_Incentive_t + epsilon_t

The elasticity coefficient beta measures the TVL change per unit change
in incentive spend. A near-zero beta with low R^2 indicates incentives
are inelastic -- they don't meaningfully move the target metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class OLSResult:
    """OLS regression result for a single elasticity estimate."""

    alpha: float
    beta: float
    se_beta: float
    t_stat: float
    p_value: float
    r_squared: float
    n_obs: int
    residuals: np.ndarray
    sigma_hat: float  # residual standard error

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"beta = {self.beta:,.2f}  (SE={self.se_beta:,.2f})\n"
            f"t = {self.t_stat:.3f}, p = {self.p_value:.4f}\n"
            f"R^2 = {self.r_squared:.4%}, n = {self.n_obs}"
        )


@dataclass
class BayesianPosterior:
    """Normal posterior for the elasticity coefficient."""

    mu: float
    sigma: float

    @property
    def ci_95(self) -> tuple[float, float]:
        return (self.mu - 1.96 * self.sigma, self.mu + 1.96 * self.sigma)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n)


class ElasticityModel:
    """Estimate TVL elasticity to incentive changes.

    Runs OLS on first differences:
        d_tvl_t = alpha + beta * d_incentive_t + eps_t

    and computes Bayesian normal-normal posterior for beta.

    Parameters
    ----------
    prior_mu : float
        Prior mean for beta (default 0 = assume no effect).
    prior_sigma : float
        Prior std for beta (default 1e6 = diffuse / uninformative).
    """

    def __init__(
        self, prior_mu: float = 0.0, prior_sigma: float = 1e6
    ) -> None:
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self._ols: OLSResult | None = None
        self._posterior: BayesianPosterior | None = None

    # ── OLS estimation ──────────────────────────────────────────────────

    def fit_ols(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
    ) -> OLSResult:
        """Fit OLS regression: d_tvl = alpha + beta * d_incentive.

        Parameters
        ----------
        d_incentive : array-like
            First differences of incentive series.
        d_tvl : array-like
            First differences of TVL series.

        Returns
        -------
        OLSResult
        """
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)

        # Drop NaN pairs
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            raise ValueError(f"Need >= 3 valid observations, got {n}")

        x_mean, y_mean = x.mean(), y.mean()

        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)

        beta = ss_xy / ss_xx if ss_xx > 0 else 0.0
        alpha = y_mean - beta * x_mean

        y_hat = alpha + beta * x
        residuals = y - y_hat
        ss_res = np.sum(residuals ** 2)

        r_squared = 1 - ss_res / ss_yy if ss_yy > 0 else 0.0
        sigma_hat = np.sqrt(ss_res / (n - 2))
        se_beta = sigma_hat / np.sqrt(ss_xx) if ss_xx > 0 else np.inf

        t_stat = beta / se_beta if se_beta > 0 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))

        self._ols = OLSResult(
            alpha=alpha,
            beta=beta,
            se_beta=se_beta,
            t_stat=t_stat,
            p_value=p_value,
            r_squared=r_squared,
            n_obs=n,
            residuals=residuals,
            sigma_hat=sigma_hat,
        )
        return self._ols

    # ── Bayesian posterior ──────────────────────────────────────────────

    def fit_posterior(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
        sigma_eps: Optional[float] = None,
    ) -> BayesianPosterior:
        """Compute conjugate normal-normal posterior for beta.

        posterior_var  = (1/prior_var + sum(dI^2)/sigma_eps^2)^{-1}
        posterior_mean = posterior_var * (prior_mu/prior_var + sum(dI*dS)/sigma_eps^2)

        Parameters
        ----------
        d_incentive, d_tvl : array-like
            First differences.
        sigma_eps : float, optional
            Residual std. If None, estimated from OLS fit.

        Returns
        -------
        BayesianPosterior
        """
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if sigma_eps is None:
            if self._ols is None:
                self.fit_ols(x, y)
            sigma_eps = self._ols.sigma_hat

        prior_var = self.prior_sigma ** 2
        eps_var = sigma_eps ** 2

        sum_x2 = np.sum(x ** 2)
        sum_xy = np.sum(x * y)

        post_var = 1.0 / (1.0 / prior_var + sum_x2 / eps_var)
        post_mu = post_var * (self.prior_mu / prior_var + sum_xy / eps_var)

        self._posterior = BayesianPosterior(mu=post_mu, sigma=np.sqrt(post_var))
        return self._posterior

    # ── Lagged elasticity ───────────────────────────────────────────────

    def fit_lagged(
        self,
        d_incentive: np.ndarray,
        tvl: np.ndarray,
        lags: list[int] | None = None,
    ) -> dict[int, OLSResult]:
        """Estimate elasticity at multiple lags.

        For each lag k, regresses delta_TVL_{t+k} on delta_I_t.

        Parameters
        ----------
        d_incentive : array-like
            First differences of incentive (length T).
        tvl : array-like
            TVL level series (length T+1 or same as d_incentive).
        lags : list of int
            Lag values in days (default [1, 7, 14, 30]).

        Returns
        -------
        dict mapping lag -> OLSResult
        """
        if lags is None:
            lags = [1, 7, 14, 30]

        x = np.asarray(d_incentive, dtype=float)
        s = np.asarray(tvl, dtype=float)

        results: dict[int, OLSResult] = {}
        for k in lags:
            if k >= len(s) - 1:
                continue
            d_tvl_lagged = np.diff(s, n=1)
            # Align: d_incentive[t] vs d_tvl[t+k]
            max_t = min(len(x), len(d_tvl_lagged) - k)
            if max_t < 3:
                continue
            xi = x[:max_t]
            yi = d_tvl_lagged[k : k + max_t]
            model = ElasticityModel(
                prior_mu=self.prior_mu, prior_sigma=self.prior_sigma
            )
            results[k] = model.fit_ols(xi, yi)

        return results

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def ols(self) -> OLSResult | None:
        return self._ols

    @property
    def posterior(self) -> BayesianPosterior | None:
        return self._posterior

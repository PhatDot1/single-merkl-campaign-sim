"""
Historical incentive effectiveness analysis.

Adapted from the Bayesian Incentives Optimizer framework and integrated
with our DeFiLlama PoolHistory data model and IRMParams.

Models
------
ElasticityModel
    OLS first-difference regression + Bayesian conjugate posterior + lagged analysis.
    Estimates how much TVL changes per $1 change in daily incentive spend.
    A low R² / non-significant beta → incentives are inelastic at this venue.

DecayModel
    AR(1) model: S_t = c + rho * S_{t-1} + eta_t
    Derives half-life (days for a TVL shock to decay) and long-run equilibrium.
    Calibrates organic_tvl_floor_fraction and recommended simulation horizon.

IndependenceTest
    Chi-square test: is TVL direction independent of incentive direction?
    Counter-example analysis: % of incentive increases followed by TVL decline.

RegimeAnalyzer
    Volatility-regime conditional elasticity with bootstrap CIs.
    Are incentives more/less effective during high-volatility periods?

ProfitabilityAnalyzer
    NII = (borrow_volume × r_b - TVL × r_s) / 365
    Breakeven utilisation and three-way market health classification.

AnalysisReport
    Container for all results on a single venue/pool.

generate_report(history)
    Full pipeline: PoolHistory → AnalysisReport.

Usage
-----
    from campaign.historical import fetch_pool_history
    from campaign.analysis import generate_report

    history = fetch_pool_history("747c1d2a-c668-4682-b9f9-296708a3dd90", days=180)
    report = generate_report(history)
    print(report.summary())

    # Calibrate simulation horizon from AR(1) half-life
    print("Recommended horizon:", report.recommended_horizon_days(), "days")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats

from .historical import PoolHistory
from .state import IRMParams


# ============================================================================
# ELASTICITY MODEL
# ============================================================================


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
        sig = " *" if self.significant else ""
        return (
            f"beta = {self.beta:,.0f}  (SE={self.se_beta:,.0f}){sig}\n"
            f"t = {self.t_stat:.3f},  p = {self.p_value:.4f}\n"
            f"R² = {self.r_squared:.4%},  n = {self.n_obs}"
        )


@dataclass
class BayesianPosterior:
    """Normal-normal conjugate posterior for the elasticity coefficient beta."""

    mu: float
    sigma: float

    @property
    def ci_95(self) -> tuple[float, float]:
        return (self.mu - 1.96 * self.sigma, self.mu + 1.96 * self.sigma)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n)


class ElasticityModel:
    """
    Estimate TVL elasticity to incentive changes via OLS on first differences.

    Model:
        delta_TVL_t = alpha + beta * delta_Incentive_t + epsilon_t

    beta is the key output: TVL change ($) per $1 change in daily incentive spend.
    A beta near zero with low R² indicates incentives are inelastic at this venue.

    Bayesian mode: conjugate normal-normal update gives a posterior density for beta.
    With diffuse priors (default sigma=1e6), the posterior mean converges to OLS.

    Parameters
    ----------
    prior_mu : float
        Prior mean for beta (default 0 = assume no effect).
    prior_sigma : float
        Prior std for beta (default 1e6 = diffuse / uninformative).
    """

    def __init__(self, prior_mu: float = 0.0, prior_sigma: float = 1e6) -> None:
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self._ols: Optional[OLSResult] = None
        self._posterior: Optional[BayesianPosterior] = None

    def fit_ols(self, d_incentive: np.ndarray, d_tvl: np.ndarray) -> OLSResult:
        """Fit OLS: d_tvl = alpha + beta * d_incentive."""
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            raise ValueError(f"Need >= 3 valid observations, got {n}")

        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = float(np.sum((x - x_mean) ** 2))
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        ss_yy = float(np.sum((y - y_mean) ** 2))

        beta = ss_xy / ss_xx if ss_xx > 0 else 0.0
        alpha = y_mean - beta * x_mean

        y_hat = alpha + beta * x
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))

        r_squared = 1.0 - ss_res / ss_yy if ss_yy > 0 else 0.0
        sigma_hat = float(np.sqrt(ss_res / (n - 2))) if n > 2 else 0.0
        se_beta = sigma_hat / float(np.sqrt(ss_xx)) if ss_xx > 0 else np.inf

        t_stat = beta / se_beta if np.isfinite(se_beta) and se_beta > 0 else 0.0
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 2, 1))))

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

    def fit_posterior(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
        sigma_eps: Optional[float] = None,
    ) -> BayesianPosterior:
        """
        Compute conjugate normal-normal posterior for beta.

        posterior_var  = (1/prior_var + sum(dI²)/sigma²)^{-1}
        posterior_mean = posterior_var * (prior_mu/prior_var + sum(dI·dS)/sigma²)
        """
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if sigma_eps is None:
            if self._ols is None:
                self.fit_ols(x, y)
            sigma_eps = self._ols.sigma_hat  # type: ignore[union-attr]

        prior_var = self.prior_sigma ** 2
        eps_var = max(float(sigma_eps) ** 2, 1e-30)

        sum_x2 = float(np.sum(x ** 2))
        sum_xy = float(np.sum(x * y))

        post_var = 1.0 / (1.0 / prior_var + sum_x2 / eps_var)
        post_mu = post_var * (self.prior_mu / prior_var + sum_xy / eps_var)

        self._posterior = BayesianPosterior(mu=post_mu, sigma=float(np.sqrt(post_var)))
        return self._posterior

    def fit_lagged(
        self,
        d_incentive: np.ndarray,
        tvl: np.ndarray,
        lags: Optional[list[int]] = None,
    ) -> dict[int, OLSResult]:
        """
        Test whether TVL responds to incentive changes with a delay.

        For each lag k, fits:
            delta_TVL(t+k) = alpha_k + beta_k * delta_Incentive_t + epsilon

        A higher R² at lag 14 than lag 1 suggests depositors take ~2 weeks to respond.
        This informs the optimal simulation dt_days and horizon_days.

        Parameters
        ----------
        d_incentive : array-like
            First differences of incentive series.
        tvl : array-like
            TVL level series (not first differences).
        lags : list of int, optional
            Lags to test (default [1, 7, 14, 30]).
        """
        lags = lags or [1, 7, 14, 30]
        results: dict[int, OLSResult] = {}
        x = np.asarray(d_incentive, dtype=float)
        s = np.asarray(tvl, dtype=float)

        for k in lags:
            if k + 1 >= len(s):
                continue
            d_tvl_k = np.diff(s[k:])
            n_align = min(len(x), len(d_tvl_k))
            if n_align < 3:
                continue
            try:
                m = ElasticityModel(prior_mu=self.prior_mu, prior_sigma=self.prior_sigma)
                results[k] = m.fit_ols(x[:n_align], d_tvl_k[:n_align])
            except ValueError:
                continue

        return results

    @property
    def ols(self) -> Optional[OLSResult]:
        return self._ols

    @property
    def posterior(self) -> Optional[BayesianPosterior]:
        return self._posterior


# ============================================================================
# DECAY MODEL (AR(1))
# ============================================================================


@dataclass
class DecayResult:
    """Fitted AR(1) parameters and derived quantities."""

    rho: float      # Persistence coefficient (0 < rho < 1 for stationary series)
    c: float        # Intercept
    se_rho: float
    t_stat: float
    p_value: float
    r_squared: float
    n_obs: int

    @property
    def decay_rate(self) -> float:
        """Daily mean-reversion rate: delta = 1 - rho."""
        return 1.0 - self.rho

    @property
    def half_life(self) -> Optional[float]:
        """Days for a TVL shock to decay by 50%. None if rho ≤ 0 or rho ≥ 1."""
        if 0 < self.rho < 1:
            return float(-np.log(2) / np.log(self.rho))
        return None

    @property
    def equilibrium(self) -> Optional[float]:
        """Long-run equilibrium TVL: mu = c / (1 - rho). None if rho ≥ 1."""
        if self.rho < 1.0:
            return self.c / (1.0 - self.rho)
        return None

    def impulse_response(self, horizons: int = 60) -> np.ndarray:
        """Fraction of a unit shock remaining after t days: rho^t."""
        return self.rho ** np.arange(horizons)

    def summary(self) -> str:
        hl = f"{self.half_life:.1f} days" if self.half_life else "N/A"
        eq = f"${self.equilibrium / 1e6:,.1f}M" if self.equilibrium else "N/A"
        return (
            f"rho = {self.rho:.6f}  (SE={self.se_rho:.6f})\n"
            f"decay rate = {self.decay_rate:.4%}/day\n"
            f"half-life = {hl}\n"
            f"equilibrium TVL = {eq}\n"
            f"R² = {self.r_squared:.4%},  n = {self.n_obs}"
        )


class DecayModel:
    """
    Fit AR(1) mean-reversion model: S_t = c + rho * S_{t-1} + eta_t

    Derived quantities:
    - decay_rate = 1 - rho  (daily fraction converging to equilibrium)
    - half_life = -ln(2) / ln(rho)  (days for shock to decay by half)
    - equilibrium = c / (1 - rho)  (long-run TVL without shocks)

    high rho (≈1) → very sticky TVL, slow mean-reversion, long half-life.
    low rho (≈0) → mercenary TVL, fast exit after incentives stop.

    The equilibrium TVL calibrates organic_tvl_floor_fraction.
    The half-life calibrates the simulation horizon_days.
    """

    def __init__(self) -> None:
        self._result: Optional[DecayResult] = None

    def fit(self, series: np.ndarray) -> DecayResult:
        """Fit AR(1) via OLS on lagged values."""
        s = np.asarray(series, dtype=float)
        s = s[np.isfinite(s)]
        n = len(s) - 1
        if n < 3:
            raise ValueError(f"Need >= 4 observations, got {len(s)}")

        y = s[1:]
        x = s[:-1]

        x_mean, y_mean = x.mean(), y.mean()
        ss_xx = float(np.sum((x - x_mean) ** 2))
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        ss_yy = float(np.sum((y - y_mean) ** 2))

        rho = ss_xy / ss_xx if ss_xx > 0 else 0.0
        c = float(y_mean - rho * x_mean)

        y_hat = c + rho * x
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))

        r_squared = float(1.0 - ss_res / ss_yy) if ss_yy > 0 else 0.0
        sigma = float(np.sqrt(ss_res / (n - 2))) if n > 2 else 0.0
        se_rho = sigma / float(np.sqrt(ss_xx)) if ss_xx > 0 else np.inf

        t_stat = rho / se_rho if np.isfinite(se_rho) and se_rho > 0 else 0.0
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 2, 1))))

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

    def decay_prior_params(self, n_markets: int = 1) -> tuple[float, float]:
        """
        Beta distribution shape params for the decay rate prior.

        Calibrated from the fitted rho.  Useful for seeding Monte Carlo
        sampling of decay uncertainty in forward simulations.

        Returns (alpha, beta) for Beta(alpha, beta) prior on delta in [0,1].
        """
        if self._result is None:
            raise RuntimeError("Must fit model before computing prior params.")
        delta = float(np.clip(self._result.decay_rate, 0.001, 0.999))
        kappa = 10.0 + 5.0 * n_markets
        return float(delta * kappa), float((1.0 - delta) * kappa)

    @property
    def result(self) -> Optional[DecayResult]:
        return self._result


# ============================================================================
# INDEPENDENCE TEST
# ============================================================================


@dataclass
class IndependenceResult:
    """Chi-square test result: are TVL and incentive directions associated?"""

    chi2: float
    p_value: float
    df: int
    contingency: np.ndarray   # 2×2 table: [Inc↑/TVL↑, Inc↑/TVL↓, Inc↓/TVL↑, Inc↓/TVL↓]
    expected: np.ndarray
    independent: bool          # True → fail to reject H₀ (incentives ≠ TVL direction)

    def summary(self) -> str:
        c = self.contingency
        verdict = (
            "INDEPENDENT (fail to reject H₀ — incentives don't predict TVL direction)"
            if self.independent
            else "DEPENDENT (reject H₀ — incentive changes correlate with TVL direction)"
        )
        return (
            f"Contingency table:\n"
            f"             TVL↑   TVL↓\n"
            f"  Incentive↑  {c[0, 0]:5d}  {c[0, 1]:5d}\n"
            f"  Incentive↓  {c[1, 0]:5d}  {c[1, 1]:5d}\n\n"
            f"chi² = {self.chi2:.3f},  p = {self.p_value:.4f},  df = {self.df}\n"
            f"Conclusion: {verdict}"
        )


class IndependenceTest:
    """
    Chi-square test for directional independence between incentive and TVL changes.

    H₀: TVL direction (↑/↓) is independent of incentive direction (↑/↓).
    A high p-value → fail to reject H₀ → incentives don't systematically predict TVL.

    Also provides counter-example analysis: what fraction of incentive increases
    were followed by TVL decline within a configurable horizon?
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def test(self, d_incentive: np.ndarray, d_tvl: np.ndarray) -> IndependenceResult:
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        n11 = int(np.sum((x > 0) & (y > 0)))
        n12 = int(np.sum((x > 0) & (y <= 0)))
        n21 = int(np.sum((x <= 0) & (y > 0)))
        n22 = int(np.sum((x <= 0) & (y <= 0)))

        observed = np.array([[n11, n12], [n21, n22]])
        n = observed.sum()
        if n == 0:
            return IndependenceResult(0.0, 1.0, 1, observed, observed.astype(float), True)

        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        expected = (row_sums * col_sums).astype(float) / n

        with np.errstate(divide="ignore", invalid="ignore"):
            chi2_terms = np.where(
                expected > 0, (observed - expected) ** 2 / expected, 0.0
            )
        chi2 = float(np.sum(chi2_terms))
        p_value = float(1.0 - stats.chi2.cdf(chi2, df=1))

        return IndependenceResult(
            chi2=chi2,
            p_value=p_value,
            df=1,
            contingency=observed,
            expected=expected,
            independent=p_value > self.alpha,
        )

    def counter_examples(
        self,
        incentive: np.ndarray,
        tvl: np.ndarray,
        inc_threshold: float = 0.10,
        tvl_horizon: int = 30,
    ) -> dict:
        """
        Find counter-examples: incentive increased ≥ inc_threshold (fractionally)
        but TVL fell within tvl_horizon days.

        A high counter-example rate is a compelling stakeholder argument against
        continuing incentive spend at this venue.

        Parameters
        ----------
        incentive : array-like
            Daily incentive spend level series.
        tvl : array-like
            Daily TVL level series.
        inc_threshold : float
            Minimum fractional incentive increase to qualify (default 10%).
        tvl_horizon : int
            Look-ahead window for TVL decline (default 30 days).
        """
        inc = np.asarray(incentive, dtype=float)
        s = np.asarray(tvl, dtype=float)
        inc_prev = np.where(inc[:-1] > 0, inc[:-1], 1.0)
        frac_change = np.diff(inc) / inc_prev

        n_incr = n_counter = n_decr = n_decl_on_decr = 0
        max_t = len(s) - tvl_horizon - 1

        for t in range(min(max_t, len(frac_change))):
            if frac_change[t] > inc_threshold:
                n_incr += 1
                if s[t + tvl_horizon] < s[t]:
                    n_counter += 1
            elif frac_change[t] < -inc_threshold:
                n_decr += 1
                if s[t + tvl_horizon] < s[t]:
                    n_decl_on_decr += 1

        return {
            "n_increases": n_incr,
            "n_counter_examples": n_counter,
            "p_counter": n_counter / n_incr if n_incr > 0 else None,
            "n_decreases": n_decr,
            "p_decline_given_decrease": n_decl_on_decr / n_decr if n_decr > 0 else None,
        }


# ============================================================================
# REGIME ANALYZER
# ============================================================================


@dataclass
class RegimeElasticityResult:
    """Elasticity estimated in-regime vs out-of-regime, with bootstrap CI."""

    regime_name: str
    n_in: int
    n_out: int
    ols_in: Optional[OLSResult]
    ols_out: Optional[OLSResult]
    beta_diff: Optional[float] = None
    bootstrap_ci: Optional[tuple[float, float]] = None

    def summary(self) -> str:
        def _fmt(r: Optional[OLSResult], n: int) -> str:
            if r is None or n < 3:
                return f"(n={n} — insufficient data)"
            return f"(n={n})  beta={r.beta:,.0f}  R²={r.r_squared:.3%}  p={r.p_value:.3f}"

        lines = [
            f"Regime: {self.regime_name}",
            f"  In-regime   {_fmt(self.ols_in, self.n_in)}",
            f"  Out-regime  {_fmt(self.ols_out, self.n_out)}",
        ]
        if self.beta_diff is not None:
            lines.append(f"  beta_diff = {self.beta_diff:,.0f}")
        if self.bootstrap_ci:
            lo, hi = self.bootstrap_ci
            lines.append(f"  95% bootstrap CI: [{lo:,.0f}, {hi:,.0f}]")
        return "\n".join(lines)


class RegimeAnalyzer:
    """
    Compute volatility-regime conditional elasticities with bootstrap CIs.

    Volatility regimes are based on realized TVL return volatility (rolling std).
    High-vol vs low-vol elasticity difference reveals whether incentives work
    differently during market stress — informs risk-adjusted campaign timing.

    Parameters
    ----------
    vol_high_q : float
        Quantile threshold for high-volatility regime (default 0.75).
    vol_extreme_q : float
        Quantile threshold for extreme-volatility regime (default 0.90).
    bootstrap_n : int
        Bootstrap resamples for CI (default 500; use 200 for speed in backtesting).
    rng_seed : int
        Random seed.
    """

    def __init__(
        self,
        vol_high_q: float = 0.75,
        vol_extreme_q: float = 0.90,
        bootstrap_n: int = 500,
        rng_seed: int = 42,
    ) -> None:
        self.vol_high_q = vol_high_q
        self.vol_extreme_q = vol_extreme_q
        self.bootstrap_n = bootstrap_n
        self.rng_seed = rng_seed

    def classify_volatility(
        self, tvl: np.ndarray, window: int = 7
    ) -> dict[str, np.ndarray]:
        """
        Classify each observation into volatility regimes.

        Uses rolling std of log-returns aligned to first-difference length.

        Returns dict with keys "high_vol", "extreme_vol", "vol".
        The arrays have length len(tvl) - 1 (aligned to first differences).
        """
        log_tvl = np.log(np.maximum(np.asarray(tvl, dtype=float), 1.0))
        returns = np.diff(log_tvl)

        # Rolling std of returns (causal window)
        n = len(returns)
        vol = np.array([
            float(np.std(returns[max(0, i - window + 1): i + 1]))
            for i in range(n)
        ])

        q_high = float(np.nanquantile(vol, self.vol_high_q))
        q_ext = float(np.nanquantile(vol, self.vol_extreme_q))
        return {
            "high_vol": vol > q_high,
            "extreme_vol": vol > q_ext,
            "vol": vol,
        }

    def fit_conditional(
        self,
        d_incentive: np.ndarray,
        d_tvl: np.ndarray,
        regime_mask: np.ndarray,
        regime_name: str = "regime",
    ) -> RegimeElasticityResult:
        """Fit OLS separately in- and out-of-regime, bootstrap the difference."""
        rng = np.random.default_rng(self.rng_seed)
        x = np.asarray(d_incentive, dtype=float)
        y = np.asarray(d_tvl, dtype=float)
        mask = np.asarray(regime_mask, dtype=bool)
        min_len = min(len(x), len(y), len(mask))
        x, y, mask = x[:min_len], y[:min_len], mask[:min_len]

        valid = np.isfinite(x) & np.isfinite(y)
        x_in, y_in = x[valid & mask], y[valid & mask]
        x_out, y_out = x[valid & ~mask], y[valid & ~mask]

        ols_in = ols_out = None
        if len(x_in) >= 3:
            ols_in = ElasticityModel().fit_ols(x_in, y_in)
        if len(x_out) >= 3:
            ols_out = ElasticityModel().fit_ols(x_out, y_out)

        beta_diff: Optional[float] = None
        bootstrap_ci: Optional[tuple[float, float]] = None

        if ols_in is not None and ols_out is not None:
            beta_diff = ols_in.beta - ols_out.beta
            diffs: list[float] = []
            indices = np.where(valid)[0]

            for _ in range(self.bootstrap_n):
                boot_idx = rng.choice(indices, size=len(indices), replace=True)
                bm = mask[boot_idx]
                bx, by = x[boot_idx], y[boot_idx]
                if bm.sum() < 3 or (~bm).sum() < 3:
                    continue
                try:
                    r1 = ElasticityModel().fit_ols(bx[bm], by[bm])
                    r2 = ElasticityModel().fit_ols(bx[~bm], by[~bm])
                    diffs.append(r1.beta - r2.beta)
                except ValueError:
                    continue

            if len(diffs) >= 10:
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


# ============================================================================
# PROFITABILITY ANALYSIS
# ============================================================================


class MarketStatus(str, Enum):
    """Three-way market health classification based on breakeven utilization."""

    WELL_UTILIZED = "WELL_UTILIZED"
    AT_BREAKEVEN = "AT_BREAKEVEN"
    UNDER_UTILIZED = "UNDER_UTILIZED"

    @property
    def label(self) -> str:
        return {
            "WELL_UTILIZED": "Well-Utilized",
            "AT_BREAKEVEN": "At Breakeven",
            "UNDER_UTILIZED": "Under-Utilized",
        }[self.value]

    @property
    def recommendation(self) -> str:
        return {
            "WELL_UTILIZED": (
                "Already profitable — incentives are a pure cost. "
                "Consider reducing or pausing supply incentives."
            ),
            "AT_BREAKEVEN": (
                "Marginal profitability. Supply incentives have low ROI. "
                "Small borrow incentives may improve utilization."
            ),
            "UNDER_UTILIZED": (
                "Losing on interest spread. "
                "Only borrow incentives can help by pulling utilization higher."
            ),
        }[self.value]

    @property
    def color(self) -> str:
        return {
            "WELL_UTILIZED": "🟢",
            "AT_BREAKEVEN": "🟡",
            "UNDER_UTILIZED": "🔴",
        }[self.value]


@dataclass
class ProfitabilitySnapshot:
    """Point-in-time NII and health assessment for a lending venue."""

    tvl: float
    utilization: float
    borrow_apr: float
    supply_apr: float
    daily_nii: float          # Net interest income per day ($)
    daily_incentives: float   # Daily incentive spend ($)
    daily_profit: float       # NII - incentives
    breakeven_utilization: float
    status: MarketStatus

    def summary(self) -> str:
        return (
            f"TVL: ${self.tvl / 1e6:.1f}M  |  Utilization: {self.utilization:.2%}\n"
            f"Borrow APR: {self.borrow_apr:.2%}  |  Supply APR: {self.supply_apr:.2%}\n"
            f"Daily NII: ${self.daily_nii:,.0f}\n"
            f"Daily Incentives: ${self.daily_incentives:,.0f}\n"
            f"Daily Profit (NII − Incentives): ${self.daily_profit:,.0f}\n"
            f"Breakeven Utilization: {self.breakeven_utilization:.2%}\n"
            f"Status: {self.status.color} {self.status.label}\n"
            f"Recommendation: {self.status.recommendation}"
        )


class ProfitabilityAnalyzer:
    """
    Compute NII, breakeven utilization, and market health.

    Can operate in two modes:
    1. IRM mode — uses IRMParams + utilization to derive borrow/supply APRs.
    2. Data mode — borrow_apr and supply_apr passed directly.

    NII formula:
        NII = (borrow_volume × r_b - TVL × r_s) / 365  [daily]
        borrow_volume = TVL × utilization

    Breakeven utilization U_BE:
        U_BE = r_s / r_b  (NII = 0 at this utilization)

    Market classification (breakeven_band = 0.05 by default):
        WELL_UTILIZED    → utilization > U_BE + 0.05
        AT_BREAKEVEN     → |utilization - U_BE| ≤ 0.05
        UNDER_UTILIZED   → utilization < U_BE - 0.05
    """

    def __init__(
        self,
        irm: Optional[IRMParams] = None,
        breakeven_band: float = 0.05,
    ) -> None:
        self.irm = irm
        self.breakeven_band = breakeven_band

    def _classify(self, utilization: float, breakeven_u: float) -> MarketStatus:
        if utilization > breakeven_u + self.breakeven_band:
            return MarketStatus.WELL_UTILIZED
        elif abs(utilization - breakeven_u) <= self.breakeven_band:
            return MarketStatus.AT_BREAKEVEN
        return MarketStatus.UNDER_UTILIZED

    def _irm_rates(self, utilization: float) -> tuple[float, float]:
        """Compute (borrow_apr, supply_apr) from IRMParams at given utilization."""
        irm = self.irm
        assert irm is not None
        u = float(utilization)
        if u <= irm.optimal_util:
            r_b = irm.base_rate + (u / max(irm.optimal_util, 1e-9)) * irm.slope1
        else:
            excess = (u - irm.optimal_util) / max(1.0 - irm.optimal_util, 1e-9)
            r_b = irm.base_rate + irm.slope1 + excess * irm.slope2
        r_s = r_b * u * (1.0 - irm.reserve_factor)
        return float(r_b), float(r_s)

    def snapshot(
        self,
        tvl: float,
        utilization: float,
        daily_incentives: float,
        borrow_apr: Optional[float] = None,
        supply_apr: Optional[float] = None,
    ) -> ProfitabilitySnapshot:
        """
        Compute single-point profitability snapshot.

        If borrow_apr / supply_apr are None, they are derived from self.irm.
        """
        if (borrow_apr is None or supply_apr is None) and self.irm is not None:
            borrow_apr, supply_apr = self._irm_rates(utilization)
        if borrow_apr is None or supply_apr is None:
            raise ValueError(
                "Provide borrow_apr and supply_apr explicitly, or supply an IRMParams."
            )

        borrow_vol = tvl * utilization
        daily_nii = (borrow_vol * borrow_apr - tvl * supply_apr) / 365.0
        daily_profit = daily_nii - daily_incentives
        breakeven_u = supply_apr / borrow_apr if borrow_apr > 0 else 0.0

        return ProfitabilitySnapshot(
            tvl=tvl,
            utilization=utilization,
            borrow_apr=float(borrow_apr),
            supply_apr=float(supply_apr),
            daily_nii=daily_nii,
            daily_incentives=daily_incentives,
            daily_profit=daily_profit,
            breakeven_utilization=breakeven_u,
            status=self._classify(utilization, breakeven_u),
        )


# ============================================================================
# ANALYSIS REPORT
# ============================================================================


@dataclass
class AnalysisReport:
    """All analysis results for a single venue/pool."""

    venue_id: str = ""

    # Elasticity
    elasticity_total: Optional[OLSResult] = None
    posterior_total: Optional[BayesianPosterior] = None
    elasticity_lagged: dict[int, OLSResult] = field(default_factory=dict)
    elasticity_base_apy: Optional[OLSResult] = None  # TVL ~ base APY changes

    # Decay
    decay: Optional[DecayResult] = None

    # Independence
    independence: Optional[IndependenceResult] = None
    counter_examples: Optional[dict] = None

    # Regimes
    regime_high_vol: Optional[RegimeElasticityResult] = None
    regime_extreme_vol: Optional[RegimeElasticityResult] = None

    # Profitability
    profitability_current: Optional[ProfitabilitySnapshot] = None

    # Metadata
    n_observations: int = 0
    date_range: tuple[str, str] = ("", "")

    def recommended_horizon_days(self) -> Optional[int]:
        """
        Recommend a simulation horizon from the half-life.

        Recommendation: horizon ≥ 2 × half-life to capture at least two
        mean-reversion cycles. Floors at 28 days.
        """
        if self.decay is None or self.decay.half_life is None:
            return None
        return int(max(28, 2 * self.decay.half_life))

    def organic_floor_fraction(self, current_tvl: float) -> Optional[float]:
        """
        Recommended organic_tvl_floor_fraction for RetailDepositorConfig.

        organic_floor = equilibrium_tvl / current_tvl.
        This is the sticky fraction that should remain even without incentives.
        """
        if self.decay is None or self.decay.equilibrium is None:
            return None
        if current_tvl <= 0:
            return None
        return float(np.clip(self.decay.equilibrium / current_tvl, 0.0, 1.0))

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"ANALYSIS REPORT: {self.venue_id or 'venue'}",
            "=" * 60,
            f"Date range:     {self.date_range[0]} → {self.date_range[1]}",
            f"Observations:   {self.n_observations}",
            "",
        ]

        lines.append("── Elasticity (TVL ~ Incentive, first differences) ──")
        if self.elasticity_total:
            lines.append(f"  Total:\n    {self.elasticity_total.summary()}")
        if self.posterior_total:
            ci = self.posterior_total.ci_95
            lines.append(
                f"  Posterior: N({self.posterior_total.mu:,.0f}, "
                f"{self.posterior_total.sigma:,.0f})"
                f"  95% CI=[{ci[0]:,.0f}, {ci[1]:,.0f}]"
            )
        if self.elasticity_lagged:
            lines.append("  Lagged R²:")
            for lag, r in sorted(self.elasticity_lagged.items()):
                sig = " *" if r.significant else ""
                lines.append(f"    lag={lag:2d}d:  R²={r.r_squared:.3%}{sig}")
        lines.append("")

        if self.decay:
            lines.append("── Decay / Persistence ──")
            lines.append("  " + self.decay.summary().replace("\n", "\n  "))
            if self.decay.half_life:
                lines.append(
                    f"  → Recommended simulation horizon: "
                    f"{self.recommended_horizon_days()} days"
                )
            lines.append("")

        if self.independence:
            lines.append("── Independence Test ──")
            lines.append("  " + self.independence.summary().replace("\n", "\n  "))
            lines.append("")

        if self.counter_examples:
            ce = self.counter_examples
            lines.append("── Counter-Examples (incentive ↑ but TVL ↓ in 30d) ──")
            if ce.get("p_counter") is not None:
                lines.append(
                    f"  {ce['n_counter_examples']}/{ce['n_increases']} = "
                    f"{ce['p_counter']:.0%} of incentive increases were followed "
                    f"by TVL decline within 30 days"
                )
            if ce.get("p_decline_given_decrease") is not None:
                lines.append(
                    f"  Baseline (incentive ↓ → TVL ↓): "
                    f"{ce['p_decline_given_decrease']:.0%}"
                )
            lines.append("")

        for r in [self.regime_high_vol, self.regime_extreme_vol]:
            if r is not None:
                lines.append(f"── Regime: {r.regime_name} ──")
                lines.append("  " + r.summary().replace("\n", "\n  "))
                lines.append("")

        if self.profitability_current:
            lines.append("── Profitability ──")
            lines.append(
                "  " + self.profitability_current.summary().replace("\n", "\n  ")
            )

        return "\n".join(lines)


# ============================================================================
# PIPELINE
# ============================================================================


def generate_report(
    history: PoolHistory,
    irm: Optional[IRMParams] = None,
    daily_incentives_usd: Optional[float] = None,
    current_utilization: float = 0.8,
    prior_mu: float = 0.0,
    prior_sigma: float = 1e6,
    bootstrap_n: int = 500,
) -> AnalysisReport:
    """
    Run the full analysis pipeline on a DeFiLlama PoolHistory.

    Incentive spend proxy: TVL × apy_reward / 365  (daily cost implied by reward APY).
    This is the best available approximation when exact daily spend data is unavailable.

    Parameters
    ----------
    history : PoolHistory
        Historical data from fetch_pool_history().
    irm : IRMParams, optional
        If provided, used for profitability analysis (overrides DeFiLlama APR data).
    daily_incentives_usd : float, optional
        Known daily incentive spend.  If None, inferred from apy_reward.
    current_utilization : float
        Utilization ratio for profitability snapshot (0.0–1.0).
    prior_mu, prior_sigma : float
        Bayesian prior parameters for the elasticity beta.
    bootstrap_n : int
        Bootstrap iterations for regime analysis.

    Returns
    -------
    AnalysisReport
    """
    report = AnalysisReport(
        venue_id=(
            f"{history.project}/{history.symbol}"
            if history.project
            else history.pool_id
        ),
        n_observations=len(history.points),
    )

    if len(history.points) < 4:
        return report

    pts = history.points
    report.date_range = (pts[0].timestamp, pts[-1].timestamp)

    tvl = history.tvl_array
    apy_reward = history.apy_reward_array
    apy_base = history.apy_base_array

    # Daily incentive spend proxy (USD)
    incentive_usd = tvl * apy_reward / 365.0
    d_tvl = np.diff(tvl)
    d_incentive = np.diff(incentive_usd)

    # ── Elasticity: incentive → TVL ─────────────────────────────────────
    if len(d_tvl) >= 3:
        try:
            em = ElasticityModel(prior_mu=prior_mu, prior_sigma=prior_sigma)
            report.elasticity_total = em.fit_ols(d_incentive, d_tvl)
            report.posterior_total = em.fit_posterior(d_incentive, d_tvl)
            report.elasticity_lagged = em.fit_lagged(d_incentive, tvl)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass

    # Elasticity: base APY changes → TVL (organic sensitivity)
    d_apy_base = np.diff(apy_base)
    if len(d_apy_base) >= 3 and np.any(d_apy_base != 0):
        try:
            em_base = ElasticityModel(prior_mu=prior_mu, prior_sigma=prior_sigma)
            report.elasticity_base_apy = em_base.fit_ols(d_apy_base, d_tvl)
        except (ValueError, ZeroDivisionError):
            pass

    # ── Decay ────────────────────────────────────────────────────────────
    if len(tvl) >= 4:
        try:
            dm = DecayModel()
            report.decay = dm.fit(tvl)
        except (ValueError, ZeroDivisionError):
            pass

    # ── Independence test ────────────────────────────────────────────────
    if len(d_tvl) >= 3:
        try:
            it = IndependenceTest()
            report.independence = it.test(d_incentive, d_tvl)
            report.counter_examples = it.counter_examples(
                incentive_usd, tvl, tvl_horizon=30
            )
        except Exception:  # noqa: BLE001
            pass

    # ── Regime analysis ──────────────────────────────────────────────────
    if len(d_tvl) >= 10:
        try:
            ra = RegimeAnalyzer(bootstrap_n=bootstrap_n)
            vol_info = ra.classify_volatility(tvl)
            high_vol = vol_info["high_vol"][: len(d_tvl)]
            ext_vol = vol_info["extreme_vol"][: len(d_tvl)]

            if high_vol.sum() >= 3 and (~high_vol).sum() >= 3:
                report.regime_high_vol = ra.fit_conditional(
                    d_incentive, d_tvl, high_vol, "High Volatility"
                )
            if ext_vol.sum() >= 3 and (~ext_vol).sum() >= 3:
                report.regime_extreme_vol = ra.fit_conditional(
                    d_incentive, d_tvl, ext_vol, "Extreme Volatility"
                )
        except Exception:  # noqa: BLE001
            pass

    # ── Profitability ────────────────────────────────────────────────────
    latest_tvl = float(tvl[-1]) if len(tvl) > 0 else 0.0
    latest_reward_apy = float(apy_reward[-1]) if len(apy_reward) > 0 else 0.0
    latest_base_apy = float(apy_base[-1]) if len(apy_base) > 0 else 0.0
    daily_inc = daily_incentives_usd or (latest_tvl * latest_reward_apy / 365.0)

    if latest_tvl > 0:
        try:
            pa = ProfitabilityAnalyzer(irm=irm)
            if irm is not None:
                report.profitability_current = pa.snapshot(
                    tvl=latest_tvl,
                    utilization=current_utilization,
                    daily_incentives=daily_inc,
                )
            else:
                # Back-calculate borrow_apr from supply APY (DeFiLlama data)
                # supply_apr = r_b × util × (1 - RF)  →  r_b = supply_apr / (util × (1-RF))
                rf = 0.10
                denom = current_utilization * (1.0 - rf)
                borrow_apr = latest_base_apy / denom if denom > 0 and latest_base_apy > 0 else 0.0
                if borrow_apr > 0:
                    report.profitability_current = pa.snapshot(
                        tvl=latest_tvl,
                        utilization=current_utilization,
                        daily_incentives=daily_inc,
                        borrow_apr=borrow_apr,
                        supply_apr=latest_base_apy,
                    )
        except (ValueError, ZeroDivisionError):
            pass

    return report

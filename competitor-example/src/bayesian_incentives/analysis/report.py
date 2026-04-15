"""
End-to-end report generator.

Takes raw market data, runs the full analysis pipeline, and returns
a structured report with all results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from bayesian_incentives.data.transforms import (
    compute_first_differences,
    compute_log_returns,
    compute_realized_volatility,
    compute_utilization,
)
from bayesian_incentives.models.elasticity import ElasticityModel, OLSResult, BayesianPosterior
from bayesian_incentives.models.decay import DecayModel, DecayResult
from bayesian_incentives.models.independence import IndependenceTest, IndependenceResult
from bayesian_incentives.models.regimes import RegimeAnalyzer, RegimeElasticityResult
from bayesian_incentives.analysis.profitability import ProfitabilityAnalyzer, ProfitabilitySnapshot


@dataclass
class AnalysisReport:
    """Container for all analysis results on a single market."""

    market_name: str

    # Elasticity
    ols_total: OLSResult | None = None
    ols_supply: OLSResult | None = None
    ols_borrow: OLSResult | None = None
    posterior_total: BayesianPosterior | None = None
    lagged_results: dict[int, OLSResult] | None = None

    # Decay
    decay: DecayResult | None = None

    # Independence
    independence: IndependenceResult | None = None
    counter_examples: dict | None = None

    # Regimes
    regime_volatility: RegimeElasticityResult | None = None
    regime_utilization: RegimeElasticityResult | None = None
    regime_stress: RegimeElasticityResult | None = None

    # Profitability
    profitability: ProfitabilitySnapshot | None = None

    # Metadata
    n_observations: int = 0
    date_range: tuple[str, str] = ("", "")

    def summary(self) -> str:
        """Full text summary."""
        lines = [
            f"{'='*60}",
            f"ANALYSIS REPORT: {self.market_name}",
            f"{'='*60}",
            f"Date range: {self.date_range[0]} to {self.date_range[1]}",
            f"Observations: {self.n_observations}",
            "",
        ]

        # Elasticity
        lines.append("── Elasticity ──")
        if self.ols_total:
            lines.append(f"  Total:  {self.ols_total.summary()}")
        if self.ols_supply:
            lines.append(f"  Supply: {self.ols_supply.summary()}")
        if self.ols_borrow:
            lines.append(f"  Borrow: {self.ols_borrow.summary()}")
        if self.posterior_total:
            ci = self.posterior_total.ci_95
            lines.append(
                f"  Posterior: N({self.posterior_total.mu:,.2f}, "
                f"{self.posterior_total.sigma:,.2f}), "
                f"95% CI=[{ci[0]:,.2f}, {ci[1]:,.2f}]"
            )
        lines.append("")

        # Decay
        if self.decay:
            lines.append("── Decay / Persistence ──")
            lines.append(f"  {self.decay.summary()}")
            lines.append("")

        # Independence
        if self.independence:
            lines.append("── Independence Test ──")
            lines.append(f"  {self.independence.summary()}")
            lines.append("")

        # Regimes
        regime_results = [
            self.regime_volatility,
            self.regime_utilization,
            self.regime_stress,
        ]
        regime_results = [r for r in regime_results if r is not None]
        if regime_results:
            lines.append("── Regime-Conditional Elasticity ──")
            for r in regime_results:
                lines.append(f"  {r.summary()}")
            lines.append("")

        # Profitability
        if self.profitability:
            lines.append("── Profitability ──")
            lines.append(f"  {self.profitability.summary()}")

        return "\n".join(lines)


def generate_report(
    df: pd.DataFrame,
    market_name: str = "Market",
    tvl_col: str = "tvl",
    incentive_total_col: str = "incentive_total",
    incentive_supply_col: str | None = "incentive_supply",
    incentive_borrow_col: str | None = "incentive_borrow",
    price_col: str | None = "price",
    utilization_col: str | None = "utilization",
    borrow_volume_col: str | None = "borrow_volume",
    borrow_apr_col: str | None = "borrow_apr",
    supply_apr_col: str | None = "supply_apr",
    prior_mu: float = 0.0,
    prior_sigma: float = 1e6,
    bootstrap_n: int = 1000,
) -> AnalysisReport:
    """Run the full analysis pipeline on a market dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Market data (datetime-indexed).
    market_name : str
        Label for the market.
    *_col : str or None
        Column name overrides. Set to None if unavailable.
    prior_mu, prior_sigma : float
        Bayesian prior parameters for elasticity.
    bootstrap_n : int
        Number of bootstrap samples for regime interaction tests.

    Returns
    -------
    AnalysisReport
    """
    report = AnalysisReport(market_name=market_name)
    report.n_observations = len(df)

    idx = df.index
    report.date_range = (str(idx.min())[:10], str(idx.max())[:10])

    # ── Prepare transformations ─────────────────────────────────────────
    working = df.copy()

    # First differences for TVL and incentives
    tvl = working[tvl_col].values
    d_tvl = np.diff(tvl)

    inc_total = working[incentive_total_col].values
    d_inc_total = np.diff(inc_total)

    has_supply = incentive_supply_col and incentive_supply_col in working.columns
    has_borrow = incentive_borrow_col and incentive_borrow_col in working.columns
    has_price = price_col and price_col in working.columns
    has_util = utilization_col and utilization_col in working.columns

    # ── Elasticity: total ───────────────────────────────────────────────
    model_total = ElasticityModel(prior_mu=prior_mu, prior_sigma=prior_sigma)
    report.ols_total = model_total.fit_ols(d_inc_total, d_tvl)
    report.posterior_total = model_total.fit_posterior(d_inc_total, d_tvl)

    # Lagged
    report.lagged_results = model_total.fit_lagged(d_inc_total, tvl)

    # Supply / borrow split
    if has_supply:
        d_inc_s = np.diff(working[incentive_supply_col].values)
        m_s = ElasticityModel(prior_mu=prior_mu, prior_sigma=prior_sigma)
        report.ols_supply = m_s.fit_ols(d_inc_s, d_tvl)

    if has_borrow:
        d_inc_b = np.diff(working[incentive_borrow_col].values)
        m_b = ElasticityModel(prior_mu=prior_mu, prior_sigma=prior_sigma)
        report.ols_borrow = m_b.fit_ols(d_inc_b, d_tvl)

    # ── Decay ───────────────────────────────────────────────────────────
    decay_model = DecayModel()
    report.decay = decay_model.fit(tvl)

    # ── Independence ────────────────────────────────────────────────────
    ind_test = IndependenceTest()
    report.independence = ind_test.test(d_inc_total, d_tvl)
    report.counter_examples = ind_test.test_counter_examples(inc_total, tvl)

    # ── Regimes ─────────────────────────────────────────────────────────
    regime_analyzer = RegimeAnalyzer()

    if has_price:
        working = compute_log_returns(working, price_col=price_col)
        working = compute_realized_volatility(working)
        vol = working["realized_vol"].values[1:]  # align with diffs
        min_len = min(len(d_inc_total), len(d_tvl), len(vol))

        report.regime_volatility = regime_analyzer.fit_conditional_elasticity(
            d_inc_total[:min_len],
            d_tvl[:min_len],
            regime_analyzer.classify_volatility(vol[:min_len])["high_vol"],
            regime_name="High Volatility",
            n_bootstrap=bootstrap_n,
        )

    if has_util:
        util = working[utilization_col].values[1:]
        min_len = min(len(d_inc_total), len(d_tvl), len(util))

        report.regime_utilization = regime_analyzer.fit_conditional_elasticity(
            d_inc_total[:min_len],
            d_tvl[:min_len],
            regime_analyzer.classify_utilization(util[:min_len])["high_util"],
            regime_name="High Utilization",
            n_bootstrap=bootstrap_n,
        )

        if has_price:
            vol_trimmed = working["realized_vol"].values[1:]
            stress_len = min(len(d_inc_total), len(d_tvl), len(vol_trimmed), len(util))
            stress_mask = regime_analyzer.classify_stress(
                vol_trimmed[:stress_len], util[:stress_len]
            )
            report.regime_stress = regime_analyzer.fit_conditional_elasticity(
                d_inc_total[:stress_len],
                d_tvl[:stress_len],
                stress_mask,
                regime_name="Stress",
                n_bootstrap=bootstrap_n,
            )

    # ── Profitability snapshot (latest observation) ─────────────────────
    if has_util:
        latest_util = float(working[utilization_col].iloc[-1])
        latest_tvl = float(working[tvl_col].iloc[-1])
        latest_inc = float(working[incentive_total_col].iloc[-1])

        b_apr = float(working[borrow_apr_col].iloc[-1]) if borrow_apr_col and borrow_apr_col in working.columns else None
        s_apr = float(working[supply_apr_col].iloc[-1]) if supply_apr_col and supply_apr_col in working.columns else None

        prof_analyzer = ProfitabilityAnalyzer()
        try:
            report.profitability = prof_analyzer.snapshot(
                tvl=latest_tvl,
                utilization=latest_util,
                incentive_total=latest_inc,
                borrow_apr=b_apr,
                supply_apr=s_apr,
            )
        except ValueError:
            pass  # Missing rate model and no APR data

    return report

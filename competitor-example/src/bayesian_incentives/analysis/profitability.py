"""
Profitability analysis for protocol markets.

Computes:
  - Daily profit: NII - incentives
  - Net interest income (NII)
  - Breakeven utilization
  - Market health classification (well-utilized / at breakeven / under-utilized)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from bayesian_incentives.models.interest_rate import InterestRateModel, RateParams


class MarketStatus(str, Enum):
    WELL_UTILIZED = "WELL_UTILIZED"
    AT_BREAKEVEN = "AT_BREAKEVEN"
    UNDER_UTILIZED = "UNDER_UTILIZED"


@dataclass
class ProfitabilitySnapshot:
    """Point-in-time profitability assessment."""

    tvl: float
    borrow_volume: float
    utilization: float
    borrow_apr: float
    supply_apr: float
    daily_interest_revenue: float
    daily_interest_cost: float
    daily_nii: float
    daily_incentives: float
    daily_profit: float
    breakeven_utilization: float
    status: MarketStatus

    def summary(self) -> str:
        return (
            f"TVL: ${self.tvl:,.0f}  |  Borrows: ${self.borrow_volume:,.0f}  |  "
            f"Util: {self.utilization:.2%}\n"
            f"Borrow APR: {self.borrow_apr:.2%}  |  Supply APR: {self.supply_apr:.2%}\n"
            f"Daily NII: ${self.daily_nii:,.2f}  |  "
            f"Daily Incentives: ${self.daily_incentives:,.2f}\n"
            f"Daily Profit: ${self.daily_profit:,.2f}\n"
            f"Breakeven U: {self.breakeven_utilization:.2%}  |  "
            f"Status: {self.status.value}"
        )


class ProfitabilityAnalyzer:
    """Analyze market profitability and classify health.

    Parameters
    ----------
    rate_model : InterestRateModel or None
        Interest rate model. If None, rates must be provided in the data.
    breakeven_band : float
        Width of the "at breakeven" band around U_BE (default 0.05).
    """

    def __init__(
        self,
        rate_model: InterestRateModel | None = None,
        breakeven_band: float = 0.05,
    ) -> None:
        self.rate_model = rate_model
        self.breakeven_band = breakeven_band

    def _classify(
        self, utilization: float, breakeven_u: float
    ) -> MarketStatus:
        if utilization > breakeven_u + self.breakeven_band:
            return MarketStatus.WELL_UTILIZED
        elif abs(utilization - breakeven_u) <= self.breakeven_band:
            return MarketStatus.AT_BREAKEVEN
        else:
            return MarketStatus.UNDER_UTILIZED

    def snapshot(
        self,
        tvl: float,
        utilization: float,
        incentive_total: float,
        borrow_apr: Optional[float] = None,
        supply_apr: Optional[float] = None,
    ) -> ProfitabilitySnapshot:
        """Compute a single point-in-time profitability snapshot.

        Parameters
        ----------
        tvl : float
            Total value locked (supply).
        utilization : float
            Utilization ratio (B/S).
        incentive_total : float
            Daily total incentive spend.
        borrow_apr, supply_apr : float, optional
            If provided, used directly. Otherwise computed from rate_model.

        Returns
        -------
        ProfitabilitySnapshot
        """
        borrow_volume = tvl * utilization

        if borrow_apr is None or supply_apr is None:
            if self.rate_model is None:
                raise ValueError(
                    "Must provide borrow_apr and supply_apr, or a rate_model."
                )
            borrow_apr = float(self.rate_model.borrow_rate(utilization))
            supply_apr = float(self.rate_model.supply_rate(utilization))

        daily_revenue = borrow_volume * borrow_apr / 365
        daily_cost = tvl * supply_apr / 365
        daily_nii = daily_revenue - daily_cost
        daily_profit = daily_nii - incentive_total

        breakeven_u = supply_apr / borrow_apr if borrow_apr > 0 else 0.0

        status = self._classify(utilization, breakeven_u)

        return ProfitabilitySnapshot(
            tvl=tvl,
            borrow_volume=borrow_volume,
            utilization=utilization,
            borrow_apr=borrow_apr,
            supply_apr=supply_apr,
            daily_interest_revenue=daily_revenue,
            daily_interest_cost=daily_cost,
            daily_nii=daily_nii,
            daily_incentives=incentive_total,
            daily_profit=daily_profit,
            breakeven_utilization=breakeven_u,
            status=status,
        )

    def time_series(
        self,
        df: pd.DataFrame,
        tvl_col: str = "tvl",
        utilization_col: str = "utilization",
        incentive_col: str = "incentive_total",
        borrow_apr_col: Optional[str] = "borrow_apr",
        supply_apr_col: Optional[str] = "supply_apr",
    ) -> pd.DataFrame:
        """Compute profitability metrics for an entire time series.

        Returns a new DataFrame with columns: nii, profit, status.
        """
        records = []
        for idx, row in df.iterrows():
            b_apr = row.get(borrow_apr_col) if borrow_apr_col in df.columns else None
            s_apr = row.get(supply_apr_col) if supply_apr_col in df.columns else None

            snap = self.snapshot(
                tvl=row[tvl_col],
                utilization=row[utilization_col],
                incentive_total=row[incentive_col],
                borrow_apr=b_apr,
                supply_apr=s_apr,
            )
            records.append(
                {
                    "date": idx,
                    "nii": snap.daily_nii,
                    "profit": snap.daily_profit,
                    "status": snap.status.value,
                    "breakeven_u": snap.breakeven_utilization,
                }
            )
        return pd.DataFrame(records).set_index("date")

    def allocation_rules(
        self, status: MarketStatus
    ) -> dict[str, str]:
        """Return recommended allocation strategy based on market status.

        These are heuristic starting points; the Bayesian optimizer
        produces data-driven recommendations.
        """
        rules = {
            MarketStatus.WELL_UTILIZED: {
                "supply": "~$0 (unnecessary)",
                "borrow": "~$0 (unnecessary)",
                "rationale": "Already profitable. Incentives are a pure cost.",
            },
            MarketStatus.AT_BREAKEVEN: {
                "supply": "~$0 (cut)",
                "borrow": "Low (marginal utility)",
                "rationale": "Supply incentives are wasteful. Small borrow incentives may help utilization.",
            },
            MarketStatus.UNDER_UTILIZED: {
                "supply": "$0 (cut entirely)",
                "borrow": "Moderate (focus on demand)",
                "rationale": "Losing money on spread. Only borrow incentives can help by increasing utilization.",
            },
        }
        return rules[status]

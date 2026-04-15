"""
Kinked interest rate model for lending protocols.

Below kink (U <= U_optimal):
    r_b(U) = r_base + (U / U_optimal) * r_slope1

Above kink (U > U_optimal):
    r_b(U) = r_base + r_slope1 + ((U - U_optimal) / (1 - U_optimal)) * r_slope2

Supply rate:
    r_s = r_b * U * (1 - reserve_factor)

This is the standard model used by Compound, Aave, and similar protocols.
For protocols with different rate mechanics, users can subclass or pass
pre-computed rates directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RateParams:
    """Parameters for the kinked interest rate curve."""

    r_base: float = 0.0       # base borrow rate
    r_slope1: float = 0.05    # slope below kink
    r_slope2: float = 3.0     # slope above kink (steep)
    u_optimal: float = 0.80   # kink utilization
    reserve_factor: float = 0.10  # protocol's cut of interest


class InterestRateModel:
    """Compute borrow/supply rates from utilization using a kinked rate model.

    Parameters
    ----------
    params : RateParams or dict
        Rate curve parameters.
    """

    def __init__(self, params: RateParams | dict | None = None) -> None:
        if params is None:
            self.params = RateParams()
        elif isinstance(params, dict):
            self.params = RateParams(**params)
        else:
            self.params = params

    def borrow_rate(self, utilization: float | np.ndarray) -> float | np.ndarray:
        """Compute annualized borrow rate for given utilization."""
        p = self.params
        u = np.asarray(utilization, dtype=float)

        below_kink = p.r_base + (u / p.u_optimal) * p.r_slope1
        above_kink = (
            p.r_base
            + p.r_slope1
            + ((u - p.u_optimal) / (1 - p.u_optimal)) * p.r_slope2
        )

        return np.where(u <= p.u_optimal, below_kink, above_kink)

    def supply_rate(self, utilization: float | np.ndarray) -> float | np.ndarray:
        """Compute annualized supply rate: r_s = r_b * U * (1 - RF)."""
        r_b = self.borrow_rate(utilization)
        u = np.asarray(utilization, dtype=float)
        return r_b * u * (1 - self.params.reserve_factor)

    def net_interest_income(
        self,
        tvl: float,
        utilization: float,
        daily: bool = True,
    ) -> float:
        """Net interest income: borrow revenue - supply cost.

        NII = S * (U * r_b - r_s) / 365

        Parameters
        ----------
        tvl : float
            Total supply (S).
        utilization : float
            Current utilization.
        daily : bool
            If True, return daily NII. Otherwise annualized.

        Returns
        -------
        float
        """
        r_b = float(self.borrow_rate(utilization))
        r_s = float(self.supply_rate(utilization))
        u = utilization
        nii = tvl * (u * r_b - r_s)
        return nii / 365 if daily else nii

    def breakeven_utilization(self) -> float:
        """Utilization at which NII = 0.

        At breakeven: U * r_b = r_s = r_b * U * (1 - RF)
        This simplifies to U_BE = r_s / r_b, but since r_s = r_b * U * (1-RF),
        breakeven is always at U where the reserve factor exactly covers spread.

        For the kinked model, solve numerically.
        """
        from scipy.optimize import brentq

        def nii_at_u(u: float) -> float:
            return self.net_interest_income(tvl=1.0, utilization=u, daily=False)

        try:
            return brentq(nii_at_u, 0.001, 0.999)
        except ValueError:
            # NII is always positive or always negative in [0,1]
            return np.nan

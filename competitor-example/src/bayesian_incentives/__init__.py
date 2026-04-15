"""
Bayesian Incentives Optimizer
=============================

A general-purpose framework for analyzing and optimizing protocol incentive
allocations using Bayesian inference, elasticity analysis, and Monte Carlo
simulation.

Core methodology:
  1. Elasticity estimation (OLS + Bayesian posterior)
  2. AR(1) decay / persistence modeling
  3. Regime-conditional analysis (volatility, utilization, stress)
  4. Independence testing (chi-square)
  5. Monte Carlo profit simulation with posterior sampling
  6. Grid-search optimization with closed-loop feedback
"""

__version__ = "0.1.0"

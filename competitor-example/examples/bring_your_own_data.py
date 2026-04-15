"""
Example: Bring Your Own Data

This script shows how to load your own CSV data, validate it,
and run the full analysis + optimization pipeline.

Your CSV should have at minimum:
  - date: datetime column
  - tvl: total value locked (USD)
  - incentive_total: total daily incentive spend (USD)

Optional columns for richer analysis:
  - incentive_supply / incentive_borrow: split by side
  - borrow_volume: for utilization computation
  - utilization: pre-computed utilization ratio
  - supply_apr / borrow_apr: interest rates
  - price: underlying asset price (for volatility regimes)

Usage:
    python bring_your_own_data.py --data path/to/your_data.csv
    python bring_your_own_data.py --data path/to/your_data.csv --market "ETH USDC"
"""

import argparse
import sys

import pandas as pd

from bayesian_incentives.data.loader import load_market_data
from bayesian_incentives.data.schema import validate_market_data
from bayesian_incentives.analysis.report import generate_report
from bayesian_incentives.optimization.bayesian import BayesianOptimizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Bayesian incentive analysis on your own data."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to CSV, JSON, or Parquet file with market data."
    )
    parser.add_argument(
        "--market", type=str, default="Market",
        help="Market name for the report (default: 'Market')."
    )
    parser.add_argument(
        "--n-mc", type=int, default=2000,
        help="Monte Carlo samples for optimization (default: 2000)."
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="Projection horizon in days (default: 30)."
    )
    parser.add_argument(
        "--grid-n", type=int, default=20,
        help="Grid resolution per dimension (default: 20)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for the report (default: print to stdout)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load and validate ───────────────────────────────────────────────
    print(f"Loading data from: {args.data}")
    try:
        df = load_market_data(args.data, validate=True)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Loaded {len(df)} observations")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Columns: {list(df.columns)}")
    print()

    # ── Full analysis report ────────────────────────────────────────────
    print(f"Running analysis for: {args.market}")
    report = generate_report(df, market_name=args.market)

    report_text = report.summary()
    print(report_text)
    print()

    # ── Bayesian optimization ───────────────────────────────────────────
    print("Running Bayesian optimization...")
    optimizer = BayesianOptimizer(
        n_mc=args.n_mc,
        horizon=args.horizon,
        grid_n=args.grid_n,
    )
    result = optimizer.run_iteration(df)
    print(result.summary())
    print()

    rec = optimizer.recommendation()
    print("─── RECOMMENDATION ───")
    print(f"Assessment: {rec['assessment']}")
    print(f"Rationale:  {rec['rationale']}")
    print(f"Optimal:    I_supply=${rec['optimal_supply']:,.2f}, "
          f"I_borrow=${rec['optimal_borrow']:,.2f}")
    print(f"E[profit]:  ${rec['expected_daily_profit']:,.2f}/day")
    print(f"Converged:  {rec['converged']}")
    print()

    # ── Save report ─────────────────────────────────────────────────────
    if args.output:
        full_report = report_text + "\n\n" + result.summary() + "\n\n"
        full_report += "─── RECOMMENDATION ───\n"
        for k, v in rec.items():
            full_report += f"  {k}: {v}\n"

        with open(args.output, "w") as f:
            f.write(full_report)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()

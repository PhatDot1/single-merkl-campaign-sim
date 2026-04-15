# Bayesian Incentives Optimizer

A general-purpose framework for analyzing and optimizing protocol incentive allocations using Bayesian inference, elasticity analysis, and Monte Carlo simulation.

Originally developed for DeFi lending protocol incentive optimization, the framework is protocol-agnostic: bring your own time-series data for any platform that distributes token incentives and wants to understand whether they work.

## What it does

Given historical data on a target metric (TVL, volume, etc.) and incentive spend, the framework answers:

1. **Do incentives move the needle?** Elasticity analysis with OLS regression and Bayesian posteriors on first differences.
2. **How quickly do effects decay?** AR(1) persistence modeling with half-life and equilibrium estimation.
3. **Does it depend on market conditions?** Regime-conditional analysis across volatility, utilization, and stress periods.
4. **Is the directional relationship real?** Chi-square independence testing and counter-example analysis.
5. **What's the profit-maximizing allocation?** Monte Carlo grid search over incentive allocations with posterior sampling.
6. **How should we update as new data arrives?** Closed-loop Bayesian feedback with convergence diagnostics.

## Installation

```bash
git clone https://github.com/osepper/bayesian-incentives-optimizer.git
cd bayesian-incentives-optimizer
pip install -e .

# With dev dependencies (pytest, ruff, mypy):
pip install -e ".[dev]"
```

## Quick start

### Synthetic demo

```bash
python examples/synthetic_demo.py
```

### Bring your own data

Prepare a CSV with at minimum: `date`, `tvl`, `incentive_total`.

```bash
python examples/bring_your_own_data.py --data your_data.csv --market "ETH USDC"
```

### Python API

```python
import pandas as pd
from bayesian_incentives.analysis.report import generate_report
from bayesian_incentives.optimization.bayesian import BayesianOptimizer

# Load your data
df = pd.read_csv("your_data.csv", parse_dates=["date"]).set_index("date")

# Full analysis report
report = generate_report(df, market_name="My Market")
print(report.summary())

# Bayesian optimization
optimizer = BayesianOptimizer(n_mc=2000, horizon=30, grid_n=20)
result = optimizer.run_iteration(df)
print(result.summary())

# Human-readable recommendation
rec = optimizer.recommendation()
print(f"Assessment: {rec['assessment']}")
print(f"Optimal: I_supply=${rec['optimal_supply']:,.2f}, I_borrow=${rec['optimal_borrow']:,.2f}")
```

## Data schema

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Observation date |
| `tvl` | float | Total value locked or target metric (USD) |
| `incentive_total` | float | Total daily incentive spend (USD) |

### Optional columns (enable richer analysis)

| Column | Type | Enables |
|--------|------|---------|
| `incentive_supply` | float | Supply-side elasticity estimation |
| `incentive_borrow` | float | Borrow-side elasticity estimation |
| `utilization` | float | Profitability analysis, utilization regimes |
| `borrow_volume` | float | Utilization computation (if not provided) |
| `supply_apr` | float | NII computation from data |
| `borrow_apr` | float | NII computation from data |
| `price` | float | Volatility regime analysis |

See `examples/sample_data_template.csv` for a reference.

## Architecture

```
src/bayesian_incentives/
├── data/           # Loading, validation, transformations
├── models/         # Statistical models
│   ├── elasticity.py      # OLS + Bayesian posterior
│   ├── decay.py           # AR(1) persistence
│   ├── interest_rate.py   # Kinked rate model
│   ├── regimes.py         # Volatility/utilization regimes
│   └── independence.py    # Chi-square test
├── optimization/   # Grid search + Bayesian optimizer
│   ├── grid_search.py     # MC profit evaluation
│   └── bayesian.py        # Closed-loop orchestrator
├── analysis/       # Report generation, profitability
└── visualization/  # Matplotlib plotting functions
```

## Methodology

The full mathematical methodology is documented in `docs/methodology.md`, covering:

- **Elasticity**: OLS on first differences with Bayesian normal-normal conjugate posterior
- **Decay**: AR(1) model with persistence coefficient, half-life, and impulse response
- **Regimes**: Quantile-based classification with bootstrap interaction tests
- **Independence**: Chi-square test on 2x2 contingency table of directional changes
- **Optimization**: Monte Carlo profit simulation over an incentive grid, sampling from posterior distributions
- **Closed-loop feedback**: Iterative posterior updates as new data arrives

## Configuration

Copy and modify `config/default.yaml`:

```yaml
market:
  name: "ETH USDC"

rate_model:
  r_base: 0.0
  r_slope1: 0.05
  r_slope2: 3.0
  u_optimal: 0.80
  reserve_factor: 0.10

optimization:
  n_mc: 2000
  horizon: 30
  grid_n: 20
```

## Development

```bash
make dev       # Install with dev dependencies
make test      # Run test suite
make lint      # Lint with ruff
make typecheck # Type check with mypy
make demo      # Run synthetic demo
```

## License

MIT

Inputs:

weekly_budget_range

tvl_target 

apr_ceiling

Loss weights - Stakeholder priority tradeoffs (adjustable in dashboard)


Current Limitations:

heuristic estimates for α⁺ and σ are decent but not calibrated


30-day time series of daily TVL + realized APR (which you can pull from your internal tools or the Morpho/Euler APIs or from an indexer like dune?), the pipeline in data.py could regress actual ΔTVL on ΔrAPR and get empirical values

The _estimate_alpha_plus and _estimate_sigma functions are marked as placeholders — once you have the time series, replace them with proper OLS regressions. 

[cross venue optimality? and adding support for more vaults]


# Run tests
pytest tests/ -v

# === OPTION A: Live data (fetches on-chain + APIs automatically) ===
python scripts/compute_surface.py --live \
  --vault-address 0x19b3...8c55 \
  --vault-type morpho \
  --asset-symbol PYUSD \
  --budget-min 100000 \
  --budget-max 250000 \
  --output results/morpho_pyusd \
  --n-paths 100

# === OPTION B: Static config (hardcoded parameters) ===
python scripts/compute_surface.py --venue morpho_pyusd --output results/morpho_pyusd --n-paths 100

# Launch dashboard (instant — loads pre-computed data)
streamlit run dashboard/app.py



# Place venues_test.json in project root, then:
python scripts/compute_surface.py --multi --config venues_test.json --output results/test_allocation --n-paths 30

# Or for a quick single-venue re-run with base_apy:
python scripts/compute_surface.py --venue morpho_pyusd --output results/morpho_pyusd_v2 --n-paths 50


singlke test ab it redundant on budget cus it will always say max for the single venue


So, mathematically, this will:
Step 1: Per-venue surface computation. For each of the 3 venues, it created a 12×13 grid of (Budget, r_max) pairs. At each grid point, it ran 30 Monte Carlo simulation paths of 28 days (4 weekly epochs), each at 6-hour timesteps. That's 156 grid points × 30 paths = 4,680 simulations per venue, 14,040 total.Step 2: Per-path simulation. Each path runs the inner loop from Section 9.3 of your math framework — retail drift responds to lagged total APR vs r_threshold with asymmetric elasticity (3x faster outflows), whales check exit thresholds against total APR, mercenaries enter/exit at their thresholds, and diffusion noise gets applied. After each path completes, the loss functional (Section 4) is evaluated: spend cost, APR variance, APR ceiling, TVL shortfall, and Merkl fee leakage, all weighted.Step 3: Surface aggregation. The 30 path losses at each grid point are averaged to get the expected loss surface. Feasibility is checked (cascade depth tolerance). Component surfaces are stored separately so the dashboard can reweight without re-simulating.Step 4: Cross-venue allocation. This is the Lagrangian from Section 7.6. The allocator reads dL/dB (marginal loss reduction per dollar) from each venue's surface, then uses bisection to find λ* such that the budgets where each venue's marginal equals λ* sum to $1.3M. Then it scales the result to exactly match.

Rigerous:
The core APR formula r(t) = min(r_max, B/TVL × 365/7), the T_bind regime switching, the loss functional decomposition, and the Lagrangian cross-venue allocation are all clean math. The Monte Carlo averaging and Hessian sensitivity analysis are standard numerical methods. The cascade resolution logic correctly implements sequential whale exit game from Section 3.3.

CURRENTLY ASSUMED - CAN BE GAINED FROM HISTORICAL DATA:
- Retail elasticity parameters (α+ = 0.2-0.3, α⁻ = 3×)
>how fast TVL responds to APR changes.

- r_threshold values (4.5% for Morpho, 5% for AAVE/Euler)
>total APR at which depositors are neutral. They're set as static competitor-weighted averages.

- Whale profiles are stylized, not calibrated
>whales in your venues_test.json have manually set positions ($29.6M, $25M for Morpho; $50M for Euler) and exit thresholds derived from assumed alt_rates.
MAKE SURE: --live mode wil pull actual on-chain depositor positions and derive thresholds from behavioral history.

- Base APY is treated as constant. You set 1.15% for Morpho, 3.32% for AAVE, 2.00% for Euler. In reality, Morpho's base APY changes as utilization changes

- Diffusion sigma (σ = 0.01-0.015). Controls random TVL noise. Underestimating this makes the optimizer overconfident in tight configurations.


WHAT IS NOT ACCOUNTED FOR:

Borrower behavior changes:  Lower utilization → lower base APY → further compounds the APR drop. This feedback loop is not in the model.

Looper dropout.

Cross-venue TVL migration

Market regime shifts. 



Two-Phase Workflow
Phase 1 — Compute
scripts/compute_surface.py → results/{venue}/surfaces.npz + metadata.json


Phase 2 — Explore 
dashboard/app.py ← loads results/{venue}/
Interactive Streamlit app. Loss weights adjustable in real-time via sliders.


Kamino (Solana): No EVM calls possible — everything goes through their REST API. Their API gives TVL/rates but not individual depositor balances.
need either:

A Solana RPC call to enumerate token account holders (expensive, slow)
Kamino's own analytics API (if they expose it)
Manual input from your internal tools
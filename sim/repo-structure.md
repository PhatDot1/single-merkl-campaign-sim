```
campaign-optimizer/
├── .env                          ← NEW: RPC keys and contract addresses
├── .env.example                  ← NEW: Template showing required vars
├── pyproject.toml
├── README.md
├── HANDOVER_PROMPT.md
├── campaign/
│   ├── __init__.py               ← UPDATED: exports new modules
│   ├── state.py                  ← UNCHANGED
│   ├── agents.py                 ← UNCHANGED
│   ├── engine.py                 ← UNCHANGED
│   ├── optimizer.py              ← UNCHANGED
│   ├── multi_venue.py            ← UNCHANGED
│   ├── serialize.py              ← UNCHANGED
│   ├── config.py                 ← NEW: centralized env/address config
│   ├── base_apy.py               ← REWRITTEN: on-chain priority routing
│   ├── data.py                   ← UNCHANGED (legacy, still used)
│   ├── evm_data.py               ← NEW: Aave/Euler on-chain + whale fetching
│   └── kamino_data.py            ← NEW: Kamino Solana API integration
├── scripts/
│   ├── compute_surface.py        ← UNCHANGED
│   └── test_data_fetch.py        ← NEW: end-to-end data validation
├── dashboard/
│   └── app.py                    ← REWRITTEN: full venue list, dynamic r_threshold
├── examples/
│   └── morpho_pyusd.py           ← UNCHANGED
├── docs/
│   └── mathematical_framework.md ← UNCHANGED
├── results/
│   └── morpho_pyusd/
│       ├── surfaces.npz
│       ├── metadata.json
│       └── mc_diagnostics.json
└── tests/
    ├── __init__.py
    └── test_campaign.py          ← UNCHANGED
```

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Pre-compute surface (takes ~5-15 min depending on grid size)
python scripts/compute_surface.py --venue morpho_pyusd --output results/morpho_pyusd --n-paths 100

# Launch dashboard (instant — loads pre-computed data)
streamlit run dashboard/app.py
```

## Two-Phase Workflow

**Phase 1 — Compute (offline, slow)**
```
scripts/compute_surface.py → results/{venue}/surfaces.npz + metadata.json
```
Run this nightly, weekly, or whenever market conditions change materially.
Grid of 16×17 points × 100 MC paths = 27,200 simulations. ~10-15 min on a modern laptop.

**Phase 2 — Explore (dashboard, instant)**
```
dashboard/app.py ← loads results/{venue}/
```
Interactive Streamlit app. Loss weights adjustable in real-time via sliders.
No re-simulation — total loss is recomputed as a weighted sum of pre-computed component surfaces.
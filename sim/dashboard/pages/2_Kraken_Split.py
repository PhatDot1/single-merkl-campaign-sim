"""
Kraken Earn Split Incentive — embedded Streamlit page.

This file is the multi-page entry point; all logic lives in
../kraken_split.py and its main() function is called directly.
"""

import os
import sys

# Ensure both the dashboard dir (for kraken_split imports) and the sim
# root (for `campaign.*` imports) are on the path.
_dash_dir = os.path.dirname(os.path.dirname(__file__))   # sim/dashboard/
_sim_dir  = os.path.dirname(_dash_dir)                   # sim/

for _p in (_dash_dir, _sim_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kraken_split import main

main()

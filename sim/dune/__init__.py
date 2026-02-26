"""
Dune Analytics data sync infrastructure.

Provides SQL query templates and a sync runner that fetches whale flow
data, mercenary detection data, and historical APR data from Dune,
storing results as local CSVs for use by the simulation engine.

Usage:
    from dune.sync import sync_all_venues, sync_venue, load_whale_flows
    from dune.queries import WHALE_FLOWS_QUERY, MERCENARY_DETECTION_QUERY
"""

from .queries import (
    HISTORICAL_APR_QUERY as HISTORICAL_APR_QUERY,
)
from .queries import (
    MERCENARY_DETECTION_QUERY as MERCENARY_DETECTION_QUERY,
)
from .queries import (
    WHALE_FLOWS_QUERY as WHALE_FLOWS_QUERY,
)
from .sync import (
    DuneSyncResult as DuneSyncResult,
)
from .sync import (
    build_whale_history_lookup as build_whale_history_lookup,
)
from .sync import (
    derive_mercenary_thresholds as derive_mercenary_thresholds,
)
from .sync import (
    load_mercenary_data as load_mercenary_data,
)
from .sync import (
    load_whale_flows as load_whale_flows,
)
from .sync import (
    sync_all_venues as sync_all_venues,
)
from .sync import (
    sync_venue as sync_venue,
)

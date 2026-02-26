"""
Dune Analytics data sync runner.

Fetches whale flow data, mercenary patterns, and historical rates from Dune,
saving results as local CSVs in dune/data/{pool_id}/.

Rate limiting: 2-second delay between Dune API calls to respect free-tier limits.

Usage:
    from dune.sync import sync_all_venues, sync_venue, load_whale_flows

    # Sync all venues (called from dashboard "Sync Dune Data" button)
    results = sync_all_venues()

    # Load cached data for a venue
    flows_df = load_whale_flows("rlusd-aave-v3-core")
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .queries import (
    MERCENARY_DETECTION_QUERY,
    WHALE_FLOWS_QUERY,
    render_query,
)

# Data directory for cached CSVs
DATA_DIR = Path(__file__).parent / "data"

# Rate limit: 2s between Dune API calls
DUNE_RATE_LIMIT_SECONDS = 2.0

DUNE_API_BASE = "https://api.dune.com/api/v1"


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class DuneSyncResult:
    """Result of syncing data for one venue."""

    pool_id: str
    venue_name: str
    whale_flows_count: int = 0
    mercenary_count: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


# ============================================================================
# TOKEN ADDRESS RESOLUTION
# ============================================================================


def get_dune_token_address(venue) -> str | None:
    """
    Resolve the queryable token address for a venue.

    For Dune whale flow queries, we need the share/receipt token address:
    - Aave: aToken address (receipt token for deposits)
    - Euler: vault address (ERC-4626 shares)
    - Morpho: vault address (ERC-4626 shares)
    - Curve: LP token address (pool contract)
    - Kamino: None (Solana — Dune not applicable)

    Args:
        venue: VenueRecord or dict with venue info

    Returns:
        Token address string, or None if not queryable via Dune
    """
    # Handle both VenueRecord and dict
    if isinstance(venue, dict):
        chain = venue.get("chain", "").lower()
        protocol = venue.get("protocol", "").lower()
        a_token = venue.get("a_token_address", "")
        vault = venue.get("vault_address", "")
        address = venue.get("address", "")
    else:
        chain = getattr(venue, "chain", "").lower()
        protocol = getattr(venue, "protocol", "").lower()
        a_token = getattr(venue, "a_token_address", "")
        vault = getattr(venue, "vault_address", "")
        address = getattr(venue, "address", "")

    # Skip non-EVM chains
    if chain != "ethereum":
        return None

    # Aave: use aToken address
    if protocol == "aave" and a_token:
        return a_token

    # Euler / Morpho: use vault address (ERC-4626 share token)
    if protocol in ("euler", "morpho") and vault:
        return vault

    # Curve: use pool address (LP token)
    if protocol == "curve" and address:
        return address

    return None


def get_token_decimals(venue) -> int:
    """
    Get token decimals for a venue's share token.

    Most stablecoin aTokens/eTokens use 6 decimals (matching underlying).
    Curve LP tokens typically use 18 decimals.
    """
    if isinstance(venue, dict):
        protocol = venue.get("protocol", "").lower()
    else:
        protocol = getattr(venue, "protocol", "").lower()

    if protocol == "curve":
        return 18
    # Stablecoin aTokens, eTokens, Morpho vault shares: 6 decimals
    return 6


# ============================================================================
# DUNE API EXECUTION
# ============================================================================


def _get_dune_api_key() -> str:
    """Get Dune API key from environment."""
    key = os.environ.get("DUNE_API_KEY")
    if not key:
        raise RuntimeError(
            "DUNE_API_KEY not set. Set it in your .env file. "
            "Get a key at https://dune.com/settings/api"
        )
    return key


def _execute_dune_query(sql: str, api_key: str, timeout: int = 120) -> list[dict]:
    """
    Execute a SQL query on Dune and return result rows.

    Uses the inline query execution endpoint. Polls for completion
    with 2s intervals. Returns empty list on failure (logged, not raised).

    Args:
        sql: Rendered SQL query
        api_key: Dune API key
        timeout: Max wait time in seconds

    Returns:
        List of result row dicts
    """
    headers = {"X-Dune-API-Key": api_key}

    # Execute query via the /sql/execute endpoint (inline SQL execution)
    try:
        exec_resp = requests.post(
            f"{DUNE_API_BASE}/sql/execute",
            headers=headers,
            json={"sql": sql, "performance": "medium"},
            timeout=30,
        )

        if exec_resp.status_code == 402:
            raise RuntimeError("Dune API quota exceeded")

        exec_data = exec_resp.json()
        execution_id = exec_data.get("execution_id")

        if not execution_id:
            raise RuntimeError(f"No execution_id in response: {exec_data}")

    except Exception as e:
        print(f"  [Dune] Query execution failed: {e}")
        return []

    # Poll for results
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(DUNE_RATE_LIMIT_SECONDS)

        try:
            status_resp = requests.get(
                f"{DUNE_API_BASE}/execution/{execution_id}/status",
                headers=headers,
                timeout=15,
            )
            state = status_resp.json().get("state", "")

            if state == "QUERY_STATE_COMPLETED":
                break
            elif state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                print(f"  [Dune] Query failed with state: {state}")
                return []
        except Exception as e:
            print(f"  [Dune] Status poll error: {e}")
            continue
    else:
        print(f"  [Dune] Query timed out after {timeout}s")
        return []

    # Fetch results
    try:
        results_resp = requests.get(
            f"{DUNE_API_BASE}/execution/{execution_id}/results",
            headers=headers,
            timeout=30,
        )
        return results_resp.json().get("result", {}).get("rows", [])
    except Exception as e:
        print(f"  [Dune] Results fetch failed: {e}")
        return []


# ============================================================================
# CSV PERSISTENCE
# ============================================================================


def _ensure_data_dir(pool_id: str) -> Path:
    """Create data directory for a venue if it doesn't exist."""
    d = DATA_DIR / pool_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_rows_csv(rows: list[dict], filepath: Path) -> int:
    """Save list of dicts as CSV. Returns row count."""
    if not rows:
        return 0

    keys = rows[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def _load_csv(filepath: Path) -> list[dict]:
    """Load CSV as list of dicts. Returns empty list if file doesn't exist."""
    if not filepath.exists():
        return []

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================================
# SYNC: Per-venue data fetch
# ============================================================================


def sync_venue(
    venue,
    days: int = 90,
    min_whale_usd: float = 500_000,
    min_merc_usd: float = 100_000,
) -> DuneSyncResult:
    """
    Sync Dune data for a single venue.

    Fetches whale flows and mercenary detection data, saves to CSV.

    Args:
        venue: VenueRecord from venue_registry
        days: Lookback period
        min_whale_usd: Minimum transfer size for whale flows
        min_merc_usd: Minimum transfer size for mercenary detection

    Returns:
        DuneSyncResult with counts and any errors
    """
    pool_id = venue.pool_id if hasattr(venue, "pool_id") else venue["pool_id"]
    name = venue.name if hasattr(venue, "name") else venue["name"]

    result = DuneSyncResult(pool_id=pool_id, venue_name=name)
    t0 = time.time()

    # Resolve token address
    token_addr = get_dune_token_address(venue)
    if not token_addr:
        result.skipped = True
        result.skip_reason = "No queryable token address (non-EVM or missing address)"
        return result

    decimals = get_token_decimals(venue)
    api_key = _get_dune_api_key()
    data_dir = _ensure_data_dir(pool_id)

    # ── Whale flows ──
    print(f"  [Dune] Fetching whale flows for {name}...")
    whale_sql = render_query(
        WHALE_FLOWS_QUERY,
        token_address=token_addr,
        decimals=decimals,
        min_amount=min_whale_usd,
        days=days,
        chain="ethereum",
    )

    whale_rows = _execute_dune_query(whale_sql, api_key)
    result.whale_flows_count = _save_rows_csv(whale_rows, data_dir / "whale_flows.csv")

    # Rate limit between queries
    time.sleep(DUNE_RATE_LIMIT_SECONDS)

    # ── Mercenary detection ──
    print(f"  [Dune] Fetching mercenary data for {name}...")
    merc_sql = render_query(
        MERCENARY_DETECTION_QUERY,
        token_address=token_addr,
        decimals=decimals,
        min_amount=min_merc_usd,
        days=days,
        chain="ethereum",
    )

    merc_rows = _execute_dune_query(merc_sql, api_key)
    result.mercenary_count = _save_rows_csv(merc_rows, data_dir / "mercenary_detection.csv")

    result.elapsed_seconds = time.time() - t0
    print(
        f"  [Dune] {name}: {result.whale_flows_count} whale flows, "
        f"{result.mercenary_count} mercenary addresses "
        f"({result.elapsed_seconds:.1f}s)"
    )

    return result


def sync_all_venues(
    days: int = 90,
    min_whale_usd: float = 500_000,
    min_merc_usd: float = 100_000,
) -> list[DuneSyncResult]:
    """
    Sync Dune data for all EVM venues in the registry.

    Iterates VENUE_REGISTRY, skips non-EVM venues (Kamino/Solana),
    fetches whale flows + mercenary detection for each.

    Rate-limited: 2s between Dune API calls.

    Args:
        days: Lookback period in days
        min_whale_usd: Minimum whale transfer size (USD)
        min_merc_usd: Minimum mercenary transfer size (USD)

    Returns:
        List of DuneSyncResult (one per venue)
    """
    from campaign.venue_registry import VENUE_REGISTRY

    results = []
    venues = list(VENUE_REGISTRY.values())

    print(f"[Dune Sync] Starting sync for {len(venues)} venues...")

    for i, venue in enumerate(venues):
        print(f"\n[{i + 1}/{len(venues)}] {venue.name}")
        try:
            r = sync_venue(
                venue,
                days=days,
                min_whale_usd=min_whale_usd,
                min_merc_usd=min_merc_usd,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                DuneSyncResult(
                    pool_id=venue.pool_id,
                    venue_name=venue.name,
                    errors=[str(e)],
                )
            )

        # Rate limit between venues
        if i < len(venues) - 1:
            time.sleep(DUNE_RATE_LIMIT_SECONDS)

    total_flows = sum(r.whale_flows_count for r in results)
    total_merc = sum(r.mercenary_count for r in results)
    skipped = sum(1 for r in results if r.skipped)
    errored = sum(1 for r in results if r.errors)
    print(
        f"\n[Dune Sync] Complete: {total_flows} whale flows, "
        f"{total_merc} mercenary addresses, "
        f"{skipped} skipped, {errored} errors"
    )

    return results


# ============================================================================
# DATA LOADING: Read cached CSVs for use in simulation
# ============================================================================


def load_whale_flows(pool_id: str) -> list[dict]:
    """
    Load cached whale flow data for a venue.

    Returns list of dicts with keys:
    - block_time, whale_address, amount, direction, tx_hash

    Returns empty list if no cached data.
    """
    filepath = DATA_DIR / pool_id / "whale_flows.csv"
    rows = _load_csv(filepath)

    # Cast numeric fields
    for row in rows:
        try:
            row["amount"] = float(row.get("amount", 0))
        except (ValueError, TypeError):
            row["amount"] = 0.0

    return rows


def load_mercenary_data(pool_id: str) -> list[dict]:
    """
    Load cached mercenary detection data for a venue.

    Returns list of dicts with keys:
    - address, total_deposited, total_withdrawn, first_deposit,
      first_withdrawal, days_held

    Returns empty list if no cached data.
    """
    filepath = DATA_DIR / pool_id / "mercenary_detection.csv"
    rows = _load_csv(filepath)

    # Cast numeric fields
    for row in rows:
        for key in ("total_deposited", "total_withdrawn", "days_held"):
            try:
                row[key] = float(row.get(key, 0))
            except (ValueError, TypeError):
                row[key] = 0.0

    return rows


def get_sync_status() -> dict[str, dict]:
    """
    Get sync status for all venues — which have cached data and how recent.

    Returns:
        {pool_id: {"has_whale_flows": bool, "has_mercenary": bool,
                    "whale_count": int, "merc_count": int,
                    "last_modified": str}}
    """
    status = {}

    if not DATA_DIR.exists():
        return status

    for pool_dir in DATA_DIR.iterdir():
        if not pool_dir.is_dir():
            continue

        pool_id = pool_dir.name
        whale_file = pool_dir / "whale_flows.csv"
        merc_file = pool_dir / "mercenary_detection.csv"

        entry = {
            "has_whale_flows": whale_file.exists(),
            "has_mercenary": merc_file.exists(),
            "whale_count": 0,
            "merc_count": 0,
            "last_modified": "",
        }

        if whale_file.exists():
            entry["whale_count"] = len(_load_csv(whale_file))
            mtime = os.path.getmtime(whale_file)
            entry["last_modified"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))

        if merc_file.exists():
            entry["merc_count"] = len(_load_csv(merc_file))

        status[pool_id] = entry

    return status


def derive_mercenary_thresholds(pool_id: str, r_threshold: float) -> dict | None:
    """
    Derive mercenary entry/exit thresholds from cached Dune data.

    If we have mercenary detection data (addresses that deposited AND withdrew
    within 7 days), we can estimate what APR levels trigger mercenary behavior:
    - entry_threshold ≈ r_threshold × (1 + average_merc_premium)
    - exit_threshold ≈ r_threshold × (1 + average_merc_premium × 0.7)

    Falls back to None if no data available.

    Args:
        pool_id: Venue pool ID
        r_threshold: Current competitor rate

    Returns:
        {"entry_threshold": float, "exit_threshold": float, "n_mercs": int}
        or None if no data
    """
    mercs = load_mercenary_data(pool_id)
    if not mercs or len(mercs) < 3:
        return None

    # The presence of mercenary capital is a signal — more mercs = lower thresholds
    n_mercs = len(mercs)
    total_merc_volume = sum(m.get("total_deposited", 0) for m in mercs)

    # Scale entry threshold: more mercenary activity → lower threshold
    # (i.e., mercs enter at lower APR premiums when there's a track record)
    if n_mercs > 20:
        entry_mult = 1.5  # Lots of mercenary activity → they enter at 1.5× r_threshold
    elif n_mercs > 10:
        entry_mult = 1.8  # Moderate activity
    else:
        entry_mult = 2.0  # Low activity → higher threshold

    entry_threshold = r_threshold * entry_mult
    exit_threshold = r_threshold * (entry_mult * 0.75)  # Exit at 75% of entry premium

    return {
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
        "n_mercs": n_mercs,
        "total_merc_volume_usd": total_merc_volume,
    }


def build_whale_history_lookup(pool_id: str) -> dict | None:
    """
    Build whale history lookup dict from cached Dune whale flow CSVs.

    Aggregates per-address: number of deposits, withdrawals, and average
    hold time (days between first deposit and first withdrawal).

    This dict is passed to build_whale_profiles_from_holders() to replace
    synthetic thresholds with empirical ones.

    Args:
        pool_id: Venue pool ID

    Returns:
        {address: {"n_deposits": int, "n_withdrawals": int, "avg_hold_days": float,
                    "total_deposited": float, "total_withdrawn": float}}
        or None if no cached data
    """
    flows = load_whale_flows(pool_id)
    if not flows or len(flows) < 2:
        return None

    # Aggregate per address
    addr_data: dict[str, dict] = {}
    for flow in flows:
        addr = (flow.get("whale_address") or "").lower()
        if not addr:
            continue
        if addr not in addr_data:
            addr_data[addr] = {
                "deposits": [],
                "withdrawals": [],
                "total_deposited": 0.0,
                "total_withdrawn": 0.0,
            }

        amount = float(flow.get("amount", 0))
        direction = flow.get("direction", "")
        timestamp = flow.get("block_time", "")

        if direction == "deposit":
            addr_data[addr]["deposits"].append(timestamp)
            addr_data[addr]["total_deposited"] += amount
        elif direction == "withdrawal":
            addr_data[addr]["withdrawals"].append(timestamp)
            addr_data[addr]["total_withdrawn"] += amount

    # Compute per-address metrics
    result = {}
    for addr, data in addr_data.items():
        n_deposits = len(data["deposits"])
        n_withdrawals = len(data["withdrawals"])

        # Estimate average hold time
        avg_hold_days = 30.0  # default fallback
        if data["deposits"] and data["withdrawals"]:
            try:
                from datetime import datetime

                first_dep = min(data["deposits"])
                first_wd = min(data["withdrawals"])
                # Parse ISO timestamps
                d1 = datetime.fromisoformat(first_dep.replace("Z", "+00:00").split("+")[0])
                d2 = datetime.fromisoformat(first_wd.replace("Z", "+00:00").split("+")[0])
                hold = abs((d2 - d1).days)
                avg_hold_days = max(1.0, float(hold))
            except Exception:
                pass

        result[addr] = {
            "n_deposits": n_deposits,
            "n_withdrawals": n_withdrawals,
            "avg_hold_days": avg_hold_days,
            "total_deposited": data["total_deposited"],
            "total_withdrawn": data["total_withdrawn"],
        }

    return result if result else None


# ============================================================================
# WHALE STICKINESS SCORING
# ============================================================================


def compute_whale_stickiness_score(
    whale_history_entry: dict,
    exits_during_rate_drops: int = 0,
) -> float:
    """
    Compute a behavioural stickiness score ∈ [0.1, 1.0] for a single whale.

    Higher score = more sticky (slow to exit, loyal depositor).
    Lower score  = mercenary (quick to exit, rate-sensitive).

    Formula:
        base        = 1.0 - n_withdrawals / max(n_deposits, 1)
        hold_bonus  = min(avg_hold_days / 90, 0.30)    # capped at 0.30
        exit_penalty = exits_during_rate_drops × 0.15
        score       = clamp(base + hold_bonus - exit_penalty, 0.1, 1.0)

    Args:
        whale_history_entry: Single dict from build_whale_history_lookup():
            {"n_deposits": int, "n_withdrawals": int, "avg_hold_days": float, ...}
        exits_during_rate_drops: Number of times this whale exited during
            a known APR decline event (0 if not tracked).

    Returns:
        Stickiness score in [0.1, 1.0].
    """
    n_deps = max(int(whale_history_entry.get("n_deposits", 0)), 0)
    n_wds = max(int(whale_history_entry.get("n_withdrawals", 0)), 0)
    avg_hold = max(float(whale_history_entry.get("avg_hold_days", 30.0)), 0.0)

    base = 1.0 - n_wds / max(n_deps, 1)
    hold_bonus = min(avg_hold / 90.0, 0.30)
    exit_penalty = max(exits_during_rate_drops, 0) * 0.15

    raw = base + hold_bonus - exit_penalty
    return max(0.10, min(1.0, raw))


def stickiness_to_profile_params(score: float) -> dict:
    """
    Map a stickiness score to WhaleProfile behavioural parameters.

    Linear interpolation across three score bands:
    - Low   [0.10, 0.40): fast exit, narrow hysteresis — mercenary-like
    - Mid   [0.40, 0.70): moderate stickiness
    - High  [0.70, 1.00]: slow exit, wide hysteresis — institutional/sticky

    Returns dict with keys:
        exit_delay_days     — reaction time before exiting
        reentry_delay_days  — time required before re-entering
        hysteresis_band     — APR gap above exit_threshold to force re-entry

    The caller should update a WhaleProfile with these values.
    """
    s = max(0.10, min(1.0, float(score)))

    # Anchor points: (score, exit_delay, reentry_delay, hysteresis)
    # Low: score=0.10
    # High: score=1.00
    LOW_SCORE, HIGH_SCORE = 0.10, 1.00
    LOW_EXIT, HIGH_EXIT = 1.0, 14.0          # days
    LOW_REENTRY, HIGH_REENTRY = 3.0, 21.0   # days
    LOW_HYST, HIGH_HYST = 0.002, 0.012      # APR decimal

    frac = (s - LOW_SCORE) / (HIGH_SCORE - LOW_SCORE)
    exit_delay = LOW_EXIT + frac * (HIGH_EXIT - LOW_EXIT)
    reentry_delay = LOW_REENTRY + frac * (HIGH_REENTRY - LOW_REENTRY)
    hysteresis = LOW_HYST + frac * (HIGH_HYST - LOW_HYST)

    return {
        "exit_delay_days": round(exit_delay, 2),
        "reentry_delay_days": round(reentry_delay, 2),
        "hysteresis_band": round(hysteresis, 5),
    }


def build_stickiness_enrichment(pool_id: str) -> dict[str, dict] | None:
    """
    Build per-address stickiness parameter dicts from cached whale flow data.

    Combines build_whale_history_lookup() + compute_whale_stickiness_score()
    + stickiness_to_profile_params() into a single convenience function.

    Returns:
        {address_lower: {"score": float, "exit_delay_days": float,
                          "reentry_delay_days": float, "hysteresis_band": float}}
        or None if no cached Dune data is available.
    """
    history = build_whale_history_lookup(pool_id)
    if not history:
        return None

    enrichment: dict[str, dict] = {}
    for addr, hist in history.items():
        score = compute_whale_stickiness_score(hist)
        params = stickiness_to_profile_params(score)
        enrichment[addr] = {"score": score, **params}

    return enrichment if enrichment else None


# ============================================================================
# TVL STICKINESS MODEL
# ============================================================================


@dataclass
class TVLStickinessModel:
    """
    Empirical TVL stickiness model derived from historical deposit/withdrawal data.

    Used to set organic-TVL floor and calibrate RetailDepositorConfig parameters.

    Attributes:
        venue:                       Pool / venue name.
        organic_tvl_estimate_usd:    Average TVL during incentive-free windows (USD).
                                     If no such windows exist, estimated conservatively
                                     as sticky_fraction × peak_tvl.
        incentive_elasticity:        Sensitivity of TVL to incentive rate changes
                                     (dTVL/dRate ÷ TVL, dimensionless fraction).
                                     0 = no response, 1 = perfectly elastic.
        mean_exit_lag_days:          Average days between incentive drop and TVL exit.
        tvl_half_life_post_incentive_days:
                                     Time for incentive TVL to halve after incentives end.
        sticky_fraction:             Fraction of incentive TVL that persists long-term.
        n_observations:              Number of daily snapshots used for estimation.
    """

    venue: str
    organic_tvl_estimate_usd: float = 0.0
    incentive_elasticity: float = 0.5          # 0 = sticky, 1 = fully elastic
    mean_exit_lag_days: float = 7.0
    tvl_half_life_post_incentive_days: float = 14.0
    sticky_fraction: float = 0.40
    n_observations: int = 0


def compute_tvl_stickiness(
    tvl_snapshots: list[dict],
    venue: str = "unknown",
    incentive_rate_key: str = "incentive_rate",
    tvl_key: str = "tvl_usd",
    date_key: str = "date",
    zero_incentive_threshold: float = 0.001,
) -> TVLStickinessModel:
    """
    Estimate TVL stickiness from a historical time series.

    Expects a list of daily records:
        [{"date": "2024-01-01", "tvl_usd": 10_000_000, "incentive_rate": 0.04}, ...]

    Algorithm:
    1. Separate periods with zero incentives vs incentivized periods.
    2. Organic TVL estimate = mean TVL in zero-incentive windows.
       If no zero-incentive windows exist, use min TVL as a conservative floor.
    3. Incentive elasticity = stddev(TVL) / mean(TVL) during incentivized windows
       (a rough proxy for rate-driven volatility).
    4. sticky_fraction = organic_tvl / mean_incentivized_tvl (clamped to [0.05, 0.95]).
    5. Exit lag = day of first TVL drop > 5 % after an incentive cutoff event.
       Averaged across all such events.
    6. Half-life = estimated from the decay speed after incentive cutoffs.

    Returns:
        TVLStickinessModel populated with estimates (or conservative defaults if
        insufficient data).
    """
    if not tvl_snapshots:
        return TVLStickinessModel(venue=venue)

    # Sort by date
    try:
        records = sorted(tvl_snapshots, key=lambda r: r.get(date_key, ""))
    except Exception:
        records = list(tvl_snapshots)

    tvls = [max(float(r.get(tvl_key, 0)), 0.0) for r in records]
    rates = [max(float(r.get(incentive_rate_key, 0)), 0.0) for r in records]
    n = len(records)

    if n == 0:
        return TVLStickinessModel(venue=venue)

    # --- Organic TVL (zero-incentive windows) ---
    organic_tvls = [tvls[i] for i in range(n) if rates[i] <= zero_incentive_threshold]
    mean_inc_tvl_all = sum(tvls) / n if n else 0.0

    if organic_tvls:
        organic_tvl = sum(organic_tvls) / len(organic_tvls)
    else:
        # No zero-incentive data; use min TVL as conservative floor
        organic_tvl = min(tvls) if tvls else 0.0

    # --- Incentive-period TVL stats (for elasticity) ---
    inc_tvls = [tvls[i] for i in range(n) if rates[i] > zero_incentive_threshold]
    if len(inc_tvls) >= 2:
        mean_inc = sum(inc_tvls) / len(inc_tvls)
        if mean_inc > 0:
            variance = sum((t - mean_inc) ** 2 for t in inc_tvls) / len(inc_tvls)
            std_inc = variance**0.5
            elasticity = min(std_inc / mean_inc, 1.0)
        else:
            elasticity = 0.5
    else:
        elasticity = 0.5

    # --- sticky_fraction ---
    mean_all_tvl = mean_inc_tvl_all if mean_inc_tvl_all > 0 else (organic_tvl or 1.0)
    sticky_fraction = max(0.05, min(0.95, organic_tvl / mean_all_tvl))

    # --- Exit lag + half-life: detect incentive cutoff events ---
    exit_lags: list[float] = []
    half_lives: list[float] = []

    for i in range(1, n - 1):
        prev_rate = rates[i - 1]
        curr_rate = rates[i]
        # Incentive cutoff: rate drops from > threshold to <= threshold
        if prev_rate > zero_incentive_threshold and curr_rate <= zero_incentive_threshold:
            tvl_at_cutoff = tvls[i]
            if tvl_at_cutoff <= 0:
                continue
            # Find first subsequent day with >5% TVL drop
            for j in range(i + 1, min(i + 30, n)):
                drop = (tvl_at_cutoff - tvls[j]) / tvl_at_cutoff
                if drop > 0.05:
                    exit_lags.append(float(j - i))
                    break
            # Estimate half-life: days until TVL is 50% of cutoff value
            half_tvl = tvl_at_cutoff * 0.5
            for j in range(i + 1, n):
                if tvls[j] <= half_tvl:
                    half_lives.append(float(j - i))
                    break

    mean_exit_lag = sum(exit_lags) / len(exit_lags) if exit_lags else 7.0
    mean_half_life = sum(half_lives) / len(half_lives) if half_lives else 14.0

    # Clamp to reasonable ranges
    mean_exit_lag = max(1.0, min(mean_exit_lag, 60.0))
    mean_half_life = max(2.0, min(mean_half_life, 90.0))

    return TVLStickinessModel(
        venue=venue,
        organic_tvl_estimate_usd=organic_tvl,
        incentive_elasticity=round(elasticity, 4),
        mean_exit_lag_days=round(mean_exit_lag, 2),
        tvl_half_life_post_incentive_days=round(mean_half_life, 2),
        sticky_fraction=round(sticky_fraction, 4),
        n_observations=n,
    )

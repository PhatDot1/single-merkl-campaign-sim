"""
Historical data fetching and calibration for campaign simulation.

Sources:
- DeFiLlama: Pool-level TVL and APY history (/chart/{poolId})
- Dune Analytics: Whale deposit/withdrawal history for EVM protocols

Calibration outputs:
- alpha (inflow/outflow elasticity) from TVL ~ APR regressions
- sigma (TVL noise) from residual volatility
- r_threshold trend from competitor APR history
- Whale behavior patterns from on-chain flow history

Usage:
    from campaign.historical import (
        fetch_pool_history,
        calibrate_retail_params,
        fetch_whale_history_dune,
    )
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import requests

# ============================================================================
# DEFILLAMA HISTORICAL DATA
# ============================================================================

DEFILLAMA_CHART_URL = "https://yields.llama.fi/chart/{pool_id}"
DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"


@dataclass
class PoolHistoryPoint:
    """Single daily data point from DeFiLlama pool history."""

    timestamp: str  # ISO date string (YYYY-MM-DD)
    tvl_usd: float
    apy: float  # decimal (0.05 = 5%)
    apy_base: float  # base APY (excluding rewards)
    apy_reward: float  # reward APY (incentives)
    il7d: float  # 7d impermanent loss (for DEX pools)


@dataclass
class PoolHistory:
    """Historical data for a DeFiLlama pool."""

    pool_id: str
    project: str
    symbol: str
    chain: str
    points: list[PoolHistoryPoint]

    @property
    def tvl_array(self) -> np.ndarray:
        return np.array([p.tvl_usd for p in self.points])

    @property
    def apy_array(self) -> np.ndarray:
        return np.array([p.apy for p in self.points])

    @property
    def apy_base_array(self) -> np.ndarray:
        return np.array([p.apy_base for p in self.points])

    @property
    def apy_reward_array(self) -> np.ndarray:
        return np.array([p.apy_reward for p in self.points])

    @property
    def days(self) -> int:
        return len(self.points)


def fetch_pool_history(
    pool_id: str,
    days: int = 90,
) -> PoolHistory:
    """
    Fetch historical TVL and APY data for a DeFiLlama pool.

    Args:
        pool_id: DeFiLlama pool UUID (e.g., "747c1d2a-c668-4682-b9f9-296708a3dd90")
        days: Number of days of history to fetch (max ~365 available)

    Returns:
        PoolHistory with daily data points

    Raises:
        RuntimeError if fetch fails or pool not found
    """
    url = DEFILLAMA_CHART_URL.format(pool_id=pool_id)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"DeFiLlama chart fetch failed for pool {pool_id}: {e}")

    raw_data = data.get("data", [])
    if not raw_data:
        raise RuntimeError(
            f"No historical data returned for DeFiLlama pool {pool_id}. "
            f"Check that the pool UUID is correct."
        )

    # Parse and convert to data points
    points = []
    for entry in raw_data:
        try:
            ts = entry.get("timestamp", "")
            tvl = float(entry.get("tvlUsd", 0) or 0)
            # DeFiLlama returns APY as percentage (3.5 = 3.5%)
            apy_pct = float(entry.get("apy", 0) or 0)
            apy_base_pct = float(entry.get("apyBase", 0) or 0)
            apy_reward_pct = float(entry.get("apyReward", 0) or 0)
            il7d = float(entry.get("il7d", 0) or 0)

            points.append(
                PoolHistoryPoint(
                    timestamp=ts[:10] if ts else "",
                    tvl_usd=tvl,
                    apy=apy_pct / 100.0,
                    apy_base=apy_base_pct / 100.0,
                    apy_reward=apy_reward_pct / 100.0,
                    il7d=il7d,
                )
            )
        except (ValueError, TypeError):
            continue

    # Trim to requested window
    if len(points) > days:
        points = points[-days:]

    # Extract metadata from the pool list (optional enrichment)
    project = ""
    symbol = ""
    chain = ""
    try:
        pools_resp = requests.get(DEFILLAMA_POOLS_URL, timeout=15)
        if pools_resp.ok:
            for p in pools_resp.json().get("data", []):
                if p.get("pool") == pool_id:
                    project = p.get("project", "")
                    symbol = p.get("symbol", "")
                    chain = p.get("chain", "")
                    break
    except Exception:
        pass  # Metadata enrichment is optional

    return PoolHistory(
        pool_id=pool_id,
        project=project,
        symbol=symbol,
        chain=chain,
        points=points,
    )


# ============================================================================
# CALIBRATION: Retail Depositor Parameters from History
# ============================================================================


@dataclass
class CalibrationResult:
    """Results from historical calibration of simulation parameters."""

    alpha_plus: float  # Inflow elasticity (APR gap → TVL growth rate)
    alpha_minus: float  # Outflow elasticity (negative gap → TVL decline rate)
    alpha_minus_multiplier: float  # alpha_minus / alpha_plus
    diffusion_sigma: float  # Daily TVL noise (residual volatility)
    response_lag_days: float  # Estimated depositor response lag
    r_threshold_mean: float  # Mean competitor rate over the period
    r_threshold_trend: float  # Linear trend in competitor rate (per day)
    data_quality: str  # "good", "sparse", "insufficient"
    n_observations: int
    details: dict = field(default_factory=dict)


def calibrate_retail_params(
    history: PoolHistory,
    competitor_apy_history: np.ndarray | None = None,
    default_r_threshold: float = 0.045,
) -> CalibrationResult:
    """
    Calibrate retail depositor parameters from historical pool data.

    Method:
    1. Compute daily TVL returns: dTVL/TVL per day
    2. Compute APR gap: pool APY - competitor APY (or default)
    3. Regress TVL returns on APR gap (asymmetric: positive vs negative gap)
    4. Residual volatility → diffusion_sigma
    5. Lag detection via cross-correlation

    Args:
        history: Pool history from fetch_pool_history()
        competitor_apy_history: Array of competitor APYs (same length as history)
        default_r_threshold: Fallback if no competitor data

    Returns:
        CalibrationResult with estimated parameters
    """
    tvl = history.tvl_array
    apy = history.apy_array
    n = len(tvl)

    # ── Data quality check ──
    if n < 14:
        return CalibrationResult(
            alpha_plus=0.15,
            alpha_minus=0.45,
            alpha_minus_multiplier=3.0,
            diffusion_sigma=0.008,
            response_lag_days=5.0,
            r_threshold_mean=default_r_threshold,
            r_threshold_trend=0.0,
            data_quality="insufficient",
            n_observations=n,
            details={"error": f"Only {n} data points, need at least 14"},
        )

    # ── Compute daily TVL returns ──
    # Filter out zero-TVL days to avoid division issues
    valid = tvl > 1000  # At least $1k TVL
    if valid.sum() < 14:
        return CalibrationResult(
            alpha_plus=0.15,
            alpha_minus=0.45,
            alpha_minus_multiplier=3.0,
            diffusion_sigma=0.008,
            response_lag_days=5.0,
            r_threshold_mean=default_r_threshold,
            r_threshold_trend=0.0,
            data_quality="sparse",
            n_observations=int(valid.sum()),
            details={"error": "Too many zero-TVL days"},
        )

    tvl_returns = np.diff(tvl[valid]) / tvl[valid][:-1]  # Daily fractional change
    apy_aligned = apy[valid][:-1]  # APY on day before the return

    # ── Competitor rates ──
    if competitor_apy_history is not None and len(competitor_apy_history) == n:
        r_thresh = competitor_apy_history[valid][:-1]
    else:
        r_thresh = np.full(len(apy_aligned), default_r_threshold)

    r_threshold_mean = float(np.mean(r_thresh))
    r_threshold_trend = 0.0
    if len(r_thresh) > 7:
        days = np.arange(len(r_thresh))
        try:
            coeffs = np.polyfit(days, r_thresh, 1)
            r_threshold_trend = float(coeffs[0])  # change per day
        except Exception:
            pass

    # ── APR gap → TVL return regression (asymmetric) ──
    apr_gap = apy_aligned - r_thresh

    # Positive gap (inflows expected)
    pos_mask = apr_gap > 0
    neg_mask = apr_gap < 0

    # Simple OLS: tvl_return = alpha * apr_gap + epsilon
    alpha_plus = 0.15  # default
    alpha_minus = 0.45  # default
    if pos_mask.sum() > 3:
        try:
            alpha_plus = float(np.mean(tvl_returns[pos_mask] / apr_gap[pos_mask]))
            alpha_plus = np.clip(alpha_plus, 0.01, 1.0)
        except Exception:
            pass
    if neg_mask.sum() > 3:
        try:
            # alpha_minus: how fast TVL leaves when APR is below threshold
            alpha_minus = float(np.mean(tvl_returns[neg_mask] / apr_gap[neg_mask]))
            alpha_minus = np.clip(alpha_minus, 0.01, 3.0)
        except Exception:
            pass

    alpha_minus_mult = alpha_minus / max(alpha_plus, 0.01)
    alpha_minus_mult = np.clip(alpha_minus_mult, 1.0, 10.0)

    # ── Residual volatility → diffusion_sigma ──
    predicted_returns = np.where(apr_gap > 0, alpha_plus * apr_gap, -alpha_minus * np.abs(apr_gap))
    residuals = tvl_returns - predicted_returns
    diffusion_sigma = float(np.std(residuals))
    diffusion_sigma = np.clip(diffusion_sigma, 0.001, 0.05)

    # ── Response lag detection via cross-correlation ──
    response_lag_days = 5.0  # default
    if n > 21:
        try:
            max_lag = min(14, n // 3)
            best_corr = 0.0
            for lag in range(1, max_lag + 1):
                gap_shifted = apr_gap[:-lag] if lag < len(apr_gap) else apr_gap
                returns_shifted = tvl_returns[lag:] if lag < len(tvl_returns) else tvl_returns
                min_len = min(len(gap_shifted), len(returns_shifted))
                if min_len > 5:
                    corr = np.corrcoef(gap_shifted[:min_len], returns_shifted[:min_len])[0, 1]
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        response_lag_days = float(lag)
        except Exception:
            pass

    data_quality = "good" if n >= 30 else "sparse"

    return CalibrationResult(
        alpha_plus=float(alpha_plus),
        alpha_minus=float(alpha_minus),
        alpha_minus_multiplier=float(alpha_minus_mult),
        diffusion_sigma=float(diffusion_sigma),
        response_lag_days=float(response_lag_days),
        r_threshold_mean=float(r_threshold_mean),
        r_threshold_trend=float(r_threshold_trend),
        data_quality=data_quality,
        n_observations=n,
        details={
            "pos_gap_samples": int(pos_mask.sum()),
            "neg_gap_samples": int(neg_mask.sum()),
            "mean_tvl": float(np.mean(tvl[valid])),
            "tvl_change_pct": float((tvl[valid][-1] / tvl[valid][0] - 1) * 100)
            if tvl[valid][0] > 0
            else 0,
        },
    )


# ============================================================================
# DUNE ANALYTICS: Whale Deposit/Withdrawal History
# ============================================================================

DUNE_API_BASE = "https://api.dune.com/api/v1"


def _get_dune_api_key() -> str:
    """Get Dune API key from environment."""
    key = os.environ.get("DUNE_API_KEY")
    if not key:
        raise RuntimeError(
            "DUNE_API_KEY not set in environment. "
            "Set it in your .env file. "
            "Get a key at https://dune.com/settings/api"
        )
    return key


@dataclass
class WhaleFlowEvent:
    """Single deposit or withdrawal event for a whale address."""

    timestamp: str  # ISO datetime
    address: str
    amount_usd: float
    direction: str  # "deposit" or "withdrawal"
    tx_hash: str = ""
    protocol: str = ""


@dataclass
class WhaleHistoryResult:
    """Whale deposit/withdrawal history for a venue."""

    venue_name: str
    token_address: str
    events: list[WhaleFlowEvent]
    total_deposits_usd: float = 0.0
    total_withdrawals_usd: float = 0.0
    unique_whales: int = 0
    data_source: str = "dune"

    @property
    def net_flow_usd(self) -> float:
        return self.total_deposits_usd - self.total_withdrawals_usd


def fetch_whale_history_dune(
    token_address: str,
    min_amount_usd: float = 500_000,
    days: int = 90,
    chain: str = "ethereum",
    venue_name: str = "",
) -> WhaleHistoryResult:
    """
    Fetch whale deposit/withdrawal history from Dune Analytics.

    Uses a parameterized query to get large ERC20 transfers for aToken/eToken
    addresses, which represent deposits and withdrawals.

    Args:
        token_address: aToken/eToken/vault share token address
        min_amount_usd: Minimum transfer size to consider (default $500K)
        days: Lookback period in days
        chain: Blockchain network
        venue_name: Display name for the venue

    Returns:
        WhaleHistoryResult with categorized flow events
    """
    api_key = _get_dune_api_key()
    headers = {"X-Dune-API-Key": api_key}

    # Use Dune's SQL query execution API
    # Query: large ERC20 transfers to/from the token address
    query_sql = f"""
    SELECT
        block_time,
        "from" as from_address,
        "to" as to_address,
        CAST(value AS DOUBLE) / POWER(10, 6) as amount,  -- assume 6 decimals for stables
        tx_hash
    FROM {chain}.transfers
    WHERE contract_address = LOWER('{token_address}')
      AND block_time >= NOW() - INTERVAL '{days}' DAY
      AND CAST(value AS DOUBLE) / POWER(10, 6) >= {min_amount_usd}
    ORDER BY block_time DESC
    LIMIT 1000
    """

    try:
        # Step 1: Execute query
        exec_resp = requests.post(
            f"{DUNE_API_BASE}/query/execute",
            headers=headers,
            json={
                "query_parameters": {},
                "query": query_sql,
            },
            timeout=30,
        )

        if exec_resp.status_code == 402:
            raise RuntimeError("Dune API quota exceeded. Try again later or upgrade your plan.")

        exec_data = exec_resp.json()
        execution_id = exec_data.get("execution_id")
        if not execution_id:
            # Try the newer /query/execute/inline endpoint
            exec_resp = requests.post(
                f"{DUNE_API_BASE}/query/execute/inline",
                headers=headers,
                json={"query_sql": query_sql},
                timeout=30,
            )
            exec_data = exec_resp.json()
            execution_id = exec_data.get("execution_id")

        if not execution_id:
            raise RuntimeError(f"Dune query execution failed: {exec_data}")

        # Step 2: Poll for results
        for attempt in range(30):
            time.sleep(2)
            status_resp = requests.get(
                f"{DUNE_API_BASE}/execution/{execution_id}/status",
                headers=headers,
                timeout=15,
            )
            status = status_resp.json().get("state", "")
            if status == "QUERY_STATE_COMPLETED":
                break
            elif status in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
                raise RuntimeError(f"Dune query failed with state: {status}")

        # Step 3: Fetch results
        results_resp = requests.get(
            f"{DUNE_API_BASE}/execution/{execution_id}/results",
            headers=headers,
            timeout=30,
        )
        results_data = results_resp.json()
        rows = results_data.get("result", {}).get("rows", [])

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Dune API fetch failed: {e}")

    # Parse results into flow events
    zero_addr = "0x0000000000000000000000000000000000000000"
    events = []
    for row in rows:
        from_addr = (row.get("from_address") or "").lower()
        to_addr = (row.get("to_address") or "").lower()
        amount = float(row.get("amount", 0))
        ts = row.get("block_time", "")
        tx = row.get("tx_hash", "")

        # Determine direction:
        # For aTokens/eTokens: mint (from 0x0) = deposit, burn (to 0x0) = withdrawal
        # For transfers between addresses: we track the TOKEN movements
        if from_addr == zero_addr:
            # Mint = someone deposited into the protocol
            events.append(
                WhaleFlowEvent(
                    timestamp=ts,
                    address=to_addr,
                    amount_usd=amount,
                    direction="deposit",
                    tx_hash=tx,
                    protocol=venue_name,
                )
            )
        elif to_addr == zero_addr:
            # Burn = someone withdrew from the protocol
            events.append(
                WhaleFlowEvent(
                    timestamp=ts,
                    address=from_addr,
                    amount_usd=amount,
                    direction="withdrawal",
                    tx_hash=tx,
                    protocol=venue_name,
                )
            )
        else:
            # Transfer between addresses — track as withdrawal from sender
            events.append(
                WhaleFlowEvent(
                    timestamp=ts,
                    address=from_addr,
                    amount_usd=amount,
                    direction="withdrawal",
                    tx_hash=tx,
                    protocol=venue_name,
                )
            )
            events.append(
                WhaleFlowEvent(
                    timestamp=ts,
                    address=to_addr,
                    amount_usd=amount,
                    direction="deposit",
                    tx_hash=tx,
                    protocol=venue_name,
                )
            )

    total_dep = sum(e.amount_usd for e in events if e.direction == "deposit")
    total_wd = sum(e.amount_usd for e in events if e.direction == "withdrawal")
    unique = len(set(e.address for e in events))

    return WhaleHistoryResult(
        venue_name=venue_name,
        token_address=token_address,
        events=events,
        total_deposits_usd=total_dep,
        total_withdrawals_usd=total_wd,
        unique_whales=unique,
        data_source="dune",
    )


# ============================================================================
# CONVENIENCE: Fetch and calibrate for a venue
# ============================================================================


def fetch_and_calibrate(
    defillama_pool_id: str,
    days: int = 90,
    default_r_threshold: float = 0.045,
) -> CalibrationResult:
    """
    End-to-end: fetch DeFiLlama history → calibrate parameters.

    Args:
        defillama_pool_id: DeFiLlama pool UUID
        days: Lookback period
        default_r_threshold: Fallback competitor rate

    Returns:
        CalibrationResult with calibrated retail parameters
    """
    history = fetch_pool_history(defillama_pool_id, days=days)
    return calibrate_retail_params(history, default_r_threshold=default_r_threshold)


def lookup_defillama_pool_id(
    project: str,
    symbol_contains: str,
    chain: str = "Ethereum",
) -> str | None:
    """
    Look up a DeFiLlama pool UUID by project name and symbol.

    Args:
        project: DeFiLlama project name (e.g., "aave-v3", "euler-v2")
        symbol_contains: Substring to match in pool symbol (e.g., "PYUSD")
        chain: Blockchain name (e.g., "Ethereum", "Solana")

    Returns:
        Pool UUID string, or None if not found
    """
    try:
        resp = requests.get(DEFILLAMA_POOLS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])

        matches = [
            p
            for p in pools
            if p.get("project", "").lower() == project.lower()
            and symbol_contains.upper() in (p.get("symbol") or "").upper()
            and p.get("chain", "").lower() == chain.lower()
        ]

        if not matches:
            return None

        # Return the largest pool by TVL
        matches.sort(key=lambda p: float(p.get("tvlUsd", 0) or 0), reverse=True)
        return matches[0].get("pool")

    except Exception:
        return None

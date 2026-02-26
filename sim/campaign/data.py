"""
Data fetching and calibration for campaign optimizer.

Pulls on-chain data via:
- Euler: EVault ABI calls + Euler price API + subgraph for depositor positions
- Morpho: MetaMorpho ABI calls + Morpho GraphQL API
- DeFiLlama: competitor APR/APY data

Produces calibrated WhaleProfile, RetailDepositorConfig, and CampaignEnvironment
objects ready for simulation.

Dependencies: web3/viem calls are done via lightweight HTTP (requests).
No web3.py dependency — uses raw JSON-RPC multicall encoding.

Competitor exclusion: uses venue_registry to exclude ALL our own venues
from competitor rate queries, preventing our own Euler/Curve/Aave/Kamino
pools from appearing as "competitors" to each other.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import requests

from .agents import MercenaryConfig, RetailDepositorConfig, WhaleProfile
from .engine import LossWeights
from .optimizer import SurfaceGrid
from .state import CampaignEnvironment

# ============================================================================
# CONSTANTS
# ============================================================================


# Ethereum JSON-RPC — loaded from environment, no hardcoded public endpoints
def _get_rpc_url() -> str:
    url = os.environ.get("ALCHEMY_ETH_RPC_URL")
    if not url:
        raise RuntimeError(
            "ALCHEMY_ETH_RPC_URL not set in environment. "
            "Set it in your .env file or environment variables."
        )
    return url


DEFAULT_RPC_URL = os.environ.get("ALCHEMY_ETH_RPC_URL", "https://eth.llamarpc.com")

EULER_PRICE_API = "https://app.euler.finance/api/v1/price"
MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"
DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"

# Known stablecoin addresses (Ethereum mainnet)
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
PYUSD_ADDRESS = "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8"
RLUSD_ADDRESS = "0x8292Bb45bf1Ee4d140127049757C0C38e47a8A75"

# ERC20 Transfer event topic
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# Known Morpho V1 -> V2 vault mappings for depositor position lookup.
# The V1 vault is what campaigns target; V2 exposes positions via GraphQL.
MORPHO_V1_TO_V2 = {
    "0x19b3cd7032b8c062e8d44eacad661a0970dd8c55": "0xb576765fb15505433af24fee2c0325895c559fb2",
}


# ============================================================================
# LOW-LEVEL RPC HELPERS
# ============================================================================


def eth_call(rpc_url: str, to: str, data: str, block: str = "latest") -> str:
    """Raw eth_call."""
    resp = requests.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": to, "data": data}, block],
            "id": 1,
        },
        timeout=30,
    )
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"eth_call error: {result['error']}")
    return result["result"]


def eth_get_logs(
    rpc_url: str,
    address: str,
    topics: list[str | None],
    from_block: str = "0x0",
    to_block: str = "latest",
) -> list[dict]:
    """Raw eth_getLogs."""
    resp = requests.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [
                {
                    "address": address,
                    "topics": topics,
                    "fromBlock": from_block,
                    "toBlock": to_block,
                }
            ],
            "id": 1,
        },
        timeout=60,
    )
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"eth_getLogs error: {result['error']}")
    return result["result"]


def decode_uint256(hex_str: str) -> int:
    """Decode a uint256 from hex."""
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    return int(hex_str, 16) if hex_str else 0


def decode_address(hex_str: str) -> str:
    """Decode an address from a 32-byte hex word."""
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    return "0x" + hex_str[-40:]


def encode_uint256(val: int) -> str:
    """Encode uint256 as 32-byte hex (no 0x prefix)."""
    return hex(val)[2:].zfill(64)


# ============================================================================
# VAULT DATA FETCHERS
# ============================================================================


@dataclass
class VaultSnapshot:
    """Raw on-chain snapshot of a vault's state."""

    address: str
    chain_id: int
    asset_address: str
    asset_decimals: int
    asset_symbol: str
    asset_price_usd: float

    total_supply_assets: int  # raw units
    total_borrows: int  # raw units
    cash: int  # raw units
    supply_cap: int  # raw units (resolved)
    borrow_cap: int  # raw units (resolved)

    # Depositor data
    top_depositors: list[dict]  # [{address, balance_raw, balance_usd}]

    # Timestamp
    timestamp: int = 0

    @property
    def total_supply_usd(self) -> float:
        return self.total_supply_assets / (10**self.asset_decimals) * self.asset_price_usd

    @property
    def total_borrows_usd(self) -> float:
        return self.total_borrows / (10**self.asset_decimals) * self.asset_price_usd

    @property
    def cash_usd(self) -> float:
        return self.cash / (10**self.asset_decimals) * self.asset_price_usd

    @property
    def utilization(self) -> float:
        total = self.cash + self.total_borrows
        if total == 0:
            return 0.0
        return self.total_borrows / total

    @property
    def idle_fraction(self) -> float:
        if self.total_supply_assets == 0:
            return 0.0
        return self.cash / self.total_supply_assets

    @property
    def supply_cap_utilization(self) -> float:
        if self.supply_cap == 0:
            return 0.0
        return self.total_supply_assets / self.supply_cap

    def whale_concentration(self, top_n: int = 5) -> dict:
        """Compute whale concentration metrics."""
        total = self.total_supply_usd
        if total == 0 or not self.top_depositors:
            return {"top_1_pct": 0, "top_5_pct": 0, "hhi": 0, "top_depositors": []}

        sorted_deps = sorted(self.top_depositors, key=lambda d: d["balance_usd"], reverse=True)
        top_1 = sorted_deps[0]["balance_usd"] / total if sorted_deps else 0
        top_5 = sum(d["balance_usd"] for d in sorted_deps[:top_n]) / total
        hhi = sum((d["balance_usd"] / total) ** 2 for d in sorted_deps if d["balance_usd"] > 0)

        return {
            "top_1_pct": top_1,
            "top_5_pct": top_5,
            "hhi": hhi,
            "top_depositors": sorted_deps[:top_n],
        }


# ============================================================================
# MORPHO GRAPHQL HELPERS
# ============================================================================


def _morpho_gql_post(query: str, variables: dict) -> dict:
    """Post to Morpho GraphQL, raising on errors."""
    resp = requests.post(
        MORPHO_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"Morpho GraphQL errors: {body['errors']}")
    return body.get("data", {})


def _fetch_morpho_v2_positions(v2_address: str, chain_id: int) -> dict:
    """
    Fetch V2 (MorphoVaultV2) depositor positions via GraphQL.

    This is the reliable way to get top depositors for Morpho vaults.
    The V2 vault exposes positions directly through the Morpho API,
    matching the query used in morpho_vault_analysis_current.py.
    """
    query = """
    query($address: String!, $chainId: Int!) {
      vaultV2ByAddress(address: $address, chainId: $chainId) {
        totalAssets
        avgApy
        avgNetApy
        performanceFee
        positions(first: 100) {
          items {
            user { address }
            assets
          }
        }
      }
    }
    """
    data = _morpho_gql_post(query, {"address": v2_address.lower(), "chainId": chain_id})
    vault = data.get("vaultV2ByAddress")
    if not vault:
        raise RuntimeError(
            f"Morpho GraphQL returned no data for V2 vault {v2_address}. "
            f"Check the address and chainId={chain_id}."
        )
    return vault


def _fetch_morpho_v1_positions(vault_address: str, chain_id: int) -> list[dict]:
    """
    Try to fetch V1 (MetaMorpho) depositor positions via GraphQL.

    Some V1 vaults may expose positions directly. Returns empty list on failure.
    """
    query = """
    query($address: String!, $chainId: Int!) {
      vaultByAddress(address: $address, chainId: $chainId) {
        state { totalAssets }
        positions(first: 100) {
          items {
            user { address }
            assets
            shares
          }
        }
      }
    }
    """
    try:
        data = _morpho_gql_post(query, {"address": vault_address.lower(), "chainId": chain_id})
        vault = data.get("vaultByAddress")
        if vault and vault.get("positions", {}).get("items"):
            return vault["positions"]["items"]
    except Exception:
        pass
    return []


def _fetch_morpho_depositors(
    vault_address: str,
    chain_id: int,
    asset_decimals: int,
    asset_price: float,
    total_supply_usd: float,
) -> list[dict]:
    """
    Fetch Morpho depositor positions using all available paths.

    Strategy:
    1. Check V1->V2 mapping, use V2 positions API (most reliable)
    2. Try V1 positions query directly
    3. FAIL with clear error if neither works

    Returns list of dicts [{address, balance_raw, balance_usd}]
    """
    divisor = 10**asset_decimals
    depositors = []

    # ── Path 1: Known V2 counterpart ──
    v2_addr = MORPHO_V1_TO_V2.get(vault_address.lower())
    if v2_addr:
        print(f"  Fetching depositors from V2 vault {v2_addr[:10]}...")
        try:
            v2_data = _fetch_morpho_v2_positions(v2_addr, chain_id)
            v2_positions = v2_data.get("positions", {}).get("items", [])
            v2_total = int(v2_data.get("totalAssets") or 0) / divisor

            for pos in v2_positions:
                raw = int(pos.get("assets") or 0)
                usd = raw / divisor * asset_price
                if usd >= 100_000:  # Only meaningful positions
                    depositors.append(
                        {
                            "address": pos["user"]["address"],
                            "balance_raw": raw,
                            "balance_usd": usd,
                        }
                    )

            print(
                f"  Found {len(depositors)} depositors >= $100k from V2 vault (total V2 TVL: ${v2_total:,.0f})"
            )

            if v2_total > 0 and total_supply_usd > 0:
                ratio = v2_total / (total_supply_usd / asset_price)
                if abs(ratio - 1.0) > 0.5:
                    print(
                        f"  Note: V2 TVL=${v2_total:,.0f} vs V1 TVL=${total_supply_usd:,.0f} — "
                        f"using V1 TVL for simulation, V2 for concentration"
                    )
        except Exception as e:
            print(f"  WARNING: V2 position fetch failed: {e}")

    # ── Path 2: V1 positions directly ──
    if not depositors:
        print("  Trying V1 positions query...")
        try:
            v1_positions = _fetch_morpho_v1_positions(vault_address, chain_id)
            for pos in v1_positions:
                raw = int(pos.get("assets") or pos.get("shares") or 0)
                usd = raw / divisor * asset_price
                if usd >= 100_000:
                    depositors.append(
                        {
                            "address": pos["user"]["address"],
                            "balance_raw": raw,
                            "balance_usd": usd,
                        }
                    )
            if depositors:
                print(f"  Found {len(depositors)} depositors >= $100k from V1 positions")
        except Exception as e:
            print(f"  WARNING: V1 position fetch failed: {e}")

    # ── Path 3: FAIL ──
    if not depositors:
        raise RuntimeError(
            f"FAILED to fetch depositor positions for Morpho vault {vault_address}. "
            f"Tried V2 vault {v2_addr or 'N/A'} and V1 positions query. "
            f"Cannot build whale profiles without depositor data. "
            f"Either add a V1->V2 mapping to MORPHO_V1_TO_V2 in data.py, "
            f"or use --venue with a static config that has manual whale profiles."
        )

    depositors.sort(key=lambda d: d["balance_usd"], reverse=True)
    return depositors


# ============================================================================
# PROTOCOL-SPECIFIC VAULT FETCHERS
# ============================================================================


def fetch_euler_vault_snapshot(
    vault_address: str,
    chain_id: int = 1,
    rpc_url: str = DEFAULT_RPC_URL,
    n_top_depositors: int = 10,
) -> VaultSnapshot:
    """
    Fetch complete snapshot of an Euler vault.

    Reads on-chain: cash, totalBorrows, caps, totalSupplyAssets
    Reads off-chain: Euler price API
    Reads logs: ERC20 Transfer events to reconstruct depositor balances
    """
    # ── On-chain reads ──
    CASH_SEL = "0xe0a65f58"  # cash()
    TOTAL_BORROWS_SEL = "0x47bd3718"  # totalBorrows()
    CAPS_SEL = "0x18442059"  # caps()
    TOTAL_SUPPLY_SEL = "0x18160ddd"  # totalSupply() (ERC20)
    ASSET_SEL = "0x38d52e0f"  # asset()
    DECIMALS_SEL = "0x313ce567"  # decimals()

    cash_raw = decode_uint256(eth_call(rpc_url, vault_address, CASH_SEL))
    borrows_raw = decode_uint256(eth_call(rpc_url, vault_address, TOTAL_BORROWS_SEL))
    caps_result = eth_call(rpc_url, vault_address, CAPS_SEL)
    # caps returns (uint16 supplyCap, uint16 borrowCap)
    caps_hex = caps_result[2:]  # strip 0x
    supply_cap_encoded = decode_uint256("0x" + caps_hex[:64])
    borrow_cap_encoded = decode_uint256("0x" + caps_hex[64:128])

    _total_supply_raw = decode_uint256(eth_call(rpc_url, vault_address, TOTAL_SUPPLY_SEL))
    asset_addr = decode_address(eth_call(rpc_url, vault_address, ASSET_SEL))
    asset_decimals = decode_uint256(eth_call(rpc_url, asset_addr, DECIMALS_SEL))

    # Resolve Euler-style encoded caps
    supply_cap = _resolve_euler_cap(supply_cap_encoded)
    borrow_cap = _resolve_euler_cap(borrow_cap_encoded)

    # ── Price from Euler API ──
    price_data = _fetch_euler_prices(chain_id)
    asset_price = price_data.get(asset_addr.lower(), {}).get("price", 1.0)
    asset_symbol = price_data.get(asset_addr.lower(), {}).get("symbol", "UNKNOWN")

    # ── Depositor balances from Transfer events ──
    depositors = _fetch_erc20_depositors(
        rpc_url, vault_address, asset_decimals, asset_price, n_top_depositors
    )

    if not depositors:
        raise RuntimeError(
            f"FAILED to fetch depositors for Euler vault {vault_address}. "
            f"ERC20 Transfer log scan returned no results. "
            f"Euler vaults require a subgraph or your evaultTransferInEuler DB. "
            f"Use --venue with a static config that has manual whale profiles."
        )

    return VaultSnapshot(
        address=vault_address,
        chain_id=chain_id,
        asset_address=asset_addr,
        asset_decimals=asset_decimals,
        asset_symbol=asset_symbol,
        asset_price_usd=asset_price,
        total_supply_assets=cash_raw + borrows_raw,  # totalAssets = cash + borrows
        total_borrows=borrows_raw,
        cash=cash_raw,
        supply_cap=supply_cap,
        borrow_cap=borrow_cap,
        top_depositors=depositors,
        timestamp=int(time.time()),
    )


def fetch_morpho_vault_snapshot(
    vault_address: str,
    chain_id: int = 1,
    rpc_url: str = DEFAULT_RPC_URL,
    n_top_depositors: int = 20,
) -> VaultSnapshot:
    """
    Fetch complete snapshot of a Morpho MetaMorpho vault.

    Uses Morpho GraphQL for:
    - V1 vault state (allocation, totalAssets)
    - Depositor positions (V2 positions API or V1 positions query)

    Does NOT use ERC20 Transfer log scan (unreliable on public RPCs).
    """
    # ── V1 state via GraphQL ──
    v1_query = """
    query($address: String!, $chainId: Int!) {
      vaultByAddress(address: $address, chainId: $chainId) {
        state {
          totalAssets
          allocation {
            market {
              uniqueKey
              lltv
              loanAsset { name decimals address priceUsd }
              collateralAsset { name decimals address priceUsd }
              state { borrowAssets supplyAssets supplyShares }
            }
            supplyAssets
            supplyShares
            supplyCap
          }
        }
      }
    }
    """
    v1_data = _morpho_gql_post(v1_query, {"address": vault_address.lower(), "chainId": chain_id})
    vault_obj = v1_data.get("vaultByAddress")
    if not vault_obj or not vault_obj.get("state"):
        raise RuntimeError(
            f"Morpho GraphQL returned no data for V1 vault {vault_address}. "
            f"Check the address and chainId={chain_id}."
        )

    allocation_data = vault_obj["state"]["allocation"]

    total_supply_assets = 0
    total_borrows = 0
    cash = 0
    total_supply_cap = 0

    asset_price = 1.0
    asset_symbol = "UNKNOWN"
    asset_decimals = 6
    asset_address = ""

    for alloc in allocation_data:
        market = alloc["market"]
        supply = int(market["state"]["supplyAssets"])
        borrow = int(market["state"]["borrowAssets"])
        cap_raw = alloc.get("supplyCap")
        cap = int(cap_raw) if cap_raw and int(cap_raw) < 10**30 else 0
        total_supply_assets += supply
        total_borrows += borrow
        cash += supply - borrow
        total_supply_cap += cap

        loan_asset = market.get("loanAsset") or {}
        if loan_asset:
            asset_price = float(loan_asset.get("priceUsd") or 1.0)
            asset_symbol = loan_asset.get("name") or "UNKNOWN"
            asset_decimals = int(loan_asset.get("decimals") or 6)
            asset_address = loan_asset.get("address") or ""

    # Use totalAssets from the vault directly (more accurate than summing allocations)
    v1_total_raw = int(vault_obj["state"]["totalAssets"])
    if v1_total_raw > 0:
        total_supply_assets = v1_total_raw

    total_supply_usd = total_supply_assets / (10**asset_decimals) * asset_price

    # ── Depositor positions via GraphQL (NOT ERC20 log scan) ──
    depositors = _fetch_morpho_depositors(
        vault_address, chain_id, asset_decimals, asset_price, total_supply_usd
    )

    return VaultSnapshot(
        address=vault_address,
        chain_id=chain_id,
        asset_address=asset_address,
        asset_decimals=asset_decimals,
        asset_symbol=asset_symbol,
        asset_price_usd=asset_price,
        total_supply_assets=total_supply_assets,
        total_borrows=total_borrows,
        cash=max(0, cash),
        supply_cap=total_supply_cap,
        borrow_cap=0,
        top_depositors=depositors[:n_top_depositors],
        timestamp=int(time.time()),
    )


# ============================================================================
# COMPETITOR RATE FETCHING
# ============================================================================


@dataclass
class CompetitorRate:
    """APR/APY data for a competing venue."""

    venue: str
    pool_id: str
    symbol: str
    tvl_usd: float
    apy_base: float  # Native APY as DECIMAL (0.035 = 3.5%)
    apy_reward: float  # Incentive APY as DECIMAL
    apy_total: float  # Total APY as DECIMAL


def fetch_competitor_rates(
    asset_symbol: str = "PYUSD",
    min_tvl: float = 1_000_000,
    exclude_pool_ids: list[str] | None = None,
    exclude_vault_address: str | None = None,
    exclude_all_own_venues: bool = True,
) -> list[CompetitorRate]:
    """
    Fetch competitor lending/vault rates from DeFiLlama Yields API.

    IMPORTANT: DeFiLlama returns APY as PERCENTAGE (e.g., 3.5 means 3.5%).
    We convert to DECIMAL here (3.5 -> 0.035) so all internal usage is consistent.

    When exclude_all_own_venues=True (default), automatically excludes ALL
    venues in our venue_registry that track this asset. This prevents our
    own Euler, Curve, Aave, Kamino, etc. pools from appearing as
    "competitors" to each other.

    Filtering:
    - exclude_pool_ids: skip specific DeFiLlama pool IDs
    - exclude_vault_address: skip pools whose pool ID contains this address
      (prevents including the venue being analyzed as its own competitor)
    - exclude_all_own_venues: skip ALL our own venues for this asset
    - Pools with apy_total < 0.1% are excluded (dust/inactive)
    - Duplicate entries for the same venue (e.g., multiple merkl entries
      for the same underlying vault) are deduplicated by keeping the
      highest-APY entry per (project, approximate TVL) pair.
    """
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])
    except Exception as e:
        raise RuntimeError(
            f"DeFiLlama competitor rate fetch failed: {e}. "
            f"Cannot calibrate r_threshold without competitor data."
        )

    exclude_ids = set(exclude_pool_ids or [])
    exclude_addr = exclude_vault_address.lower() if exclude_vault_address else None

    # Build set of ALL our own venue addresses to exclude
    own_addresses: set[str] = set()
    if exclude_all_own_venues:
        try:
            from .venue_registry import get_all_venue_addresses

            own_addresses = get_all_venue_addresses(asset_symbol)
            print(f"  Excluding {len(own_addresses)} own venue addresses from competitors")
        except ImportError:
            print(
                "  WARNING: venue_registry not available, using single exclude_vault_address only"
            )

    raw = []
    for pool in pools:
        symbol = pool.get("symbol", "")
        if asset_symbol.upper() not in symbol.upper():
            continue
        tvl = pool.get("tvlUsd", 0)
        if tvl < min_tvl:
            continue

        pool_id = pool.get("pool", "")

        # Skip excluded pools
        if pool_id in exclude_ids:
            continue
        # Skip pools containing the vault address being analyzed
        if exclude_addr and exclude_addr in pool_id.lower():
            continue

        # Skip pools that match ANY of our own venue addresses
        if own_addresses:
            pool_id_lower = pool_id.lower()
            pool_underlying = [t.lower() for t in (pool.get("underlyingTokens") or [])]
            is_own = False
            for addr in own_addresses:
                if addr in pool_id_lower:
                    is_own = True
                    break
                if any(addr in t for t in pool_underlying):
                    is_own = True
                    break
            if is_own:
                continue

        # DeFiLlama returns percentage (3.5 = 3.5%), convert to decimal (0.035)
        apy_base_pct = pool.get("apyBase") or 0
        apy_reward_pct = pool.get("apyReward") or 0
        apy_base = apy_base_pct / 100.0
        apy_reward = apy_reward_pct / 100.0
        apy_total = apy_base + apy_reward

        # Skip dust/inactive pools (< 0.1% total APY)
        if apy_total < 0.001:
            continue

        raw.append(
            CompetitorRate(
                venue=pool.get("project", "unknown"),
                pool_id=pool_id,
                symbol=symbol,
                tvl_usd=tvl,
                apy_base=apy_base,
                apy_reward=apy_reward,
                apy_total=apy_total,
            )
        )

    # Deduplicate: for pools with very similar TVL from the same project,
    # keep only the highest-APY entry (DeFiLlama often lists multiple
    # merkl reward entries for the same underlying vault)
    seen: dict[str, CompetitorRate] = {}
    for c in sorted(raw, key=lambda x: -x.apy_total):
        # Key: project + TVL bucket (within 5% = same underlying pool)
        tvl_bucket = round(c.tvl_usd / max(c.tvl_usd * 0.05, 1_000_000))
        dedup_key = f"{c.venue}_{tvl_bucket}"
        if dedup_key not in seen:
            seen[dedup_key] = c

    competitors = sorted(seen.values(), key=lambda c: c.tvl_usd, reverse=True)

    if not competitors:
        print(
            f"  ⚠ No competitors found for {asset_symbol} with TVL >= ${min_tvl:,.0f} "
            f"(after filtering {len(own_addresses)} own venues). "
            f"compute_r_threshold will fall back to stablecoin benchmark."
        )

    return competitors


def fetch_usdc_benchmark(
    min_tvl: float = 10_000_000,
    chain: str = "Ethereum",
) -> float:
    """
    Fetch a USDC stablecoin benchmark rate from DeFiLlama.

    Uses organic yield only (apyBase, no rewards) from large USDC lending
    pools on Ethereum.  Excludes pools from projects in our venue_registry
    to avoid circular references.

    Returns TVL-weighted average of apyBase as a decimal (0.035 = 3.5%).
    """
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])
    except Exception as e:
        raise RuntimeError(f"DeFiLlama fetch failed for USDC benchmark: {e}")

    # Build set of our own venue project slugs to exclude
    own_projects: set[str] = set()
    try:
        from .venue_registry import VENUE_REGISTRY

        for rec in VENUE_REGISTRY.values():
            if rec.defillama_project:
                own_projects.add(rec.defillama_project.lower())
    except ImportError:
        pass

    candidates = []
    for pool in pools:
        symbol = pool.get("symbol", "")
        if "USDC" not in symbol.upper():
            continue
        pool_chain = (pool.get("chain") or "").lower()
        if pool_chain != chain.lower():
            continue
        tvl = pool.get("tvlUsd", 0)
        if tvl < min_tvl:
            continue
        project = (pool.get("project") or "unknown").lower()
        if project in own_projects:
            continue

        # Organic yield only — no reward APY
        apy_base_pct = pool.get("apyBase") or 0
        apy_base = apy_base_pct / 100.0
        if apy_base < 0.001:
            continue

        candidates.append({"project": project, "tvl": tvl, "apy_base": apy_base})

    if not candidates:
        print("  ⚠ No USDC benchmark pools found, using 3.5% hardcoded fallback")
        return 0.035

    candidates.sort(key=lambda x: -x["tvl"])
    top = candidates[:20]
    total_tvl = sum(c["tvl"] for c in top)
    benchmark = sum(c["apy_base"] * c["tvl"] for c in top) / total_tvl

    print(
        f"  USDC benchmark: {benchmark:.2%} "
        f"(TVL-weighted apyBase from {len(top)} {chain} pools, "
        f"total TVL ${total_tvl / 1e6:,.0f}M)"
    )
    return benchmark


def compute_r_threshold(
    competitors: list[CompetitorRate],
    cap: float = 0.08,
    stablecoin_benchmark: float | None = None,
) -> dict:
    """
    Compute r_threshold from competitor rates with robustness.

    Edge-case handling:
    - 0 competitors → use stablecoin benchmark (auto-fetched if not provided)
    - < 3 competitors or single outlier dominates → blend 50/50 with benchmark
    - >= 3 competitors → pure TVL-weighted peer average
    - Outliers (> 3× median) are filtered before averaging
    - Final value capped at `cap` (default 8%)

    All CompetitorRate.apy_total values are already DECIMAL (0.035 = 3.5%).

    Args:
        competitors: list of CompetitorRate (may be empty)
        cap: maximum r_threshold (default 0.08 = 8%)
        stablecoin_benchmark: pre-fetched USDC benchmark; auto-fetched if None

    Returns dict with r_threshold, r_threshold_lo, r_threshold_hi,
    r_threshold_source, and diagnostic fields.
    """
    # Lazy-fetch USDC benchmark if not provided
    if stablecoin_benchmark is None:
        try:
            stablecoin_benchmark = fetch_usdc_benchmark()
        except RuntimeError:
            stablecoin_benchmark = 0.035  # absolute last resort

    # ── Case 1: Zero competitors ──
    if not competitors:
        r_threshold = min(stablecoin_benchmark, cap)
        print(f"  r_threshold: {r_threshold:.2%} (stablecoin_benchmark — no asset peers found)")
        return {
            "r_threshold": r_threshold,
            "r_threshold_lo": r_threshold * 0.8,
            "r_threshold_hi": min(r_threshold * 1.2, cap),
            "r_threshold_source": "stablecoin_benchmark",
            "n_peers_raw": 0,
            "n_peers_after_outlier_filter": 0,
            "usdc_benchmark": stablecoin_benchmark,
        }

    # ── Outlier removal ──
    rates = sorted([c.apy_total for c in competitors])
    median_rate = float(np.median(rates))
    outlier_cutoff = median_rate * 3.0

    filtered = [c for c in competitors if c.apy_total <= outlier_cutoff]
    n_removed = len(competitors) - len(filtered)
    if n_removed > 0:
        print(
            f"  Outlier filter: removed {n_removed} competitor(s) "
            f"with APY > {outlier_cutoff:.2%} (3× median {median_rate:.2%})"
        )

    # Compute peer TVL-weighted average on filtered set
    if filtered:
        peer_tvl = sum(c.tvl_usd for c in filtered)
        peer_rates = [c.apy_total for c in filtered]
        if peer_tvl > 0:
            peer_weighted_avg = sum(c.apy_total * c.tvl_usd for c in filtered) / peer_tvl
        else:
            peer_weighted_avg = float(np.mean(peer_rates))
    else:
        # All competitors were outliers — treat as zero-peer case
        r_threshold = min(stablecoin_benchmark, cap)
        print(
            f"  r_threshold: {r_threshold:.2%} "
            f"(all {len(competitors)} peers were outliers, using benchmark)"
        )
        return {
            "r_threshold": r_threshold,
            "r_threshold_lo": r_threshold * 0.8,
            "r_threshold_hi": min(r_threshold * 1.2, cap),
            "r_threshold_source": "stablecoin_benchmark",
            "n_peers_raw": len(competitors),
            "n_peers_after_outlier_filter": 0,
            "outliers_removed": n_removed,
            "usdc_benchmark": stablecoin_benchmark,
        }

    # ── Case 2: Few competitors (< 3) → blend with benchmark ──
    if len(filtered) < 3:
        r_threshold = 0.5 * peer_weighted_avg + 0.5 * stablecoin_benchmark
        source = "blended"
        print(
            f"  r_threshold: {r_threshold:.2%} "
            f"(blended — {len(filtered)} peers @ {peer_weighted_avg:.2%} "
            f"+ USDC benchmark @ {stablecoin_benchmark:.2%})"
        )
    # ── Case 3: Enough competitors → pure peer average ──
    else:
        r_threshold = peer_weighted_avg
        source = "asset_peers"
        print(
            f"  r_threshold: {r_threshold:.2%} "
            f"(asset_peers — {len(filtered)} venues, "
            f"TVL-weighted avg)"
        )

    r_threshold = min(r_threshold, cap)

    all_rates = [c.apy_total for c in filtered]
    return {
        "r_threshold": r_threshold,
        "r_threshold_lo": min(all_rates),
        "r_threshold_hi": min(max(all_rates), cap),
        "r_threshold_source": source,
        "n_peers_raw": len(competitors),
        "n_peers_after_outlier_filter": len(filtered),
        "outliers_removed": n_removed,
        "peer_weighted_avg": peer_weighted_avg,
        "usdc_benchmark": stablecoin_benchmark,
    }


def fetch_stablecoin_class_benchmark(
    proxy_assets: tuple[str, ...] = ("USDC", "USDT", "DAI", "USDS"),
    min_pool_tvl: float = 50_000_000,
    friction_discount: float = 0.005,
    top_n: int = 20,
) -> dict:
    """
    Compute a TVL-weighted stablecoin class benchmark rate.

    For assets like RLUSD that have few direct competitors, we estimate
    r_threshold by looking at the best rates for similar stablecoins
    (USDC/USDT/DAI/USDS) in large lending pools, then apply a friction
    discount for the swap/bridge cost of moving between the assets.

    Returns dict with r_threshold, source breakdown, and metadata.
    """
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])
    except Exception as e:
        raise RuntimeError(f"DeFiLlama fetch failed for stablecoin benchmark: {e}")

    # Collect top pools for each proxy asset
    per_asset: dict[str, list[dict]] = {a: [] for a in proxy_assets}
    for pool in pools:
        symbol = pool.get("symbol", "")
        tvl = pool.get("tvlUsd", 0)
        if tvl < min_pool_tvl:
            continue
        chain = (pool.get("chain") or "").lower()
        if chain not in ("ethereum", "solana", "arbitrum", "base", "polygon"):
            continue

        apy_base_pct = pool.get("apyBase") or 0
        apy_reward_pct = pool.get("apyReward") or 0
        apy_total = (apy_base_pct + apy_reward_pct) / 100.0
        if apy_total < 0.001:
            continue

        for asset in proxy_assets:
            if asset.upper() in symbol.upper():
                per_asset[asset].append(
                    {
                        "project": pool.get("project", "unknown"),
                        "pool_id": pool.get("pool", ""),
                        "symbol": symbol,
                        "tvl": tvl,
                        "apy_total": apy_total,
                    }
                )
                break

    # Keep top N pools per asset by TVL
    all_pools = []
    asset_breakdown = {}
    for asset in proxy_assets:
        sorted_pools = sorted(per_asset[asset], key=lambda x: -x["tvl"])[:top_n]
        if sorted_pools:
            total_tvl = sum(p["tvl"] for p in sorted_pools)
            weighted_rate = sum(p["apy_total"] * p["tvl"] for p in sorted_pools) / total_tvl
            asset_breakdown[asset] = {
                "n_pools": len(sorted_pools),
                "total_tvl": total_tvl,
                "weighted_rate": weighted_rate,
            }
            all_pools.extend(sorted_pools)

    if not all_pools:
        raise RuntimeError(
            f"No stablecoin pools found for {proxy_assets} with TVL >= ${min_pool_tvl:,.0f}"
        )

    # Overall TVL-weighted average across all proxy assets
    grand_tvl = sum(p["tvl"] for p in all_pools)
    grand_rate = sum(p["apy_total"] * p["tvl"] for p in all_pools) / grand_tvl

    # Apply friction discount (swap/bridge cost)
    benchmark = max(0.005, grand_rate - friction_discount)

    return {
        "r_threshold": benchmark,
        "raw_benchmark": grand_rate,
        "friction_discount": friction_discount,
        "n_pools": len(all_pools),
        "total_tvl_sampled": grand_tvl,
        "per_asset": asset_breakdown,
        "source": "stablecoin_class_benchmark",
    }


# ============================================================================
# CALIBRATION: Raw Data -> Simulation Parameters
# ============================================================================


@dataclass
class CalibratedVenueParams:
    """
    Fully calibrated parameters for one venue, ready for simulation.

    This is the output of the data pipeline and the input to optimize_surface().
    """

    venue_name: str
    initial_tvl: float
    whale_profiles: list[WhaleProfile]
    retail_config: RetailDepositorConfig
    mercenary_config: MercenaryConfig
    env: CampaignEnvironment
    weights: LossWeights
    grid: SurfaceGrid

    # Raw data for reference
    snapshot: VaultSnapshot | None = None
    competitors: list[CompetitorRate] = field(default_factory=list)


def calibrate_from_snapshot(
    snapshot: VaultSnapshot,
    competitors: list[CompetitorRate],
    # User inputs (the only things you need to specify)
    weekly_budget_range: tuple[float, float] = (100_000, 250_000),
    tvl_target: float | None = None,
    apr_target: float | None = None,
    apr_ceiling: float = 0.10,
    budget_steps: int = 16,
    r_max_steps: int = 17,
    r_max_range: tuple[float, float] = (0.04, 0.12),
) -> CalibratedVenueParams:
    """
    Calibrate all simulation parameters from on-chain data.

    User provides:
    - weekly_budget_range: min/max budget to explore
    - tvl_target: desired TVL (defaults to current)
    - apr_target: desired APR center (defaults to current competitor avg)
    - apr_ceiling: hard APR ceiling

    Everything else is derived from data.
    """
    conc = snapshot.whale_concentration()
    r_thresh = compute_r_threshold(competitors)

    # Log competitor landscape
    print(f"\n  Competitor landscape ({len(competitors)} venues):")
    for c in competitors[:8]:
        print(
            f"    {c.venue:<25} {c.symbol:<20} TVL=${c.tvl_usd / 1e6:>8.1f}M  APY={c.apy_total:.2%}"
        )
    source = r_thresh.get("r_threshold_source", "unknown")
    print(f"  r_threshold (source={source}): {r_thresh['r_threshold']:.2%}")
    print(
        f"  r_threshold range: [{r_thresh['r_threshold_lo']:.2%}, {r_thresh['r_threshold_hi']:.2%}]"
    )

    # ── Whale Profiles ──
    whale_profiles = _build_whale_profiles(snapshot, conc, r_thresh["r_threshold"], competitors)

    # ── Retail Config ──
    alpha_plus = _estimate_alpha_plus(snapshot)
    sigma = _estimate_sigma(snapshot)

    retail_config = RetailDepositorConfig(
        alpha_plus=alpha_plus,
        alpha_minus_multiplier=3.0,  # Conservative assumption
        response_lag_days=5.0,
        diffusion_sigma=sigma,
    )

    # ── Mercenary Config ──
    # apy_total is already decimal — no /100 needed
    top_rate = max((c.apy_total for c in competitors), default=0.06)
    mercenary_config = MercenaryConfig(
        entry_threshold=max(top_rate * 1.3, 0.08),  # 30% above top competitor
        exit_threshold=max(top_rate * 0.9, 0.05),
        max_capital_usd=snapshot.total_supply_usd * 0.10,  # Max 10% of TVL
        entry_rate_per_day=0.3,
        exit_rate_per_day=0.5,
    )

    # ── Environment ──
    # apy_total is already decimal — no /100 needed
    env = CampaignEnvironment(
        competitor_rates={c.venue: c.apy_total for c in competitors[:10]},
        r_threshold=r_thresh["r_threshold"],
        r_threshold_lo=r_thresh["r_threshold_lo"],
        r_threshold_hi=r_thresh["r_threshold_hi"],
    )

    # ── Loss Weights ──
    _tvl_target = tvl_target or snapshot.total_supply_usd
    _apr_target = apr_target or r_thresh["r_threshold"]

    weights = LossWeights(
        w_spend=1.0,
        w_spend_waste_penalty=2.0,
        w_apr_variance=3.0,
        w_apr_ceiling=5.0,
        w_tvl_shortfall=8.0,
        w_mercenary=6.0,
        w_whale_proximity=6.0,
        w_apr_floor=7.0,
        apr_target=_apr_target,
        apr_ceiling=apr_ceiling,
        tvl_target=_tvl_target,
        apr_stability_on_total=True,
    )

    # ── Grid ──
    grid = SurfaceGrid.from_ranges(
        B_min=weekly_budget_range[0],
        B_max=weekly_budget_range[1],
        B_steps=budget_steps,
        r_max_min=r_max_range[0],
        r_max_max=r_max_range[1],
        r_max_steps=r_max_steps,
    )

    return CalibratedVenueParams(
        venue_name=f"{snapshot.asset_symbol}_{snapshot.address[:10]}",
        initial_tvl=snapshot.total_supply_usd,
        whale_profiles=whale_profiles,
        retail_config=retail_config,
        mercenary_config=mercenary_config,
        env=env,
        weights=weights,
        grid=grid,
        snapshot=snapshot,
        competitors=competitors,
    )


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _resolve_euler_cap(encoded: int) -> int:
    """
    Resolve Euler-style encoded cap.
    Matches: resolveCap() from your TypeScript.
    """
    if encoded == 0:
        return 2**256 - 1  # Max uint256

    exponent = encoded & 63
    mantissa = encoded >> 6
    return (10**exponent * mantissa) // 100


def _fetch_euler_prices(chain_id: int) -> dict:
    """Fetch prices from Euler API."""
    try:
        resp = requests.get(f"{EULER_PRICE_API}?chainId={chain_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Remove non-price fields
        for key in ["countryCode", "isProxyOrVpn", "is_vpn"]:
            data.pop(key, None)
        # Normalize addresses to lowercase for lookup
        return {k.lower(): v for k, v in data.items()}
    except Exception as e:
        print(f"Warning: Euler price fetch failed: {e}")
        return {}


def _fetch_morpho_allocation(vault_address: str, chain_id: int) -> list[dict]:
    """Fetch allocation data from Morpho GraphQL (legacy helper, kept for compatibility)."""
    query = """
    query($address: String!, $chainId: Int!) {
      vaultByAddress(address: $address, chainId: $chainId) {
        state {
          allocation {
            market {
              uniqueKey
              lltv
              loanAsset { name decimals address priceUsd }
              collateralAsset { name decimals address priceUsd }
              state { borrowAssets supplyAssets supplyShares }
            }
            supplyAssets
            supplyShares
            supplyCap
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            MORPHO_GRAPHQL_URL,
            json={"query": query, "variables": {"address": vault_address, "chainId": chain_id}},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("vaultByAddress", {}).get("state", {}).get("allocation", [])
    except Exception as e:
        print(f"Warning: Morpho GraphQL fetch failed: {e}")
        return []


def _fetch_erc20_depositors(
    rpc_url: str,
    token_address: str,
    decimals: int,
    price_usd: float,
    top_n: int,
    from_block: str = "0x0",
) -> list[dict]:
    """
    Reconstruct top depositor balances from ERC20 Transfer events.

    NOTE: For production, use a subgraph or indexed DB instead of
    scanning all Transfer events. This is a reference implementation.
    For Euler, your DB already has evaultTransferInEuler.
    For Morpho, use the Morpho subgraph or events from the vault contract.
    """
    try:
        logs = eth_get_logs(
            rpc_url,
            token_address,
            [TRANSFER_TOPIC],
            from_block=from_block,
        )
    except Exception as e:
        print(f"Warning: Transfer log fetch failed (may need subgraph): {e}")
        return []

    balances: dict[str, int] = {}
    for log_entry in logs:
        topics = log_entry.get("topics", [])
        if len(topics) < 3:
            continue
        from_addr = decode_address(topics[1])
        to_addr = decode_address(topics[2])
        value = decode_uint256(log_entry.get("data", "0x0"))

        # Zero address = mint
        if from_addr != "0x" + "0" * 40:
            balances[from_addr] = balances.get(from_addr, 0) - value
        if to_addr != "0x" + "0" * 40:
            balances[to_addr] = balances.get(to_addr, 0) + value

    # Filter positive, sort, take top N
    positive = [
        {"address": addr, "balance_raw": bal, "balance_usd": bal / (10**decimals) * price_usd}
        for addr, bal in balances.items()
        if bal > 0
    ]
    positive.sort(key=lambda d: d["balance_raw"], reverse=True)
    return positive[:top_n]


def _build_whale_profiles(
    snapshot: VaultSnapshot,
    concentration: dict,
    r_threshold: float,
    competitors: list[CompetitorRate],
) -> list[WhaleProfile]:
    """
    Build WhaleProfile objects from depositor data.

    Estimates exit thresholds based on:
    - Competitor rates (alt_rate)
    - Position size (switching costs scale sublinearly)
    - Heuristic whale type classification
    """
    top_deps = concentration.get("top_depositors", [])
    if not top_deps:
        raise RuntimeError(
            "No top depositors found in concentration data. "
            "Cannot build whale profiles without depositor data. "
            "Check that the vault fetcher returned depositors."
        )

    # apy_total is already decimal — no /100 needed
    best_alt = max((c.apy_total for c in competitors), default=r_threshold)

    total_usd = snapshot.total_supply_usd
    profiles = []
    for i, dep in enumerate(top_deps):
        position = dep["balance_usd"]
        if position < 1_000_000:  # Skip sub-$1M depositors
            continue

        share = position / total_usd if total_usd > 0 else 0

        # Classify whale type heuristically
        if share > 0.10:
            whale_type = "institutional"
            exit_delay = 3.0
            reentry_delay = 10.0
            hysteresis = 0.008
        elif share > 0.05:
            whale_type = "quant_desk"
            exit_delay = 2.0
            reentry_delay = 7.0
            hysteresis = 0.006
        else:
            whale_type = "opportunistic"
            exit_delay = 1.0
            reentry_delay = 4.0
            hysteresis = 0.005

        # Switching cost: gas + slippage, scales with sqrt(position)
        switching_cost = 500 + (position / 10_000_000) * 100

        # Alt rate: jitter around best competitor rate
        alt_rate = best_alt * (0.85 + 0.25 * (i / max(len(top_deps), 1)))
        alt_rate = min(alt_rate, r_threshold + 0.02)

        # Risk premium: smaller for larger, more sophisticated depositors
        risk_premium = 0.003 + 0.001 * min(i, 5)

        addr = dep.get("address", f"unknown_{i}")
        profiles.append(
            WhaleProfile(
                whale_id=f"whale_{i + 1}_{addr[:8]}",
                position_usd=position,
                alt_rate=alt_rate,
                risk_premium=risk_premium,
                switching_cost_usd=switching_cost,
                exit_delay_days=exit_delay,
                reentry_delay_days=reentry_delay,
                hysteresis_band=hysteresis,
                whale_type=whale_type,
            )
        )

    if not profiles:
        print(
            f"  WARNING: No depositors above $1M whale threshold. "
            f"Largest depositor: ${top_deps[0]['balance_usd']:,.0f}"
            if top_deps
            else ""
        )

    return profiles


def _estimate_alpha_plus(snapshot: VaultSnapshot) -> float:
    """
    Estimate inflow elasticity alpha+.

    Without historical TVL time series, use heuristics:
    - Mature markets (high TVL, low idle) -> lower alpha (sticky capital)
    - Growth markets (lower TVL, high idle) -> higher alpha (responsive capital)
    """
    _tvl = snapshot.total_supply_usd
    idle = snapshot.idle_fraction
    conc = snapshot.whale_concentration()
    top5 = conc.get("top_5_pct", 0)

    alpha = 0.3 + 0.3 * idle - 0.2 * top5
    return max(0.1, min(0.8, alpha))


def _estimate_sigma(snapshot: VaultSnapshot) -> float:
    """
    Estimate TVL diffusion volatility sigma.

    Without historical data, derive from whale concentration:
    - High concentration -> higher vol (one exit = big move)
    - Low concentration -> lower vol (smooth)
    """
    conc = snapshot.whale_concentration()
    top5 = conc.get("top_5_pct", 0)
    hhi = conc.get("hhi", 0)

    sigma = 0.01 + 0.03 * top5 + 0.05 * hhi
    return max(0.005, min(0.05, sigma))


# ============================================================================
# PROTOCOL-SPECIFIC FETCHERS (Aave, Curve, Kamino)
# ============================================================================


# ── AAVE ──

AAVE_V3_POOL_ADDRESS = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"  # Ethereum mainnet
AAVE_V3_DATA_PROVIDER = "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3"

AAVE_GET_RESERVE_DATA_SEL = "0x35ea6a75"  # getReserveData(address)
AAVE_GET_RESERVE_CAPS_SEL = "0x46fbe558"  # getReserveCaps(address)


def fetch_aave_vault_snapshot(
    asset_address: str,
    market_name: str = "Core",
    chain_id: int = 1,
    rpc_url: str = DEFAULT_RPC_URL,
    n_top_depositors: int = 10,
    pool_address: str = AAVE_V3_POOL_ADDRESS,
) -> VaultSnapshot:
    """
    Fetch Aave V3 reserve data for a given asset.

    Aave doesn't have "vaults" — it has reserves per asset in a pool.
    We query the Pool contract for reserve data and the aToken for depositors.
    """
    aave_data = _fetch_aave_subgraph(asset_address, chain_id, market_name)

    if not aave_data:
        raise RuntimeError(
            f"FAILED to fetch Aave data for asset {asset_address} on market {market_name}. "
            f"Subgraph and DeFiLlama fallback both returned no data. "
            f"Use --venue with a static config."
        )

    a_token = aave_data.get("aToken", {}).get("id", "")

    depositors = []
    if a_token:
        asset_decimals = aave_data.get("decimals", 6)
        asset_price = aave_data.get("price", 1.0)
        depositors = _fetch_erc20_depositors(
            rpc_url, a_token, asset_decimals, asset_price, n_top_depositors
        )

    if not depositors:
        raise RuntimeError(
            f"FAILED to fetch depositors for Aave asset {asset_address}. "
            f"aToken Transfer log scan returned no results. "
            f"Use --venue with a static config that has manual whale profiles."
        )

    total_liquidity = int(aave_data.get("totalATokenSupply", 0))
    total_borrows = int(aave_data.get("totalCurrentVariableDebt", 0))
    decimals = aave_data.get("decimals", 6)
    price = aave_data.get("price", 1.0)

    return VaultSnapshot(
        address=pool_address,
        chain_id=chain_id,
        asset_address=asset_address,
        asset_decimals=decimals,
        asset_symbol=aave_data.get("symbol", "UNKNOWN"),
        asset_price_usd=price,
        total_supply_assets=total_liquidity,
        total_borrows=total_borrows,
        cash=max(0, total_liquidity - total_borrows),
        supply_cap=int(aave_data.get("supplyCap", 0)) * (10**decimals),
        borrow_cap=int(aave_data.get("borrowCap", 0)) * (10**decimals),
        top_depositors=depositors,
    )


def _fetch_aave_subgraph(asset_address: str, chain_id: int, market_name: str) -> dict | None:
    """Fetch Aave V3 reserve data from the Aave subgraph."""
    subgraph_urls = {
        1: "https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
    }
    url = subgraph_urls.get(chain_id)
    if not url:
        return _fetch_aave_from_defillama(asset_address, chain_id, market_name)

    query = """
    query($asset: String!) {
      reserves(where: {underlyingAsset: $asset}) {
        symbol
        decimals
        aToken { id }
        totalATokenSupply
        totalCurrentVariableDebt
        supplyCap
        borrowCap
        liquidityRate
        variableBorrowRate
        price { priceInEth }
      }
    }
    """
    try:
        resp = requests.post(
            url,
            json={"query": query, "variables": {"asset": asset_address.lower()}},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        reserves = data.get("data", {}).get("reserves", [])
        if reserves:
            reserve = reserves[0]
            reserve["price"] = 1.0  # Stablecoin assumption
            return reserve
    except Exception as e:
        print(f"Warning: Aave subgraph failed: {e}")

    return _fetch_aave_from_defillama(asset_address, chain_id, market_name)


def _fetch_aave_from_defillama(asset_address: str, chain_id: int, market_name: str) -> dict | None:
    """Fallback: get Aave data from DeFiLlama pools API."""
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        pools = resp.json().get("data", [])
        for pool in pools:
            if (
                pool.get("project", "").startswith("aave")
                and pool.get("underlyingTokens")
                and asset_address.lower() in pool["underlyingTokens"][0].lower()
            ):
                return {
                    "symbol": pool.get("symbol", "UNKNOWN"),
                    "decimals": 6,
                    "totalATokenSupply": int(pool.get("tvlUsd", 0) * 1e6),
                    "totalCurrentVariableDebt": 0,
                    "supplyCap": 0,
                    "borrowCap": 0,
                    "price": 1.0,
                    "aToken": {},
                }
    except Exception:
        pass
    return None


# ── CURVE ──


def fetch_curve_pool_snapshot(
    pool_address: str,
    chain_id: int = 1,
    rpc_url: str = DEFAULT_RPC_URL,
    n_top_depositors: int = 10,
) -> VaultSnapshot:
    """Fetch Curve pool data."""
    GET_BALANCES_SEL = "0x14f05979"

    try:
        balances_hex = eth_call(rpc_url, pool_address, GET_BALANCES_SEL)
        hex_data = balances_hex[2:]
        bal_0 = decode_uint256("0x" + hex_data[:64])
        bal_1 = decode_uint256("0x" + hex_data[64:128])
    except Exception:
        bal_0, bal_1 = 0, 0

    LP_TOKEN_SEL = "0xfc0c546a"
    try:
        lp_token = decode_address(eth_call(rpc_url, pool_address, LP_TOKEN_SEL))
    except Exception:
        lp_token = pool_address

    total_tvl_raw = bal_0 + bal_1
    decimals = 6

    depositors = _fetch_erc20_depositors(rpc_url, lp_token, 18, 1.0, n_top_depositors)

    if not depositors:
        raise RuntimeError(
            f"FAILED to fetch depositors for Curve pool {pool_address}. "
            f"LP token Transfer log scan returned no results. "
            f"Use --venue with a static config that has manual whale profiles."
        )

    return VaultSnapshot(
        address=pool_address,
        chain_id=chain_id,
        asset_address=pool_address,
        asset_decimals=decimals,
        asset_symbol="CURVE-LP",
        asset_price_usd=1.0,
        total_supply_assets=total_tvl_raw,
        total_borrows=0,
        cash=total_tvl_raw,
        supply_cap=2**256 - 1,
        borrow_cap=0,
        top_depositors=depositors,
    )


# ── KAMINO (Solana) ──

KAMINO_STRATEGIES_URL = "https://api.kamino.finance/strategies"


def fetch_kamino_snapshot(
    market_name: str = "main",
    asset_symbol: str = "PYUSD",
    strategy_address: str | None = None,
) -> VaultSnapshot:
    """
    Fetch Kamino market/strategy data.

    Kamino is on Solana — no EVM calls. Data sources:
    1. /strategies/{addr}/metrics — for CLMM/Earn vaults with a known address
    2. DeFiLlama Yields API — for lending market reserves

    NOTE: Kamino depositor tracking is limited without a Solana indexer.
    Whale profiles will need to be manually specified or fetched from
    Kamino's analytics dashboard.
    """
    if strategy_address:
        return _fetch_kamino_strategy(strategy_address, asset_symbol)
    return _fetch_kamino_from_defillama(market_name, asset_symbol)


def _fetch_kamino_strategy(strategy_address: str, asset_symbol: str) -> VaultSnapshot:
    """Fetch CLMM/Earn strategy metrics from Kamino API."""
    try:
        resp = requests.get(
            f"{KAMINO_STRATEGIES_URL}/{strategy_address}/metrics",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        tvl_usd = float(data.get("totalValueLocked", 0))
        decimals = 6
        price = 1.0

        return VaultSnapshot(
            address=strategy_address,
            chain_id=0,  # Solana
            asset_address=data.get("tokenAMint", ""),
            asset_decimals=decimals,
            asset_symbol=asset_symbol,
            asset_price_usd=price,
            total_supply_assets=int(tvl_usd * (10**decimals)),
            total_borrows=0,
            cash=int(tvl_usd * (10**decimals)),
            supply_cap=0,
            borrow_cap=0,
            top_depositors=[],  # Not available via REST
            timestamp=int(time.time()),
        )
    except Exception as e:
        raise RuntimeError(
            f"FAILED to fetch Kamino strategy {strategy_address}: {e}. "
            f"Use --venue with a static config."
        )


def _fetch_kamino_from_defillama(market_name: str, asset_symbol: str) -> VaultSnapshot:
    """Fetch Kamino lending market data from DeFiLlama Yields API."""
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])

        project_map = {
            "main": "kamino-lend",
            "earn": "kamino-earn",
            "clmm": "kamino-liquidity",
            "jlp": "kamino-lend",
            "maple": "kamino-lend",
        }
        target_project = project_map.get(market_name, "kamino-lend")

        target = None
        for pool in pools:
            if (
                pool.get("project", "") == target_project
                and asset_symbol.upper() in pool.get("symbol", "").upper()
                and pool.get("chain", "") == "Solana"
            ):
                target = pool
                break

        if not target:
            for pool in pools:
                if (
                    "kamino" in pool.get("project", "")
                    and asset_symbol.upper() in pool.get("symbol", "").upper()
                    and pool.get("chain", "") == "Solana"
                ):
                    target = pool
                    break

        if not target:
            raise RuntimeError(
                f"FAILED: {asset_symbol} not found in Kamino {market_name} on DeFiLlama. "
                f"Use --venue with a static config."
            )

        tvl_usd = target.get("tvlUsd", 0)
        decimals = 6
        price = 1.0
        total_supply = int(tvl_usd * (10**decimals))

        return VaultSnapshot(
            address=f"kamino_{market_name}_{target.get('pool', '')}",
            chain_id=0,  # Solana
            asset_address=target.get("underlyingTokens", [""])[0]
            if target.get("underlyingTokens")
            else "",
            asset_decimals=decimals,
            asset_symbol=asset_symbol,
            asset_price_usd=price,
            total_supply_assets=total_supply,
            total_borrows=0,
            cash=total_supply,
            supply_cap=0,
            borrow_cap=0,
            top_depositors=[],  # Not available from DeFiLlama
            timestamp=int(time.time()),
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"FAILED to fetch Kamino data from DeFiLlama: {e}. Use --venue with a static config."
        )


def _empty_snapshot(address: str, symbol: str) -> VaultSnapshot:
    """Return empty snapshot when data is unavailable."""
    return VaultSnapshot(
        address=address,
        chain_id=0,
        asset_address="",
        asset_decimals=6,
        asset_symbol=symbol,
        asset_price_usd=1.0,
        total_supply_assets=0,
        total_borrows=0,
        cash=0,
        supply_cap=0,
        borrow_cap=0,
        top_depositors=[],
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def fetch_and_calibrate(
    vault_address: str,
    vault_type: str = "euler",  # "euler", "morpho", "aave", "curve", "kamino"
    chain_id: int = 1,
    asset_symbol: str = "PYUSD",
    weekly_budget_range: tuple[float, float] = (100_000, 250_000),
    tvl_target: float | None = None,
    rpc_url: str = DEFAULT_RPC_URL,
    # Aave-specific
    aave_market_name: str = "Core",
    # Kamino-specific
    kamino_market_name: str = "main",
) -> CalibratedVenueParams:
    """
    Full pipeline: fetch on-chain data -> calibrate -> return simulation params.

    Supports: euler, morpho, aave, curve, kamino.

    FAILS LOUDLY if data fetching fails — no silent empty results.
    """
    print(f"Fetching {vault_type} vault {vault_address[:10]}...")

    if vault_type == "euler":
        snapshot = fetch_euler_vault_snapshot(vault_address, chain_id, rpc_url)
    elif vault_type == "morpho":
        snapshot = fetch_morpho_vault_snapshot(vault_address, chain_id, rpc_url)
    elif vault_type == "aave":
        snapshot = fetch_aave_vault_snapshot(vault_address, aave_market_name, chain_id, rpc_url)
    elif vault_type == "curve":
        snapshot = fetch_curve_pool_snapshot(vault_address, chain_id, rpc_url)
    elif vault_type == "kamino":
        snapshot = fetch_kamino_snapshot(kamino_market_name, asset_symbol)
    else:
        raise ValueError(f"Unknown vault type: {vault_type}")

    conc = snapshot.whale_concentration()
    print(f"  TVL: ${snapshot.total_supply_usd:,.0f}")
    print(f"  Utilization: {snapshot.utilization:.1%}")
    print(f"  Idle: {snapshot.idle_fraction:.1%}")
    print(f"  Depositors found: {len(snapshot.top_depositors)}")
    print(f"  Top-1 concentration: {conc['top_1_pct']:.1%}")
    print(f"  Top-5 concentration: {conc['top_5_pct']:.1%}")

    if snapshot.top_depositors:
        total_usd = snapshot.total_supply_usd
        print("  Top depositors:")
        for i, d in enumerate(snapshot.top_depositors[:5]):
            share = d["balance_usd"] / total_usd if total_usd > 0 else 0
            print(f"    #{i + 1}: {d['address'][:10]}... ${d['balance_usd']:>14,.0f} ({share:.1%})")

    print(f"Fetching competitor rates for {asset_symbol}...")
    competitors = fetch_competitor_rates(
        asset_symbol,
        exclude_vault_address=vault_address,
    )
    print(f"  Found {len(competitors)} competing venues")

    print("Calibrating simulation parameters...")
    params = calibrate_from_snapshot(
        snapshot,
        competitors,
        weekly_budget_range=weekly_budget_range,
        tvl_target=tvl_target,
    )

    print(f"\n  Whale profiles: {len(params.whale_profiles)}")
    for wp in params.whale_profiles[:5]:
        print(
            f"    {wp.whale_id}: ${wp.position_usd / 1e6:.1f}M, "
            f"exit_threshold={wp.exit_threshold:.2%}, type={wp.whale_type}"
        )
    print(f"  r_threshold: {params.env.r_threshold:.2%}")
    print(f"  alpha+: {params.retail_config.alpha_plus:.2f}")
    print(f"  sigma: {params.retail_config.diffusion_sigma:.3f}")
    print("Calibration complete.")

    return params

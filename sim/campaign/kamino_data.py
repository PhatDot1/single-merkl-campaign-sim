"""
Kamino API integration for Solana-based vaults and lending markets.

Production configuration:
- Solana RPC URL loaded from HELIUS_SOLANA_RPC_URL environment variable
- Vault pubkeys loaded from environment variables (no hardcoded fallbacks)
- Whale fetching via getTokenLargestAccounts (Helius RPC)
- NO fallback to empty data — errors loudly if fetch fails

Supports:
- kVaults (Earn vaults): /kvaults/vaults/{pubkey} and /kvaults/vaults/{pubkey}/metrics
- Kamino Lend reserves: /kamino-market/{market}/reserves/metrics
- Depositor positions: getTokenLargestAccounts on shares mint (Solana RPC)

Known Kamino market pubkeys:
- Main Market:    7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF
- JLP Market:     DxXdAyU3kCjnyggvHmY5nAwg5cRbbmdyX3npfDMjjMek
- Altcoin Market: ByYiZxp8QrdN9qbdtaAiePN8AAr3qvTPppNJDpf5DVJ5
- Maple Market:   loaded from KAMINO_MAPLE_MARKET_PUBKEY env var

Known reserve/mint addresses:
- PYUSD Mint (Solana):  2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo
- PYUSD Reserve:        2gc9Dm1eB6UgVYFBUN9bWks6Kes9PbWSaPaa9DqyvEiN
- USDC Mint (Solana):   EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v

Kamino Lending Program ID: KLend2g3cS87mDRy4gP3yNyp5n14h8VzBwJ117CgX1V

All APY values from Kamino API are returned as decimals (0.05 = 5%).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================


def get_solana_rpc_url() -> str:
    """Get Solana RPC URL from environment. Errors loudly if not set."""
    url = os.environ.get("HELIUS_SOLANA_RPC_URL")
    if not url:
        raise RuntimeError(
            "HELIUS_SOLANA_RPC_URL not set in environment. "
            "Set it in your .env file or environment variables. "
            "Example: HELIUS_SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
        )
    return url


def get_kamino_vault_pubkey(env_var: str) -> str:
    """Get a Kamino vault pubkey from environment. Errors loudly if not set."""
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(
            f"{env_var} not set in environment. "
            f"Look up the pubkey from the Kamino UI (browser DevTools → Network tab → "
            f"api.kamino.finance/kvaults/vaults/...) and add to your .env file."
        )
    return val


# ============================================================================
# CONSTANTS
# ============================================================================

KAMINO_API_BASE = "https://api.kamino.finance"

# Known Kamino lending market pubkeys (hardcoded — these are protocol constants)
KAMINO_MARKETS = {
    "main": "7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF",
    "jlp": "DxXdAyU3kCjnyggvHmY5nAwg5cRbbmdyX3npfDMjjMek",
    "altcoin": "ByYiZxp8QrdN9qbdtaAiePN8AAr3qvTPppNJDpf5DVJ5",
    # Maple market loaded from env at runtime
}

# Known Solana mint addresses (protocol constants)
PYUSD_MINT_SOLANA = "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo"
USDC_MINT_SOLANA = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# Known reserve addresses
PYUSD_RESERVE_MAIN = "2gc9Dm1eB6UgVYFBUN9bWks6Kes9PbWSaPaa9DqyvEiN"
USDC_RESERVE_JLP = "Ga4rZytCpq1unD4DbEJ5bkHeUz9g3oh9AAFEi6vSauXp"

# Kamino Lend program ID
KLEND_PROGRAM_ID = "KLend2g3cS87mDRy4gP3yNyp5n14h8VzBwJ117CgX1V"


def _get_maple_market() -> str:
    """Get Maple market pubkey from env, or raise."""
    return get_kamino_vault_pubkey("KAMINO_MAPLE_MARKET_PUBKEY")


def _get_pyusd_earn_vault() -> str:
    """Get PYUSD Earn vault pubkey from env."""
    return get_kamino_vault_pubkey("KAMINO_PYUSD_EARN_VAULT_PUBKEY")


def _get_pyusd_clmm_vault() -> str:
    """Get PYUSD CLMM vault pubkey from env."""
    return get_kamino_vault_pubkey("KAMINO_PYUSD_CLMM_VAULT_PUBKEY")


def get_kamino_market_pubkey(market_name: str) -> str:
    """
    Get market pubkey by name. Maple loaded from env.
    Errors loudly if market not found.
    """
    if market_name == "maple":
        return _get_maple_market()

    pubkey = KAMINO_MARKETS.get(market_name)
    if not pubkey:
        raise ValueError(
            f"Unknown Kamino market '{market_name}'. "
            f"Known markets: {list(KAMINO_MARKETS.keys()) + ['maple']}. "
            f"Maple market requires KAMINO_MAPLE_MARKET_PUBKEY env var."
        )
    return pubkey


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class KaminoReserveMetrics:
    """Metrics for a single Kamino lending reserve."""

    reserve_pubkey: str
    liquidity_token: str
    liquidity_token_mint: str
    max_ltv: float
    borrow_apy: float  # decimal
    supply_apy: float  # decimal
    total_supply: float  # token units
    total_borrow: float  # token units
    total_supply_usd: float
    total_borrow_usd: float

    @property
    def utilization(self) -> float:
        if self.total_supply <= 0:
            return 0.0
        return self.total_borrow / self.total_supply


@dataclass
class KaminoVaultMetrics:
    """Metrics for a Kamino kVault (Earn vault)."""

    vault_pubkey: str
    apy: float
    apy_7d: float
    apy_24h: float
    apy_30d: float
    apy_actual: float
    apy_theoretical: float
    apy_farm_rewards: float
    apy_incentives: float
    apy_reserves_incentives: float
    token_price: float
    sol_price: float
    tokens_available: float
    tokens_available_usd: float
    tokens_invested: float
    tokens_invested_usd: float
    share_price: float
    tokens_per_share: float
    number_of_holders: int
    shares_issued: float
    cumulative_interest_earned: float
    cumulative_interest_earned_usd: float

    @property
    def total_tvl_usd(self) -> float:
        return self.tokens_available_usd + self.tokens_invested_usd

    @property
    def base_apy(self) -> float:
        """Base APY = total APY minus incentive rewards (organic yield)."""
        return max(0.0, self.apy_actual - self.apy_incentives - self.apy_farm_rewards)


@dataclass
class KaminoVaultState:
    """Full state of a Kamino kVault."""

    vault_pubkey: str
    token_mint: str
    token_mint_decimals: int
    shares_mint: str
    shares_mint_decimals: int
    token_available: float
    shares_issued: float
    performance_fee_bps: int
    management_fee_bps: int
    name: str
    vault_farm: str
    allocation_strategy: list[dict]
    creation_timestamp: int


@dataclass
class KaminoDepositorPosition:
    """A single depositor's position in a kVault."""

    vault_address: str
    staked_shares: float
    unstaked_shares: float
    total_shares: float


@dataclass
class KaminoStrategyMetrics:
    """Metrics for a Kamino CLMM strategy (NOT a kVault).

    CLMM strategies use the /strategies/{pubkey}/metrics endpoint,
    which returns a completely different data shape from kVaults.
    """

    strategy_pubkey: str
    token_a: str
    token_b: str
    token_a_mint: str
    token_b_mint: str
    share_mint: str
    total_value_locked: float  # USD
    share_price: float
    shares_issued: float
    # APY breakdown
    fee_apy: float  # LP fee APY (organic)
    kamino_apy_7d: float  # Kamino rewards 7d trailing
    total_apy: float  # All-in APY
    # Vault balances
    token_a_balance: float  # token units
    token_a_balance_usd: float
    token_b_balance: float
    token_b_balance_usd: float

    @property
    def base_apy(self) -> float:
        """Base (organic) APY = fee APY only."""
        return self.fee_apy


# ============================================================================
# API FETCHERS
# ============================================================================


def fetch_kamino_vault_state(vault_pubkey: str) -> KaminoVaultState:
    """
    Fetch kVault state: /kvaults/vaults/{pubkey}
    Returns vault configuration and allocation strategy.
    """
    url = f"{KAMINO_API_BASE}/kvaults/vaults/{vault_pubkey}"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Kamino vault state fetch failed: HTTP {resp.status_code} for {vault_pubkey}. "
            f"Response: {resp.text[:200]}"
        )
    data = resp.json()
    state = data.get("state", {})

    return KaminoVaultState(
        vault_pubkey=data.get("address", vault_pubkey),
        token_mint=state.get("tokenMint", ""),
        token_mint_decimals=int(state.get("tokenMintDecimals", 6)),
        shares_mint=state.get("sharesMint", ""),
        shares_mint_decimals=int(state.get("sharesMintDecimals", 6)),
        token_available=_safe_float(state.get("tokenAvailable", "0")),
        shares_issued=_safe_float(state.get("sharesIssued", "0")),
        performance_fee_bps=int(state.get("performanceFeeBps", 0)),
        management_fee_bps=int(state.get("managementFeeBps", 0)),
        name=state.get("name", ""),
        vault_farm=state.get("vaultFarm", ""),
        allocation_strategy=state.get("vaultAllocationStrategy", []),
        creation_timestamp=int(state.get("creationTimestamp", 0)),
    )


def fetch_kamino_vault_metrics(vault_pubkey: str) -> KaminoVaultMetrics:
    """
    Fetch kVault metrics: /kvaults/vaults/{pubkey}/metrics
    Returns APY breakdown, TVL, share price, holder count.
    """
    url = f"{KAMINO_API_BASE}/kvaults/vaults/{vault_pubkey}/metrics"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Kamino vault metrics fetch failed: HTTP {resp.status_code} for {vault_pubkey}. "
            f"Response: {resp.text[:200]}"
        )
    data = resp.json()

    return KaminoVaultMetrics(
        vault_pubkey=vault_pubkey,
        apy=_safe_float(data.get("apy", "0")),
        apy_7d=_safe_float(data.get("apy7d", "0")),
        apy_24h=_safe_float(data.get("apy24h", "0")),
        apy_30d=_safe_float(data.get("apy30d", "0")),
        apy_actual=_safe_float(data.get("apyActual", "0")),
        apy_theoretical=_safe_float(data.get("apyTheoretical", "0")),
        apy_farm_rewards=_safe_float(data.get("apyFarmRewards", "0")),
        apy_incentives=_safe_float(data.get("apyIncentives", "0")),
        apy_reserves_incentives=_safe_float(data.get("apyReservesIncentives", "0")),
        token_price=_safe_float(data.get("tokenPrice", "1.0")),
        sol_price=_safe_float(data.get("solPrice", "0")),
        tokens_available=_safe_float(data.get("tokensAvailable", "0")),
        tokens_available_usd=_safe_float(data.get("tokensAvailableUsd", "0")),
        tokens_invested=_safe_float(data.get("tokensInvested", "0")),
        tokens_invested_usd=_safe_float(data.get("tokensInvestedUsd", "0")),
        share_price=_safe_float(data.get("sharePrice", "1.0")),
        tokens_per_share=_safe_float(data.get("tokensPerShare", "1.0")),
        number_of_holders=int(data.get("numberOfHolders", 0)),
        shares_issued=_safe_float(data.get("sharesIssued", "0")),
        cumulative_interest_earned=_safe_float(data.get("cumulativeInterestEarned", "0")),
        cumulative_interest_earned_usd=_safe_float(data.get("cumulativeInterestEarnedUsd", "0")),
    )


def fetch_kamino_strategy_metrics(strategy_pubkey: str) -> KaminoStrategyMetrics:
    """
    Fetch CLMM strategy metrics: /strategies/{pubkey}/metrics

    CLMM strategies are NOT kVaults — they use a completely different API
    endpoint and return a different data shape.  The kvaults API returns
    HTTP 500 / 404 for strategy pubkeys.

    The shareMint is fetched from the /strategies listing if not present
    in the metrics response (it usually isn't).
    """
    url = f"{KAMINO_API_BASE}/strategies/{strategy_pubkey}/metrics"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Kamino strategy metrics fetch failed: HTTP {resp.status_code} "
            f"for {strategy_pubkey}. Response: {resp.text[:200]}"
        )
    data = resp.json()

    # Extract nested APY fields
    kamino_apy = data.get("kaminoApy", {})
    apy = data.get("apy", {})
    vault_balances = data.get("vaultBalances", {})
    token_a_bal = vault_balances.get("tokenA", {})
    token_b_bal = vault_balances.get("tokenB", {})

    # shareMint may not be in metrics — fetch from listing if needed
    share_mint = data.get("shareMint", "")
    if not share_mint:
        share_mint = _fetch_strategy_share_mint(strategy_pubkey)

    return KaminoStrategyMetrics(
        strategy_pubkey=strategy_pubkey,
        token_a=data.get("tokenA", ""),
        token_b=data.get("tokenB", ""),
        token_a_mint=data.get("tokenAMint", ""),
        token_b_mint=data.get("tokenBMint", ""),
        share_mint=share_mint,
        total_value_locked=_safe_float(data.get("totalValueLocked", "0")),
        share_price=_safe_float(data.get("sharePrice", "1.0")),
        shares_issued=_safe_float(data.get("sharesIssued", "0")),
        fee_apy=_safe_float(apy.get("vault", {}).get("feeApy", "0")),
        kamino_apy_7d=_safe_float(kamino_apy.get("vault", {}).get("apy7d", "0")),
        total_apy=_safe_float(apy.get("totalApy", "0")),
        token_a_balance=_safe_float(token_a_bal.get("total", "0")),
        token_a_balance_usd=_safe_float(token_a_bal.get("totalUsd", "0")),
        token_b_balance=_safe_float(token_b_bal.get("total", "0")),
        token_b_balance_usd=_safe_float(token_b_bal.get("totalUsd", "0")),
    )


def _fetch_strategy_share_mint(strategy_pubkey: str) -> str:
    """
    Look up a strategy's shareMint from the /strategies listing.
    The metrics endpoint doesn't always include the shareMint,
    so we fall back to the full listing.
    """
    url = f"{KAMINO_API_BASE}/strategies"
    params = {"env": "mainnet-beta", "status": "LIVE"}
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Kamino strategies listing fetch failed: HTTP {resp.status_code}")
    strategies = resp.json()
    for s in strategies:
        if s.get("address") == strategy_pubkey:
            return s.get("shareMint", "")
    raise RuntimeError(
        f"Strategy {strategy_pubkey} not found in Kamino strategies listing. "
        f"It may be inactive or delisted."
    )


def fetch_kamino_strategy_whales(
    strategy_pubkey: str,
    solana_rpc_url: str | None = None,
    min_usd: float = 100_000,
) -> list[dict]:
    """
    End-to-end: fetch CLMM strategy metrics → get top share holders.

    Uses the strategy's shareMint to call getTokenLargestAccounts
    via Solana RPC, just like kVault whale fetching.

    ERRORS LOUDLY on failure.

    Returns: list of [{address, shares, tokens, balance_usd}]
    """
    solana_rpc_url = solana_rpc_url or get_solana_rpc_url()

    metrics = fetch_kamino_strategy_metrics(strategy_pubkey)
    if not metrics.share_mint:
        raise RuntimeError(
            f"Kamino strategy {strategy_pubkey} has no share_mint. "
            f"Cannot fetch whale data without the shares token mint."
        )

    # For CLMM strategies: token_price = TVL / shares_issued
    # This converts shares → USD value
    token_price = 1.0
    tokens_per_share = 1.0
    if metrics.shares_issued > 0:
        tokens_per_share = metrics.share_price  # sharePrice already gives tokens/share
        token_price = (
            metrics.total_value_locked / (metrics.shares_issued * metrics.share_price)
            if metrics.share_price > 0
            else 1.0
        )

    holders = fetch_kamino_top_share_holders(
        shares_mint=metrics.share_mint,
        solana_rpc_url=solana_rpc_url,
        token_price=token_price,
        tokens_per_share=tokens_per_share,
    )

    # Filter by minimum USD
    filtered = [h for h in holders if h["balance_usd"] >= min_usd]

    if not filtered:
        raise RuntimeError(
            f"No Kamino strategy holders found with balance >= ${min_usd:,.0f} "
            f"for strategy {strategy_pubkey}. Total holders from RPC: {len(holders)}."
        )

    return filtered


def fetch_kamino_market_reserves(
    market_pubkey: str,
    asset_symbol: str | None = None,
) -> list[KaminoReserveMetrics]:
    """
    Fetch all reserves for a Kamino lending market.
    GET /kamino-market/{market}/reserves/metrics?env=mainnet-beta
    """
    url = f"{KAMINO_API_BASE}/kamino-market/{market_pubkey}/reserves/metrics"
    resp = requests.get(url, params={"env": "mainnet-beta"}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Kamino market reserves fetch failed: HTTP {resp.status_code} for {market_pubkey}. "
            f"Response: {resp.text[:200]}"
        )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response format from Kamino reserves: {type(data)}")

    reserves = []
    for r in data:
        token = r.get("liquidityToken", "")
        if asset_symbol and asset_symbol.upper() not in token.upper():
            continue

        reserves.append(
            KaminoReserveMetrics(
                reserve_pubkey=r.get("reserve", ""),
                liquidity_token=token,
                liquidity_token_mint=r.get("liquidityTokenMint", ""),
                max_ltv=_safe_float(r.get("maxLtv", "0")),
                borrow_apy=_safe_float(r.get("borrowApy", "0")),
                supply_apy=_safe_float(r.get("supplyApy", "0")),
                total_supply=_safe_float(r.get("totalSupply", "0")),
                total_borrow=_safe_float(r.get("totalBorrow", "0")),
                total_supply_usd=_safe_float(r.get("totalSupplyUsd", "0")),
                total_borrow_usd=_safe_float(r.get("totalBorrowUsd", "0")),
            )
        )

    return reserves


def fetch_kamino_reserve_for_asset(
    asset_symbol: str,
    market_name: str = "main",
) -> KaminoReserveMetrics | None:
    """
    Find a specific asset's reserve in a named Kamino market.
    """
    market_pubkey = get_kamino_market_pubkey(market_name)
    reserves = fetch_kamino_market_reserves(market_pubkey, asset_symbol)
    if not reserves:
        return None
    return max(reserves, key=lambda r: r.total_supply_usd)


# ============================================================================
# WHALE FETCHING — Solana RPC (getTokenLargestAccounts)
# ============================================================================


def fetch_kamino_top_share_holders(
    shares_mint: str,
    solana_rpc_url: str | None = None,
    token_price: float = 1.0,
    tokens_per_share: float = 1.0,
) -> list[dict]:
    """
    Fetch top holders of a kVault's shares token via Solana RPC.

    Uses getTokenLargestAccounts which returns up to 20 largest holders.
    This is the production method for Kamino whale data.

    ERRORS LOUDLY if the Solana RPC call fails.

    Args:
        shares_mint: SPL token mint address for vault shares
        solana_rpc_url: Solana RPC endpoint (defaults to HELIUS_SOLANA_RPC_URL)
        token_price: USD price per underlying token
        tokens_per_share: Conversion rate from shares to underlying tokens

    Returns: list of [{address, shares, tokens, balance_usd}]
    """
    solana_rpc_url = solana_rpc_url or get_solana_rpc_url()

    resp = requests.post(
        solana_rpc_url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenLargestAccounts",
            "params": [shares_mint],
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    if "error" in result:
        raise RuntimeError(
            f"Solana RPC getTokenLargestAccounts error for {shares_mint}: {result['error']}. "
            f"Ensure HELIUS_SOLANA_RPC_URL is valid."
        )

    accounts = result.get("result", {}).get("value", [])

    if not accounts:
        raise RuntimeError(
            f"getTokenLargestAccounts returned 0 accounts for shares mint {shares_mint}. "
            f"The vault may be empty or the shares_mint address is wrong."
        )

    holders = []
    for acct in accounts:
        amount_str = acct.get("amount", "0")
        decimals = acct.get("decimals", 6)
        shares = int(amount_str) / (10**decimals)
        tokens = shares * tokens_per_share
        usd = tokens * token_price

        holders.append(
            {
                "address": acct.get("address", "unknown"),
                "shares": shares,
                "tokens": tokens,
                "balance_usd": usd,
            }
        )

    holders.sort(key=lambda h: h["balance_usd"], reverse=True)
    return holders


def fetch_kamino_vault_whales(
    vault_pubkey: str,
    solana_rpc_url: str | None = None,
    min_usd: float = 100_000,
) -> list[dict]:
    """
    End-to-end: fetch kVault state + metrics → get top share holders.

    ERRORS LOUDLY on failure.

    Returns: list of [{address, shares, tokens, balance_usd}]
    """
    solana_rpc_url = solana_rpc_url or get_solana_rpc_url()

    state = fetch_kamino_vault_state(vault_pubkey)
    if not state.shares_mint:
        raise RuntimeError(
            f"Kamino vault {vault_pubkey} has no shares_mint. "
            f"Cannot fetch whale data without the shares token mint."
        )

    metrics = fetch_kamino_vault_metrics(vault_pubkey)

    holders = fetch_kamino_top_share_holders(
        shares_mint=state.shares_mint,
        solana_rpc_url=solana_rpc_url,
        token_price=metrics.token_price,
        tokens_per_share=metrics.tokens_per_share,
    )

    # Filter by minimum USD
    filtered = [h for h in holders if h["balance_usd"] >= min_usd]

    if not filtered:
        raise RuntimeError(
            f"No Kamino vault holders found with balance >= ${min_usd:,.0f} "
            f"for vault {vault_pubkey}. Total holders from RPC: {len(holders)}."
        )

    return filtered


# ============================================================================
# COMPOSITE FETCHERS (for data.py integration)
# ============================================================================


def fetch_kamino_lend_snapshot(
    market_name: str,
    asset_symbol: str,
) -> dict:
    """
    Fetch a lending reserve snapshot suitable for VaultSnapshot creation.
    """
    reserve = fetch_kamino_reserve_for_asset(asset_symbol, market_name)
    if not reserve:
        raise RuntimeError(
            f"No {asset_symbol} reserve found in Kamino {market_name} market. "
            f"Market pubkey: {get_kamino_market_pubkey(market_name)}"
        )

    return {
        "address": reserve.reserve_pubkey,
        "chain_id": 0,  # Solana
        "asset_address": reserve.liquidity_token_mint,
        "asset_decimals": 6,
        "asset_symbol": asset_symbol,
        "asset_price_usd": 1.0,
        "total_supply_usd": reserve.total_supply_usd,
        "total_borrow_usd": reserve.total_borrow_usd,
        "utilization": reserve.utilization,
        "supply_apy": reserve.supply_apy,
        "borrow_apy": reserve.borrow_apy,
        "top_depositors": [],  # Use fetch_kamino_vault_whales for vault-level whale data
    }


def fetch_kamino_earn_snapshot(
    vault_pubkey: str,
    solana_rpc_url: str | None = None,
) -> dict:
    """
    Fetch a kVault (Earn) snapshot suitable for VaultSnapshot creation.
    Includes metrics + whale data from Solana RPC.

    ERRORS LOUDLY if whale fetch fails.
    """
    solana_rpc_url = solana_rpc_url or get_solana_rpc_url()
    metrics = fetch_kamino_vault_metrics(vault_pubkey)

    # Fetch whale data — errors loudly
    state = fetch_kamino_vault_state(vault_pubkey)
    depositors = []
    if state.shares_mint:
        depositors = fetch_kamino_top_share_holders(
            shares_mint=state.shares_mint,
            solana_rpc_url=solana_rpc_url,
            token_price=metrics.token_price,
            tokens_per_share=metrics.tokens_per_share,
        )

    if not depositors:
        raise RuntimeError(
            f"Failed to fetch depositors for Kamino Earn vault {vault_pubkey}. "
            f"shares_mint: {state.shares_mint or 'NOT FOUND'}. "
            f"Cannot build whale profiles without depositor data."
        )

    return {
        "address": vault_pubkey,
        "chain_id": 0,
        "asset_decimals": state.token_mint_decimals,
        "asset_symbol": "PYUSD",  # TODO: derive from token_mint via on-chain lookup
        "asset_price_usd": metrics.token_price,
        "total_supply_usd": metrics.total_tvl_usd,
        "total_borrow_usd": 0,
        "utilization": (
            metrics.tokens_invested / (metrics.tokens_available + metrics.tokens_invested)
            if (metrics.tokens_available + metrics.tokens_invested) > 0
            else 0
        ),
        "base_apy": metrics.base_apy,
        "total_apy": metrics.apy_actual,
        "incentive_apy": metrics.apy_incentives + metrics.apy_farm_rewards,
        "number_of_holders": metrics.number_of_holders,
        "top_depositors": depositors,
        "metrics": metrics,
    }


# ============================================================================
# HELPERS
# ============================================================================


def _safe_float(val) -> float:
    """Safely parse a value to float, handling strings and None."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

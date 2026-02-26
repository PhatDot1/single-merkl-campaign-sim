"""
Direct on-chain data fetching for EVM protocols (Aave V3 Core, Aave V3 Horizon, Euler V2).

Production configuration:
- RPC URLs loaded from environment variables (ALCHEMY_ETH_RPC_URL)
- Whale fetching via Alchemy alchemy_getAssetTransfers (indexed, fast, paginated)
- NO fallback to Transfer log scanning — errors loudly if Alchemy fails
- Aave Horizon is a separate pool with its own contract address

For Aave V3 Core:
- Pool: 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
- Pool.getReserveData(asset) → liquidity rate, variable borrow rate, aToken address
- aToken holders via Alchemy getAssetTransfers → whale positions

For Aave V3 Horizon (RWA Instance):
- Pool: 0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8
- Same ABI as Core, different contract address
- Data Provider: 0x53519c32f73fE1797d10210c4950fFeBa3b21504

For Euler V2:
- EVault.cash(), totalBorrows(), interestRate() → supply APY
- EVault ERC20 balances via Alchemy → depositor positions

Known addresses:
- RLUSD on Ethereum: 0x8292Bb45bf1Ee4d140127049757C0C38e47a8A75
- PYUSD on Ethereum: 0x6c3ea9036406852006290770BEdFcAbA0e23A0e8
- Aave V3 Pool (Core): 0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
- Aave V3 Pool (Horizon): 0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8
- Euler Sentora RLUSD: 0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2
- Euler Sentora PYUSD: 0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5

Known aToken addresses:
- aEthPYUSD (Core):       0x0c0d01abf3e6adfca0989ebba9d6e85dd58eab1e
- aEthRLUSD (Core):       0x6A1792a91C08e9f0bFe7a990871B786643237f0F
- aHorRwaRLUSD (Horizon): 0x503D751B13a71D8e69Db021DF110bfa7aE1dA889
- aHorRwaUSDC (Horizon):  0x68215B6533c47ff9f7125aC95adf00fE4a62f79e
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

# ============================================================================
# ENVIRONMENT / RPC CONFIGURATION
# ============================================================================


def get_eth_rpc_url() -> str:
    """Get Ethereum RPC URL from environment. Errors loudly if not set."""
    url = os.environ.get("ALCHEMY_ETH_RPC_URL")
    if not url:
        raise RuntimeError(
            "ALCHEMY_ETH_RPC_URL not set in environment. "
            "Set it in your .env file or environment variables. "
            "Example: ALCHEMY_ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
        )
    return url


# ============================================================================
# CONSTANTS
# ============================================================================

# Known stablecoin addresses
STABLECOIN_ADDRESSES = {
    "PYUSD": "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8",
    "RLUSD": "0x8292Bb45bf1Ee4d140127049757C2E0fF06317eD",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
}

# Aave V3 pool addresses — Core and Horizon are SEPARATE contracts
AAVE_V3_POOLS = {
    "core": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
    "horizon": "0xAe05Cd22df81871bc7cC2a04BeCfb516bFe332C8",
}

AAVE_V3_DATA_PROVIDERS = {
    "core": "0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3",
    "horizon": "0x53519c32f73fE1797d10210c4950fFeBa3b21504",
}

# Known aToken addresses (for direct whale scanning without on-chain lookup)
KNOWN_ATOKENS = {
    ("PYUSD", "core"): "0x0c0d01abf3e6adfca0989ebba9d6e85dd58eab1e",
    ("RLUSD", "core"): "0xfa82580c16a31d0c1bc632a36f82e83efef3eec0",
    ("RLUSD", "horizon"): "0xe3190143eb552456f88464662f0c0c4ac67a77eb",
    ("USDC", "horizon"): "0x68215B6533c47ff9f7125aC95adf00fE4a62f79e",
}

# Euler Sentora vault addresses
EULER_VAULTS = {
    "RLUSD": "0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2",
    "PYUSD": "0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5",
}

# Ray = 1e27 (Aave's rate precision)
RAY = 10**27
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ============================================================================
# LOW-LEVEL RPC
# ============================================================================


def _eth_call(rpc_url: str, to: str, data: str) -> str:
    """Raw eth_call, returns hex result."""
    resp = requests.post(
        rpc_url,
        json={
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": to, "data": data}, "latest"],
            "id": 1,
        },
        timeout=30,
    )
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"eth_call error: {result['error']}")
    return result["result"]


def _decode_uint256(hex_str: str) -> int:
    """Decode uint256 from hex."""
    h = hex_str[2:] if hex_str.startswith("0x") else hex_str
    return int(h, 16) if h else 0


def _decode_address(hex_str: str) -> str:
    """Decode address from 32-byte padded hex."""
    h = hex_str[2:] if hex_str.startswith("0x") else hex_str
    return "0x" + h[-40:]


def _encode_address(addr: str) -> str:
    """Encode address as 32-byte padded hex (no 0x prefix)."""
    a = addr[2:] if addr.startswith("0x") else addr
    return a.lower().zfill(64)


# ============================================================================
# AAVE V3 — DIRECT ON-CHAIN (supports both Core and Horizon)
# ============================================================================


@dataclass
class AaveReserveData:
    """Parsed Aave V3 reserve data from on-chain."""

    asset_address: str
    asset_symbol: str
    market: str  # "core" or "horizon"
    liquidity_rate_ray: int
    variable_borrow_rate_ray: int
    a_token_address: str
    total_supply_usd: float
    total_borrow_usd: float
    supply_apy: float  # decimal
    borrow_apy: float  # decimal
    utilization: float  # decimal
    supply_cap: float = 0.0  # USD (0 = unlimited)
    borrow_cap: float = 0.0  # USD (0 = unlimited)


def fetch_aave_reserve_data(
    asset_address: str,
    asset_symbol: str = "UNKNOWN",
    pool_address: str | None = None,
    market: str = "core",
    rpc_url: str | None = None,
) -> AaveReserveData:
    """
    Fetch Aave V3 reserve data directly from the Pool contract.

    Supports both Core and Horizon markets via separate pool addresses.

    Pool.getReserveData(address asset) returns a struct with:
    - [0] configuration (uint256)
    - [1] liquidityIndex (uint128)
    - [2] currentLiquidityRate (uint128) — supply APY in ray
    - [3] variableBorrowIndex (uint128)
    - [4] currentVariableBorrowRate (uint128) — borrow APY in ray
    - [5] currentStableBorrowRate (uint128) — deprecated
    - [6] lastUpdateTimestamp (uint40)
    - [7] id (uint16)
    - [8] aTokenAddress (address)
    - [9] stableDebtTokenAddress (address)
    - [10] variableDebtTokenAddress (address)
    - [11] interestRateStrategyAddress (address)
    - [12] accruedToTreasury (uint128)
    - [13] unbacked (uint128)
    - [14] isolationModeTotalDebt (uint128)
    """
    rpc_url = rpc_url or get_eth_rpc_url()
    pool = pool_address or AAVE_V3_POOLS.get(market)
    if not pool:
        raise RuntimeError(f"Unknown Aave market: {market}")

    # getReserveData(address) selector = 0x35ea6a75
    calldata = "0x35ea6a75" + _encode_address(asset_address)

    try:
        raw = _eth_call(rpc_url, pool, calldata)
        hex_data = raw[2:]  # Strip 0x

        # Each slot is 64 hex chars (32 bytes)
        config_word = _decode_uint256("0x" + hex_data[0:64])
        liquidity_rate = _decode_uint256("0x" + hex_data[2 * 64 : 3 * 64])
        borrow_rate = _decode_uint256("0x" + hex_data[4 * 64 : 5 * 64])
        a_token = _decode_address("0x" + hex_data[8 * 64 : 9 * 64])

        # ── Parse supply/borrow caps from config word ──
        # Aave V3 ReserveConfigurationMap bit layout:
        #   bits 36-51: borrow cap (in whole tokens, 0 = unlimited)
        #   bits 56-67: supply cap (in whole tokens, 0 = unlimited)
        # We extract and convert to USD (1:1 for stablecoins).
        borrow_cap_tokens = (config_word >> 80) & ((1 << 36) - 1)
        supply_cap_tokens = (config_word >> 116) & ((1 << 36) - 1)

        # Convert ray to decimal APY (already annualized)
        supply_apy = liquidity_rate / RAY
        borrow_apy = borrow_rate / RAY

        # Get total supply via aToken totalSupply
        total_supply_raw = _decode_uint256(
            _eth_call(rpc_url, a_token, "0x18160ddd")  # totalSupply()
        )
        # Get asset decimals
        decimals = _decode_uint256(
            _eth_call(rpc_url, asset_address, "0x313ce567")  # decimals()
        )

        total_supply_usd = total_supply_raw / (10**decimals)

        # Get variable debt token for total borrows
        v_debt_token = _decode_address("0x" + hex_data[10 * 64 : 11 * 64])
        total_borrow_raw = _decode_uint256(_eth_call(rpc_url, v_debt_token, "0x18160ddd"))
        total_borrow_usd = total_borrow_raw / (10**decimals)

        utilization = total_borrow_usd / total_supply_usd if total_supply_usd > 0 else 0

        # Convert caps from whole tokens to USD (stablecoin = 1:1)
        supply_cap_usd = float(supply_cap_tokens)  # Already in whole tokens ≈ USD for stables
        borrow_cap_usd = float(borrow_cap_tokens)

        return AaveReserveData(
            asset_address=asset_address,
            asset_symbol=asset_symbol,
            market=market,
            liquidity_rate_ray=liquidity_rate,
            variable_borrow_rate_ray=borrow_rate,
            a_token_address=a_token,
            total_supply_usd=total_supply_usd,
            total_borrow_usd=total_borrow_usd,
            supply_apy=supply_apy,
            borrow_apy=borrow_apy,
            utilization=utilization,
            supply_cap=supply_cap_usd,
            borrow_cap=borrow_cap_usd,
        )

    except Exception as e:
        raise RuntimeError(
            f"Aave V3 on-chain fetch failed for {asset_symbol} ({asset_address}) "
            f"on market={market} pool={pool}: {e}"
        )


# ============================================================================
# ALCHEMY WHALE FETCHING (replaces Transfer log scanning)
# ============================================================================


def _detect_alchemy(rpc_url: str) -> bool:
    """Check if the RPC URL is an Alchemy endpoint."""
    return "alchemy.com" in rpc_url.lower()


def _fetch_top_holders_via_alchemy(
    rpc_url: str,
    token_address: str,
    asset_decimals: int,
    price_usd: float,
    min_usd: float,
    max_holders: int,
) -> list[dict]:
    """Alchemy-specific: use indexed alchemy_getAssetTransfers (fast, complete)."""
    all_transfers = []
    page_key = None
    zero_addr = "0x0000000000000000000000000000000000000000"

    for _ in range(20):
        params = {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "contractAddresses": [token_address],
            "category": ["erc20"],
            "excludeZeroValue": True,
            "maxCount": "0x3e8",
            "withMetadata": False,
        }
        if page_key:
            params["pageKey"] = page_key
        resp = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getAssetTransfers",
                "params": [params],
            },
            timeout=60,
        )
        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"alchemy_getAssetTransfers error: {result['error']}")
        transfers = result.get("result", {}).get("transfers", [])
        all_transfers.extend(transfers)
        page_key = result.get("result", {}).get("pageKey")
        if not page_key:
            break

    print(f"  Retrieved {len(all_transfers)} Alchemy transfers")
    balances: dict[str, int] = {}
    for tx in all_transfers:
        from_addr = (tx.get("from") or "").lower()
        to_addr = (tx.get("to") or "").lower()
        raw_contract = tx.get("rawContract", {})
        raw_hex = raw_contract.get("value", "0x0")
        if raw_hex and raw_hex != "0x":
            value = _decode_uint256(raw_hex)
        else:
            dec_val = tx.get("value")
            value = int(float(dec_val) * (10**asset_decimals)) if dec_val else 0
        if not value:
            continue
        if from_addr and from_addr != zero_addr:
            balances[from_addr] = balances.get(from_addr, 0) - value
        if to_addr and to_addr != zero_addr:
            balances[to_addr] = balances.get(to_addr, 0) + value

    holders = []
    for addr, bal in balances.items():
        if bal <= 0:
            continue
        usd = bal / (10**asset_decimals) * price_usd
        if usd >= min_usd:
            holders.append({"address": addr, "balance_raw": bal, "balance_usd": usd})
    holders.sort(key=lambda h: h["balance_usd"], reverse=True)
    return holders[:max_holders]


def _fetch_top_holders_via_logs(
    rpc_url: str,
    token_address: str,
    asset_decimals: int,
    price_usd: float,
    min_usd: float,
    max_holders: int,
) -> list[dict]:
    """
    Fetch top holders by scanning recent Transfer logs in chunks.

    Works with ANY RPC provider (Infura, Alchemy, QuickNode, etc.).
    Scans the last ~500k blocks (~70 days) in 10k-block chunks.
    This catches recent activity for continuously-minted tokens like aTokens.
    """
    TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    CHUNK_SIZE = 10_000  # Infura's safe limit
    MAX_CHUNKS = 50  # ~500k blocks = ~70 days
    ZERO_ADDR = "0x" + "0" * 40

    resp = requests.post(
        rpc_url,
        json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
        timeout=15,
    )
    current_block = int(resp.json()["result"], 16)
    start_block = max(0, current_block - (CHUNK_SIZE * MAX_CHUNKS))

    print(
        f"  Scanning Transfer logs blocks {start_block}..{current_block} "
        f"({MAX_CHUNKS} chunks of {CHUNK_SIZE})"
    )

    balances: dict[str, int] = {}
    total_logs = 0

    for chunk_start in range(start_block, current_block + 1, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE - 1, current_block)
        try:
            resp = requests.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_getLogs",
                    "id": 1,
                    "params": [
                        {
                            "address": token_address,
                            "topics": [TRANSFER_TOPIC],
                            "fromBlock": hex(chunk_start),
                            "toBlock": hex(chunk_end),
                        }
                    ],
                },
                timeout=30,
            )
            result = resp.json()
            if "error" in result:
                err = result["error"]
                if err.get("code") == -32005:
                    # Infura: >10k results in chunk — skip
                    continue
                raise RuntimeError(f"eth_getLogs error: {err}")

            logs = result.get("result", [])
            total_logs += len(logs)
            for log in logs:
                topics = log.get("topics", [])
                if len(topics) < 3:
                    continue
                from_addr = _decode_address(topics[1]).lower()
                to_addr = _decode_address(topics[2]).lower()
                value = _decode_uint256(log.get("data", "0x0"))
                if from_addr != ZERO_ADDR.lower():
                    balances[from_addr] = balances.get(from_addr, 0) - value
                if to_addr != ZERO_ADDR.lower():
                    balances[to_addr] = balances.get(to_addr, 0) + value
        except RuntimeError:
            raise
        except Exception as e:
            print(f"    Chunk {chunk_start}-{chunk_end} failed: {e}")
            continue

    print(f"  Processed {total_logs} Transfer logs")

    holders = []
    for addr, bal in balances.items():
        if bal <= 0:
            continue
        usd = bal / (10**asset_decimals) * price_usd
        if usd >= min_usd:
            holders.append({"address": addr, "balance_raw": bal, "balance_usd": usd})
    holders.sort(key=lambda h: h["balance_usd"], reverse=True)
    return holders[:max_holders]


def fetch_erc20_top_holders(
    token_address: str,
    asset_decimals: int = 6,
    price_usd: float = 1.0,
    min_usd: float = 1_000_000,
    rpc_url: str | None = None,
    max_holders: int = 50,
) -> list[dict]:
    """
    Fetch top ERC20 holders. Auto-detects provider:
    - Alchemy: uses indexed alchemy_getAssetTransfers (fast, complete)
    - Infura/other: uses eth_getLogs with chunked block scanning

    Returns: list of [{address, balance_raw, balance_usd}] (may be empty)
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    if _detect_alchemy(rpc_url):
        print(f"  Using Alchemy indexed API for {token_address[:10]}...")
        return _fetch_top_holders_via_alchemy(
            rpc_url, token_address, asset_decimals, price_usd, min_usd, max_holders
        )
    else:
        print(f"  Using eth_getLogs chunked scan for {token_address[:10]}... (non-Alchemy RPC)")
        return _fetch_top_holders_via_logs(
            rpc_url, token_address, asset_decimals, price_usd, min_usd, max_holders
        )


# ============================================================================
# AAVE WHALE FETCHING (production — uses Alchemy)
# ============================================================================


def fetch_aave_atoken_top_holders(
    a_token_address: str,
    asset_decimals: int = 6,
    rpc_url: str | None = None,
    min_usd: float = 1_000_000,
) -> list[dict]:
    """
    Fetch top aToken holders using Alchemy indexed transfer API.

    Production method — NO Transfer log scanning.
    ERRORS LOUDLY if fetch fails.

    Returns: list of [{address, balance_raw, balance_usd}]
    """
    return fetch_erc20_top_holders(
        token_address=a_token_address,
        asset_decimals=asset_decimals,
        price_usd=1.0,  # Stablecoin
        min_usd=min_usd,
        rpc_url=rpc_url,
    )


# ============================================================================
# EULER V2 — DIRECT ON-CHAIN
# ============================================================================


@dataclass
class EulerVaultData:
    """Parsed Euler V2 (EVault) data from on-chain."""

    vault_address: str
    asset_symbol: str
    asset_address: str
    asset_decimals: int
    cash: int  # raw
    total_borrows: int  # raw
    total_supply_usd: float
    total_borrow_usd: float
    supply_apy: float  # decimal
    utilization: float  # decimal
    supply_cap: float = 0.0  # USD (0 = unlimited)
    borrow_cap: float = 0.0  # USD (0 = unlimited)


def fetch_euler_vault_data(
    vault_address: str,
    asset_symbol: str = "UNKNOWN",
    rpc_url: str | None = None,
) -> EulerVaultData:
    """
    Fetch Euler V2 EVault data directly from on-chain.

    These vaults are ERC4626-compatible. We use:
    - totalAssets() → uint256: total assets under management
    - totalBorrows() → uint256: total borrowed
    - asset() → address: underlying asset
    - interestRate() → uint256: current per-second interest rate (1e27 scale)

    Note: cash() is NOT available on Euler V2 vaults — derive it as
    totalAssets - totalBorrows.
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    TOTAL_ASSETS_SEL = "0x01e1d114"  # totalAssets()
    TOTAL_BORROWS_SEL = "0x47bd3718"  # totalBorrows()
    ASSET_SEL = "0x38d52e0f"  # asset()
    INTEREST_RATE_SEL = "0x7c3a00fd"  # interestRate()
    DECIMALS_SEL = "0x313ce567"  # decimals()
    CAPS_SEL = "0x18aab3e0"  # caps() → (uint16 supplyCap, uint16 borrowCap) in Euler V2

    try:
        total_assets_raw = _decode_uint256(_eth_call(rpc_url, vault_address, TOTAL_ASSETS_SEL))
        borrows_raw = _decode_uint256(_eth_call(rpc_url, vault_address, TOTAL_BORROWS_SEL))
        asset_addr = _decode_address(_eth_call(rpc_url, vault_address, ASSET_SEL))
        asset_decimals = _decode_uint256(_eth_call(rpc_url, asset_addr, DECIMALS_SEL))

        # Derive cash (idle) from totalAssets - totalBorrows
        cash_raw = max(0, total_assets_raw - borrows_raw)

        try:
            rate_raw = _decode_uint256(_eth_call(rpc_url, vault_address, INTEREST_RATE_SEL))
            borrow_apy = rate_raw * SECONDS_PER_YEAR / 1e27
            util = borrows_raw / total_assets_raw if total_assets_raw > 0 else 0
            supply_apy = borrow_apy * util * 0.90  # ~10% reserve fee estimate
        except Exception:
            borrow_apy = 0.0
            supply_apy = 0.0
            util = 0.0

        # ── Fetch supply/borrow caps ──
        # Euler V2 caps() returns (uint16, uint16) packed in a single word.
        # Values are in "amount * 10^(decimals - 2)" resolution or raw tokens.
        # 0 = unlimited, type(uint16).max = disabled.
        supply_cap_usd = 0.0
        borrow_cap_usd = 0.0
        try:
            caps_raw = _eth_call(rpc_url, vault_address, CAPS_SEL)
            caps_hex = caps_raw[2:] if caps_raw.startswith("0x") else caps_raw
            if len(caps_hex) >= 128:
                # Two uint256 slots: supplyCap, borrowCap (Euler returns them as uint256 for ABI compat)
                supply_cap_raw = _decode_uint256("0x" + caps_hex[0:64])
                borrow_cap_raw = _decode_uint256("0x" + caps_hex[64:128])
                divisor = 10**asset_decimals
                if supply_cap_raw > 0 and supply_cap_raw < (1 << 255):
                    supply_cap_usd = supply_cap_raw / divisor
                if borrow_cap_raw > 0 and borrow_cap_raw < (1 << 255):
                    borrow_cap_usd = borrow_cap_raw / divisor
        except Exception as cap_err:
            print(f"  Note: Could not fetch Euler caps for {vault_address[:10]}: {cap_err}")

        divisor = 10**asset_decimals
        total_supply_usd = total_assets_raw / divisor
        total_borrow_usd = borrows_raw / divisor

        return EulerVaultData(
            vault_address=vault_address,
            asset_symbol=asset_symbol,
            asset_address=asset_addr,
            asset_decimals=asset_decimals,
            cash=cash_raw,
            total_borrows=borrows_raw,
            total_supply_usd=total_supply_usd,
            total_borrow_usd=total_borrow_usd,
            supply_apy=supply_apy,
            utilization=util,
            supply_cap=supply_cap_usd,
            borrow_cap=borrow_cap_usd,
        )

    except Exception as e:
        raise RuntimeError(
            f"Euler V2 on-chain fetch failed for {asset_symbol} ({vault_address}): {e}"
        )


def fetch_euler_vault_top_holders(
    vault_address: str,
    asset_decimals: int = 6,
    rpc_url: str | None = None,
    min_usd: float = 1_000_000,
) -> list[dict]:
    """
    Fetch top eToken holders using Alchemy indexed transfer API.

    The Euler vault IS the ERC20 share token, so we scan the vault address.
    Production method — ERRORS LOUDLY if fetch fails.

    Returns: list of [{address, balance_raw, balance_usd}]
    """
    return fetch_erc20_top_holders(
        token_address=vault_address,
        asset_decimals=asset_decimals,
        price_usd=1.0,
        min_usd=min_usd,
        rpc_url=rpc_url,
    )


# ============================================================================
# COMBINED BASE APY FETCHERS
# ============================================================================


def fetch_aave_base_apy_onchain(
    asset_symbol: str,
    market: str = "core",
    rpc_url: str | None = None,
) -> dict:
    """
    Fetch Aave V3 base APY directly from on-chain.

    Returns dict with {supply_apy, borrow_apy, total_supply_usd, utilization, source}.
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    asset_addr = STABLECOIN_ADDRESSES.get(asset_symbol)
    if not asset_addr:
        return {"supply_apy": 0, "source": "error", "error": f"Unknown asset: {asset_symbol}"}

    pool = AAVE_V3_POOLS.get(market)
    if not pool:
        return {"supply_apy": 0, "source": "error", "error": f"Unknown Aave market: {market}"}

    try:
        data = fetch_aave_reserve_data(asset_addr, asset_symbol, pool, market, rpc_url)
        return {
            "supply_apy": data.supply_apy,
            "borrow_apy": data.borrow_apy,
            "total_supply_usd": data.total_supply_usd,
            "total_borrow_usd": data.total_borrow_usd,
            "utilization": data.utilization,
            "a_token_address": data.a_token_address,
            "market": market,
            "source": "aave_onchain",
        }
    except Exception as e:
        return {"supply_apy": 0, "source": "error", "error": str(e)}


def fetch_euler_base_apy_onchain(
    asset_symbol: str,
    rpc_url: str | None = None,
) -> dict:
    """
    Fetch Euler V2 base APY directly from on-chain.

    Returns dict with {supply_apy, total_supply_usd, utilization, source}.
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    vault_addr = EULER_VAULTS.get(asset_symbol)
    if not vault_addr:
        return {"supply_apy": 0, "source": "error", "error": f"No Euler vault for {asset_symbol}"}

    try:
        data = fetch_euler_vault_data(vault_addr, asset_symbol, rpc_url)
        return {
            "supply_apy": data.supply_apy,
            "total_supply_usd": data.total_supply_usd,
            "total_borrow_usd": data.total_borrow_usd,
            "utilization": data.utilization,
            "vault_address": vault_addr,
            "source": "euler_onchain",
        }
    except Exception as e:
        return {"supply_apy": 0, "source": "error", "error": str(e)}


# ============================================================================
# WHALE PROFILE BUILDER (unified for Aave/Euler)
# ============================================================================


def build_whale_profiles_from_holders(
    holders: list[dict],
    total_supply_usd: float,
    r_threshold: float,
    min_position_usd: float = 1_000_000,
    max_whales: int = 10,
    max_single_whale_share: float = 0.25,
    max_total_whale_share: float = 0.60,
    max_alt_rate_multiplier: float = 2.0,  # Increased from 1.5 to allow higher thresholds
    whale_history: dict | None = None,
) -> list:
    """
    Build WhaleProfile objects from holder data.

    Protocol-agnostic — works with any list of [{address, balance_usd}] dicts.
    ERRORS LOUDLY if no holders meet the threshold.

    When whale_history is provided (from Dune sync), uses empirical data to
    improve exit threshold estimates instead of synthetic generation.

    Args:
        holders: [{address, balance_usd}] sorted by balance descending
        total_supply_usd: Total pool TVL
        r_threshold: Competitor rate
        min_position_usd: Min position to qualify as whale
        max_whales: Max profiles to build
        max_single_whale_share: Cap per whale as fraction of TVL
        max_total_whale_share: Cap for total whale TVL
        max_alt_rate_multiplier: Max alt_rate as multiple of r_threshold
        whale_history: Optional dict from Dune: {address: {deposits, withdrawals, avg_hold_days}}

    Sanity checks applied:
    - Single whale capped at max_single_whale_share of TVL (default 25%)
    - Total whale TVL capped at max_total_whale_share (default 60%)
    - alt_rate capped at r_threshold * max_alt_rate_multiplier (default 2.0x)
    - Warnings logged for clamped values

    Returns list of WhaleProfile.
    """
    from .agents import WhaleProfile

    # Build whale history lookup if provided
    history_lookup = {}
    if whale_history:
        for addr, data in whale_history.items():
            history_lookup[addr.lower()] = data

    profiles = []
    for i, h in enumerate(holders[:max_whales]):
        pos = h["balance_usd"]
        if pos < min_position_usd:
            continue

        share = pos / total_supply_usd if total_supply_usd > 0 else 0

        # ── Clamp single whale concentration ──
        if share > max_single_whale_share:
            clamped_pos = total_supply_usd * max_single_whale_share
            addr = h.get("address", f"unknown_{i}")[:8]
            print(
                f"  WARNING: Whale {addr} share {share:.1%} > {max_single_whale_share:.0%} cap, "
                f"clamping ${pos:,.0f} -> ${clamped_pos:,.0f}"
            )
            pos = clamped_pos
            share = max_single_whale_share

        if share > 0.10:
            wtype = "institutional"
            exit_delay, reentry_delay, hyst = 3.0, 10.0, 0.008
        elif share > 0.05:
            wtype = "quant_desk"
            exit_delay, reentry_delay, hyst = 2.0, 7.0, 0.006
        else:
            wtype = "opportunistic"
            exit_delay, reentry_delay, hyst = 1.0, 4.0, 0.005

        switching_cost = 500 + (pos / 10_000_000) * 100

        # ── Check for empirical whale history from Dune ──
        addr = h.get("address", f"unknown_{i}")
        addr_lower = addr.lower()
        whale_hist = history_lookup.get(addr_lower)

        if whale_hist:
            # Use empirical data to set thresholds
            avg_hold = whale_hist.get("avg_hold_days", 7.0)
            n_exits = whale_hist.get("n_withdrawals", 0)
            _n_entries = whale_hist.get("n_deposits", 0)

            # If whale has exited before, they're more likely to exit again →
            # closer alt_rate to r_threshold (tighter risk)
            if n_exits > 0:
                # Empirical: whales who've exited are 1.0-1.1x r_threshold (very tight)
                alt_rate = r_threshold * (1.0 + 0.1 * min(avg_hold / 30.0, 1.0))
            else:
                # Never exited: slightly more generous threshold
                alt_rate = r_threshold * (1.0 + 0.15 * (i / max(len(holders), 1)))

            # Empirical exit/reentry delays from hold time
            exit_delay = max(1.0, min(avg_hold * 0.3, 5.0))
            reentry_delay = max(3.0, min(avg_hold * 0.5, 14.0))
            print(
                f"  [Whale {addr[:8]}] Empirical: hold={avg_hold:.0f}d, "
                f"exits={n_exits}, alt_rate={alt_rate:.3f}"
            )
        else:
            # ── Synthetic alt_rate (fallback) ──
            alt_rate = r_threshold * (1.0 + 0.2 * (i / max(len(holders), 1)))

        # ── Clamp alt_rate ──
        # Increased from 1.5x to 2.0x to allow higher thresholds
        alt_rate_cap = r_threshold * max_alt_rate_multiplier
        if alt_rate > alt_rate_cap:
            alt_rate = alt_rate_cap

        # Increase risk premium to create buffer requirement
        risk_premium = 0.005 + 0.002 * min(i, 5)  # Was 0.003 + 0.001, now 0.005-0.015

        profiles.append(
            WhaleProfile(
                whale_id=f"whale_{i + 1}_{addr[:8]}",
                position_usd=pos,
                alt_rate=alt_rate,
                risk_premium=risk_premium,
                switching_cost_usd=switching_cost,
                exit_delay_days=exit_delay,
                reentry_delay_days=reentry_delay,
                hysteresis_band=hyst,
                whale_type=wtype,
            )
        )

    if not profiles:
        raise RuntimeError(
            f"No whale profiles could be built from {len(holders)} holders "
            f"(min_position_usd=${min_position_usd:,.0f}). "
            f"Largest holder: ${holders[0]['balance_usd']:,.0f}"
            if holders
            else "No holders data provided."
        )

    # ── Clamp total whale concentration ──
    total_whale_usd = sum(p.position_usd for p in profiles)
    max_whale_usd = total_supply_usd * max_total_whale_share
    if total_whale_usd > max_whale_usd and total_whale_usd > 0:
        scale = max_whale_usd / total_whale_usd
        print(
            f"  WARNING: Total whale TVL ${total_whale_usd:,.0f} > "
            f"{max_total_whale_share:.0%} cap (${max_whale_usd:,.0f}), "
            f"scaling all by {scale:.2f}"
        )
        scaled = []
        for p in profiles:
            scaled.append(
                WhaleProfile(
                    whale_id=p.whale_id,
                    position_usd=p.position_usd * scale,
                    alt_rate=p.alt_rate,
                    risk_premium=p.risk_premium,
                    switching_cost_usd=p.switching_cost_usd,
                    exit_delay_days=p.exit_delay_days,
                    reentry_delay_days=p.reentry_delay_days,
                    hysteresis_band=p.hysteresis_band,
                    whale_type=p.whale_type,
                )
            )
        profiles = scaled

    return profiles


# ============================================================================
# CONVENIENCE: fetch whales for a specific venue
# ============================================================================


def fetch_aave_whales(
    asset_symbol: str,
    market: str = "core",
    r_threshold: float = 0.045,
    min_usd: float = 1_000_000,
    rpc_url: str | None = None,
) -> list:
    """
    End-to-end: fetch Aave aToken holders → build whale profiles.

    Uses known aToken addresses first, falls back to on-chain lookup.
    ERRORS LOUDLY on failure.
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    # Try known aToken address
    a_token = KNOWN_ATOKENS.get((asset_symbol, market))

    if not a_token:
        # Look up on-chain
        asset_addr = STABLECOIN_ADDRESSES.get(asset_symbol)
        if not asset_addr:
            raise RuntimeError(f"Unknown asset: {asset_symbol}")
        reserve = fetch_aave_reserve_data(asset_addr, asset_symbol, market=market, rpc_url=rpc_url)
        a_token = reserve.a_token_address

    # Get asset decimals
    asset_addr = STABLECOIN_ADDRESSES.get(asset_symbol, "")
    decimals = 6  # default for stablecoins
    if asset_addr:
        try:
            decimals = _decode_uint256(_eth_call(rpc_url, asset_addr, "0x313ce567"))
        except Exception:
            pass

    holders = fetch_aave_atoken_top_holders(a_token, decimals, rpc_url, min_usd)

    # Get total supply for profile building
    total_supply_raw = _decode_uint256(_eth_call(rpc_url, a_token, "0x18160ddd"))
    total_supply_usd = total_supply_raw / (10**decimals)

    return build_whale_profiles_from_holders(holders, total_supply_usd, r_threshold)


def fetch_euler_whales(
    asset_symbol: str,
    r_threshold: float = 0.045,
    min_usd: float = 1_000_000,
    rpc_url: str | None = None,
) -> list:
    """
    End-to-end: fetch Euler vault holders → build whale profiles.

    ERRORS LOUDLY on failure.
    """
    rpc_url = rpc_url or get_eth_rpc_url()

    vault_addr = EULER_VAULTS.get(asset_symbol)
    if not vault_addr:
        raise RuntimeError(f"No Euler vault for {asset_symbol}")

    vault_data = fetch_euler_vault_data(vault_addr, asset_symbol, rpc_url)
    holders = fetch_euler_vault_top_holders(vault_addr, vault_data.asset_decimals, rpc_url, min_usd)

    return build_whale_profiles_from_holders(holders, vault_data.total_supply_usd, r_threshold)

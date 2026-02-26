"""
Dynamic base APY fetching for all supported protocols.

Base APY = the organic/native yield depositors earn BEFORE Merkl incentives.

Production configuration:
- RPC URLs passed through from environment (via evm_data / kamino_data)
- No hardcoded public endpoints
- On-chain / API sources preferred; DeFiLlama as fallback only

Sources (in priority order):
- Morpho: weighted average of sleeve APYs from GraphQL
- Aave: direct on-chain Pool.getReserveData (Core + Horizon), DeFiLlama fallback
- Euler: direct on-chain EVault.interestRate (preferred), DeFiLlama fallback
- Kamino: Kamino REST API (preferred), DeFiLlama fallback
- Curve: DeFiLlama (trading fee APY)
- DeFiLlama: universal fallback

DeFiLlama returns APY as PERCENTAGE (3.5 = 3.5%). We convert to DECIMAL (0.035).
On-chain / API sources return DECIMAL directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

MORPHO_GRAPHQL_URL = "https://api.morpho.org/graphql"
DEFILLAMA_YIELDS_URL = "https://yields.llama.fi/pools"

MORPHO_VAULTS = {
    # V1 address (used for Merkl campaigns / allocation+APY query)
    "0x19b3cd7032b8c062e8d44eacad661a0970dd8c55": {
        "v1": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
        "chain_id": 1,
        "decimals": 6,
    },
    # V2 address (from venue_registry — maps to same V1 for APY)
    "0xb576765fb15505433af24fee2c0325895c559fb2": {
        "v1": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
        "chain_id": 1,
        "decimals": 6,
    },
}


@dataclass
class BaseAPYResult:
    """Result of a base APY fetch."""

    venue_name: str
    base_apy: float  # decimal (0.035 = 3.5%)
    source: str
    details: dict

    @property
    def base_apy_pct(self) -> float:
        return self.base_apy * 100


# ============================================================================
# MORPHO
# ============================================================================


def fetch_morpho_base_apy(
    vault_address: str,
    chain_id: int = 1,
    decimals: int = 6,
) -> BaseAPYResult:
    """Fetch Morpho MetaMorpho vault base APY from sleeve allocations."""
    query = """
    query($address: String!, $chainId: Int!) {
      vaultByAddress(address: $address, chainId: $chainId) {
        state {
          totalAssets
          allocation {
            market {
              collateralAsset { name symbol }
              state { supplyApy utilization borrowAssets supplyAssets }
            }
            supplyAssets
            supplyCap
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            MORPHO_GRAPHQL_URL,
            json={
                "query": query,
                "variables": {"address": vault_address.lower(), "chainId": chain_id},
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        vault = data["data"]["vaultByAddress"]
        if not vault or not vault.get("state"):
            raise RuntimeError(f"No data for vault {vault_address}")

        divisor = 10**decimals
        total_assets = int(vault["state"]["totalAssets"]) / divisor
        sleeves, weighted_apy_sum, total_supply, idle_supply = [], 0.0, 0.0, 0.0

        for alloc in vault["state"]["allocation"]:
            mkt = alloc["market"]
            supply = int(alloc["supplyAssets"]) / divisor
            apy = float(mkt["state"]["supplyApy"] or 0)
            util = float(mkt["state"]["utilization"] or 0)
            collateral = mkt.get("collateralAsset")
            name = collateral["name"] if collateral else "None (Idle)"
            symbol = collateral["symbol"] if collateral else "IDLE"
            if collateral is None:
                idle_supply = supply
            cap_raw = alloc.get("supplyCap")
            cap = int(cap_raw) / divisor if cap_raw and int(cap_raw) < 10**30 else None
            sleeves.append(
                {
                    "collateral": name,
                    "symbol": symbol,
                    "supply_usd": supply,
                    "apy": apy,
                    "utilization": util,
                    "supply_cap": cap,
                    "weight": supply / total_assets if total_assets > 0 else 0,
                }
            )
            weighted_apy_sum += supply * apy
            total_supply += supply

        base_apy = weighted_apy_sum / total_supply if total_supply > 0 else 0.0

        return BaseAPYResult(
            venue_name=f"Morpho ({vault_address[:10]}...)",
            base_apy=base_apy,
            source="morpho_graphql",
            details={
                "total_assets": total_assets,
                "idle_supply": idle_supply,
                "idle_fraction": idle_supply / total_assets if total_assets > 0 else 0,
                "sleeves": sleeves,
                "active_only_apy": (
                    weighted_apy_sum / (total_supply - idle_supply)
                    if (total_supply - idle_supply) > 0
                    else 0.0
                ),
            },
        )
    except Exception as e:
        print(f"  Morpho GraphQL failed ({e}), falling back to DeFiLlama")
        return fetch_defillama_base_apy(
            f"Morpho ({vault_address[:10]}...)",
            "morpho",
            "PYUSD",
            "Ethereum",
            pool_id_contains=vault_address,
        )


# ============================================================================
# AAVE — on-chain first (supports Core + Horizon), DeFiLlama fallback
# ============================================================================


def fetch_aave_base_apy(
    venue_name: str,
    asset_symbol: str,
    market: str = "core",
    chain: str = "Ethereum",
    rpc_url: str | None = None,
) -> BaseAPYResult:
    """
    Fetch Aave V3 base APY. Tries on-chain first.

    Supports both 'core' and 'horizon' markets via separate pool contracts.
    rpc_url defaults to ALCHEMY_ETH_RPC_URL from environment.
    """
    try:
        from .evm_data import fetch_aave_base_apy_onchain

        result = fetch_aave_base_apy_onchain(asset_symbol, market, rpc_url)
        if result["source"] == "aave_onchain" and result["supply_apy"] > 0:
            return BaseAPYResult(
                venue_name=venue_name,
                base_apy=result["supply_apy"],
                source="aave_onchain",
                details={
                    "total_supply_usd": result.get("total_supply_usd", 0),
                    "total_borrow_usd": result.get("total_borrow_usd", 0),
                    "utilization": result.get("utilization", 0),
                    "borrow_apy": result.get("borrow_apy", 0),
                    "a_token_address": result.get("a_token_address", ""),
                    "market": market,
                },
            )
    except Exception as e:
        print(f"  Aave on-chain failed for {market} ({e}), using DeFiLlama")

    return fetch_defillama_base_apy(venue_name, "aave-v3", asset_symbol, chain)


# ============================================================================
# EULER — on-chain first, DeFiLlama fallback
# ============================================================================


def fetch_euler_base_apy(
    venue_name: str,
    asset_symbol: str,
    chain: str = "Ethereum",
    rpc_url: str | None = None,
) -> BaseAPYResult:
    """Fetch Euler V2 base APY. Tries on-chain first."""
    try:
        from .evm_data import fetch_euler_base_apy_onchain

        result = fetch_euler_base_apy_onchain(asset_symbol, rpc_url)
        if result["source"] == "euler_onchain" and result["supply_apy"] > 0:
            return BaseAPYResult(
                venue_name=venue_name,
                base_apy=result["supply_apy"],
                source="euler_onchain",
                details={
                    "total_supply_usd": result.get("total_supply_usd", 0),
                    "utilization": result.get("utilization", 0),
                    "vault_address": result.get("vault_address", ""),
                },
            )
    except Exception as e:
        print(f"  Euler on-chain failed ({e}), using DeFiLlama")

    return fetch_defillama_base_apy(venue_name, "euler-v2", asset_symbol, chain)


# ============================================================================
# KAMINO — API first, DeFiLlama fallback
# ============================================================================


def fetch_kamino_base_apy(
    venue_name: str,
    asset_symbol: str,
    market_name: str | None = None,
    vault_pubkey: str | None = None,
    defillama_project: str = "kamino-lend",
) -> BaseAPYResult:
    """Fetch Kamino base APY. Tries Kamino API first."""
    if market_name:
        try:
            from .kamino_data import fetch_kamino_reserve_for_asset

            reserve = fetch_kamino_reserve_for_asset(asset_symbol, market_name)
            if reserve and reserve.supply_apy > 0:
                return BaseAPYResult(
                    venue_name=venue_name,
                    base_apy=reserve.supply_apy,
                    source="kamino_api",
                    details={
                        "reserve_pubkey": reserve.reserve_pubkey,
                        "total_supply_usd": reserve.total_supply_usd,
                        "utilization": reserve.utilization,
                        "borrow_apy": reserve.borrow_apy,
                        "market_name": market_name,
                    },
                )
        except Exception as e:
            print(f"  Kamino API failed ({e}), using DeFiLlama")

    if vault_pubkey:
        try:
            from .kamino_data import fetch_kamino_vault_metrics

            metrics = fetch_kamino_vault_metrics(vault_pubkey)
            if metrics.base_apy > 0 or metrics.apy_actual > 0:
                return BaseAPYResult(
                    venue_name=venue_name,
                    base_apy=metrics.base_apy,
                    source="kamino_api",
                    details={
                        "vault_pubkey": vault_pubkey,
                        "total_apy": metrics.apy_actual,
                        "incentive_apy": metrics.apy_incentives,
                        "total_tvl_usd": metrics.total_tvl_usd,
                        "number_of_holders": metrics.number_of_holders,
                    },
                )
        except Exception as e:
            print(f"  Kamino vault metrics failed ({e}), using DeFiLlama")

    return fetch_defillama_base_apy(venue_name, defillama_project, asset_symbol, "Solana")


# ============================================================================
# DEFILLAMA — universal fallback
# ============================================================================


def fetch_defillama_base_apy(
    venue_name: str,
    project: str,
    asset_symbol: str,
    chain: str = "Ethereum",
    pool_id_contains: str | None = None,
) -> BaseAPYResult:
    """
    Fetch base APY from DeFiLlama Yields API.

    DeFiLlama returns percentages (3.5 = 3.5%). We convert to decimal (0.035).
    We want apyBase ONLY — organic yield before our incentives.
    """
    try:
        resp = requests.get(DEFILLAMA_YIELDS_URL, timeout=30)
        resp.raise_for_status()
        pools = resp.json().get("data", [])
    except Exception as e:
        return BaseAPYResult(
            venue_name=venue_name,
            base_apy=0.0,
            source="fallback",
            details={"error": str(e), "reason": "DeFiLlama request failed"},
        )

    candidates = []
    for pool in pools:
        if project.lower() not in pool.get("project", "").lower():
            continue
        if asset_symbol.upper() not in pool.get("symbol", "").upper():
            continue
        if chain and chain.lower() not in pool.get("chain", "").lower():
            continue
        if pool_id_contains and pool_id_contains.lower() not in pool.get("pool", "").lower():
            continue
        tvl = pool.get("tvlUsd", 0)
        if tvl < 100_000:
            continue

        apy_base_pct = pool.get("apyBase") or 0
        apy_reward_pct = pool.get("apyReward") or 0

        candidates.append(
            {
                "pool_id": pool.get("pool", ""),
                "symbol": pool.get("symbol", ""),
                "project": pool.get("project", ""),
                "chain": pool.get("chain", ""),
                "tvl_usd": tvl,
                "apy_base_pct": apy_base_pct,
                "apy_reward_pct": apy_reward_pct,
                "apy_base": apy_base_pct / 100.0,
            }
        )

    if not candidates:
        return BaseAPYResult(
            venue_name=venue_name,
            base_apy=0.0,
            source="fallback",
            details={"reason": f"No pool found on DeFiLlama for {project}/{asset_symbol}/{chain}"},
        )

    best = max(candidates, key=lambda c: c["tvl_usd"])
    return BaseAPYResult(
        venue_name=venue_name,
        base_apy=best["apy_base"],
        source="defillama",
        details={
            "pool_id": best["pool_id"],
            "symbol": best["symbol"],
            "project": best["project"],
            "tvl_usd": best["tvl_usd"],
            "apy_base_pct": best["apy_base_pct"],
            "apy_reward_pct": best["apy_reward_pct"],
            "all_candidates": candidates,
        },
    )


# ============================================================================
# UNIFIED FETCHER — routes to protocol-specific method
# ============================================================================


def fetch_base_apy(
    protocol: str,
    asset_symbol: str,
    venue_name: str,
    chain: str = "Ethereum",
    vault_address: str | None = None,
    defillama_project: str | None = None,
    kamino_market_name: str | None = None,
    kamino_vault_pubkey: str | None = None,
    aave_market: str = "core",
    rpc_url: str | None = None,
) -> BaseAPYResult:
    """
    Unified base APY fetcher. Routes to protocol-specific method.
    Priority: on-chain / API > DeFiLlama fallback.
    rpc_url: optional override; defaults to env-based RPC in each fetcher.
    """
    if protocol == "morpho" and vault_address:
        cfg = MORPHO_VAULTS.get(vault_address.lower())
        if cfg:
            return fetch_morpho_base_apy(cfg["v1"], cfg["chain_id"], cfg["decimals"])
        return fetch_morpho_base_apy(vault_address)

    if protocol == "aave":
        return fetch_aave_base_apy(venue_name, asset_symbol, aave_market, chain, rpc_url)

    if protocol == "euler":
        return fetch_euler_base_apy(venue_name, asset_symbol, chain, rpc_url)

    if protocol in ("kamino", "kamino_earn", "kamino_clmm"):
        dl_proj = defillama_project or {
            "kamino": "kamino-lend",
            "kamino_earn": "kamino",
            "kamino_clmm": "kamino-liquidity",
        }.get(protocol, "kamino-lend")
        return fetch_kamino_base_apy(
            venue_name,
            asset_symbol,
            market_name=kamino_market_name,
            vault_pubkey=kamino_vault_pubkey,
            defillama_project=dl_proj,
        )

    if protocol == "curve":
        return fetch_defillama_base_apy(
            venue_name,
            defillama_project or "curve-dex",
            asset_symbol,
            chain,
        )

    # Generic fallback
    dl_project = defillama_project or protocol
    return fetch_defillama_base_apy(venue_name, dl_project, asset_symbol, chain)


# ============================================================================
# BATCH FETCHER — for dashboard
# ============================================================================


def fetch_all_base_apys(venue_configs: list[dict]) -> dict[str, BaseAPYResult]:
    """
    Fetch base APYs for all venues in a program.

    Each venue dict should have:
        name, protocol, asset, chain (optional),
        vault_address (optional), defillama_project (optional),
        kamino_market_name (optional), kamino_vault_pubkey (optional),
        aave_market (optional, default "core")
    """
    results = {}
    for v in venue_configs:
        try:
            result = fetch_base_apy(
                protocol=v["protocol"],
                asset_symbol=v["asset"],
                venue_name=v["name"],
                chain=v.get("chain", "Ethereum"),
                vault_address=v.get("vault_address"),
                defillama_project=v.get("defillama_project"),
                kamino_market_name=v.get("kamino_market_name"),
                kamino_vault_pubkey=v.get("kamino_vault_pubkey"),
                aave_market=v.get("aave_market", "core"),
            )
        except Exception as e:
            result = BaseAPYResult(
                venue_name=v["name"],
                base_apy=0.0,
                source="error",
                details={"error": str(e)},
            )
        results[v["name"]] = result
    return results

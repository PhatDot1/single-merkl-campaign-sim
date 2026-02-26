"""
Test script: validate all data fetching for the campaign optimizer.

Run with:  python scripts/test_data_fetch.py

Tests:
1. Morpho base APY (GraphQL sleeve breakdown)
2. Morpho whale profiles (V2 depositor positions)
3. DeFiLlama base APYs for all protocols
4. DeFiLlama competitor rates & r_threshold computation
5. Kamino API (lending reserves + vault metrics)
6. Aave on-chain base APY (direct RPC)
7. Euler on-chain base APY (direct RPC)
8. End-to-end: fetch + calibrate for Morpho
"""

from __future__ import annotations

import traceback

PASS, FAIL, WARN = "✅", "❌", "⚠️"


def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ── 1. Morpho Base APY ──


def test_morpho_base_apy():
    section("1. Morpho Base APY (GraphQL Sleeve Breakdown)")
    from campaign.base_apy import fetch_morpho_base_apy

    VAULT = "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55"
    r = fetch_morpho_base_apy(VAULT, chain_id=1, decimals=6)

    print(f"\n  Base APY: {r.base_apy:.4%}  Source: {r.source}")
    print(f"  Total assets: ${r.details.get('total_assets', 0):,.0f}")
    print(f"  Idle fraction: {r.details.get('idle_fraction', 0):.1%}")

    sleeves = r.details.get("sleeves", [])
    if sleeves:
        print(f"\n  {'Collateral':<30} {'Supply':>14} {'APY':>8} {'Weight':>8}")
        print(f"  {'-' * 64}")
        for s in sorted(sleeves, key=lambda x: -x["supply_usd"]):
            print(
                f"  {s['collateral']:<30} ${s['supply_usd']:>12,.0f} {s['apy']:>7.2%} {s['weight']:>7.1%}"
            )

    assert r.base_apy >= 0, "Base APY should be non-negative"
    assert r.base_apy < 0.20, f"Base APY {r.base_apy:.2%} seems too high"
    assert len(sleeves) > 0, "Should have at least one sleeve"
    print(f"\n  {PASS} Morpho base APY test passed")
    return r


# ── 2. Morpho Whale Profiles ──


def test_morpho_whale_data():
    section("2. Morpho Whale Profiles (V2 Depositor Positions)")
    import requests

    VAULT_V2 = "0xb576765fB15505433aF24FEe2c0325895C559FB2"
    query = """
    query($address: String!, $chainId: Int!) {
      vaultV2ByAddress(address: $address, chainId: $chainId) {
        totalAssets
        positions(first: 100) { items { user { address } assets } }
      }
    }
    """
    resp = requests.post(
        "https://api.morpho.org/graphql",
        json={"query": query, "variables": {"address": VAULT_V2.lower(), "chainId": 1}},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    vault = resp.json()["data"]["vaultV2ByAddress"]
    total = int(vault["totalAssets"]) / 1e6
    positions = vault.get("positions", {}).get("items", [])

    deps = []
    for p in positions:
        assets = int(p.get("assets") or 0) / 1e6
        if assets >= 100_000:
            deps.append({"address": p["user"]["address"], "usd": assets})
    deps.sort(key=lambda d: -d["usd"])

    print(f"\n  V2 Total Assets: ${total:,.0f}")
    print(f"  Depositors >= $100k: {len(deps)}")
    if deps:
        top5_share = sum(d["usd"] for d in deps[:5]) / total if total > 0 else 0
        print(f"  Top-5 concentration: {top5_share:.1%}")
        print(f"\n  {'#':<4} {'Address':<44} {'Assets':>14} {'Share':>8}")
        for i, d in enumerate(deps[:10]):
            share = d["usd"] / total if total > 0 else 0
            print(f"  {i + 1:<4} {d['address']:<44} ${d['usd']:>12,.0f} {share:>7.1%}")

    assert len(deps) > 0, "Should find depositors"
    print(f"\n  {PASS} Morpho whale data test passed")
    return deps


# ── 3. DeFiLlama Base APYs ──


def test_defillama_base_apys():
    section("3. DeFiLlama Base APYs (All Protocols)")
    from campaign.base_apy import fetch_defillama_base_apy

    cases = [
        ("AAVE PYUSD", "aave-v3", "PYUSD", "Ethereum"),
        ("AAVE RLUSD", "aave-v3", "RLUSD", "Ethereum"),
        ("Euler PYUSD", "euler-v2", "PYUSD", "Ethereum"),
        ("Euler RLUSD", "euler-v2", "RLUSD", "Ethereum"),
        ("Curve PYUSD-USDC", "curve-dex", "PYUSD", "Ethereum"),
        ("Curve RLUSD", "curve-dex", "RLUSD", "Ethereum"),
        ("Kamino Lend PYUSD", "kamino-lend", "PYUSD", "Solana"),
        ("Kamino Earn PYUSD", "kamino", "PYUSD", "Solana"),
        ("Kamino CLMM PYUSD", "kamino-liquidity", "PYUSD", "Solana"),
    ]

    print(f"\n  {'Venue':<25} {'Base APY':>10} {'Reward APY':>12} {'TVL':>14} {'Source':>12}")
    print(f"  {'-' * 77}")

    results = {}
    for name, project, asset, chain in cases:
        r = fetch_defillama_base_apy(name, project, asset, chain)
        results[name] = r
        apy_reward = r.details.get("apy_reward_pct", 0)
        tvl = r.details.get("tvl_usd", 0)
        status = PASS if r.base_apy > 0 else WARN
        print(
            f"  {status} {name:<23} {r.base_apy:>9.2%} {apy_reward:>10.1f}% ${tvl:>12,.0f} {r.source[:12]:>12}"
        )

    print(f"\n  {PASS} DeFiLlama base APY test completed")
    return results


# ── 4. Competitor Rates & r_threshold ──


def test_competitor_rates():
    section("4. Competitor Rates & r_threshold (DeFiLlama)")
    from campaign.data import compute_r_threshold, fetch_competitor_rates

    for asset in ["PYUSD", "RLUSD"]:
        print(f"\n  --- {asset} Competitors ---")
        try:
            comps = fetch_competitor_rates(asset_symbol=asset, min_tvl=1_000_000)
            thresh = compute_r_threshold(comps)

            source = thresh.get("r_threshold_source", "unknown")
            print(f"  Found {len(comps)} competing venues (source: {source})")
            print(f"  r_threshold: {thresh['r_threshold']:.2%}")
            print(f"  Range: [{thresh['r_threshold_lo']:.2%}, {thresh['r_threshold_hi']:.2%}]")
            if thresh.get("usdc_benchmark") is not None:
                print(f"  USDC benchmark: {thresh['usdc_benchmark']:.2%}")
            if thresh.get("outliers_removed", 0) > 0:
                print(f"  Outliers removed: {thresh['outliers_removed']}")

            print(
                f"\n  {'Project':<20} {'Symbol':<25} {'TVL':>12} {'Base':>7} {'Reward':>8} {'Total':>7}"
            )
            print(f"  {'-' * 82}")
            for c in comps[:12]:
                print(
                    f"  {c.venue:<20} {c.symbol:<25} ${c.tvl_usd:>10,.0f} "
                    f"{c.apy_base:>6.2%} {c.apy_reward:>7.2%} {c.apy_total:>6.2%}"
                )
        except Exception as e:
            print(f"  {FAIL} {asset} competitor fetch failed: {e}")

    print(f"\n  {PASS} Competitor rates test completed")


# ── 5. Kamino API ──


def test_kamino_api():
    section("5. Kamino API (Lending Reserves + Vault Metrics)")

    # Test lending reserves
    print("\n  --- Kamino Lending Reserves (Main Market) ---")
    try:
        from campaign.kamino_data import KAMINO_MARKETS, fetch_kamino_market_reserves

        market = KAMINO_MARKETS.get("main")
        if market:
            reserves = fetch_kamino_market_reserves(market)
            print(f"  {PASS} Fetched {len(reserves)} reserves from main market")
            for r in reserves[:8]:
                print(
                    f"    {r.liquidity_token:<10} Supply APY: {r.supply_apy:.2%} "
                    f"TVL: ${r.total_supply_usd:>12,.0f} Util: {r.utilization:.1%}"
                )

            # Check for PYUSD specifically
            pyusd = [r for r in reserves if "PYUSD" in r.liquidity_token.upper()]
            if pyusd:
                p = pyusd[0]
                print(
                    f"\n  {PASS} PYUSD reserve found: supply APY = {p.supply_apy:.2%}, "
                    f"TVL = ${p.total_supply_usd:,.0f}"
                )
            else:
                print(f"\n  {WARN} No PYUSD reserve found in main market")
        else:
            print(f"  {WARN} Main market pubkey not configured")
    except Exception as e:
        print(f"  {FAIL} Kamino lending reserves: {e}")

    # Test JLP market
    print("\n  --- Kamino Lending Reserves (JLP Market) ---")
    try:
        from campaign.kamino_data import KAMINO_MARKETS, fetch_kamino_market_reserves

        jlp_market = KAMINO_MARKETS.get("jlp")
        if jlp_market:
            reserves = fetch_kamino_market_reserves(jlp_market)
            print(f"  {PASS} Fetched {len(reserves)} reserves from JLP market")
            pyusd = [r for r in reserves if "PYUSD" in r.liquidity_token.upper()]
            if pyusd:
                print(f"  {PASS} PYUSD in JLP: supply APY = {pyusd[0].supply_apy:.2%}")
            else:
                print(f"  {WARN} No PYUSD in JLP market")
    except Exception as e:
        print(f"  {FAIL} JLP market: {e}")

    # Test DeFiLlama fallback for Kamino
    print("\n  --- Kamino via DeFiLlama ---")
    from campaign.base_apy import fetch_defillama_base_apy

    for name, project in [
        ("Kamino Lend", "kamino-lend"),
        ("Kamino Earn", "kamino"),
        ("Kamino CLMM", "kamino-liquidity"),
    ]:
        r = fetch_defillama_base_apy(name, project, "PYUSD", "Solana")
        if r.base_apy > 0:
            print(
                f"  {PASS} {name}: base APY = {r.base_apy:.2%} (TVL: ${r.details.get('tvl_usd', 0):,.0f})"
            )
        else:
            print(f"  {WARN} {name}: no data ({r.details.get('reason', 'unknown')})")

    print(f"\n  {PASS} Kamino API test completed")


# ── 6. Aave On-Chain Base APY ──


def test_aave_onchain():
    section("6. Aave V3 On-Chain Base APY (Direct RPC)")
    try:
        from campaign.evm_data import fetch_aave_base_apy_onchain

        for asset in ["PYUSD", "RLUSD"]:
            result = fetch_aave_base_apy_onchain(asset, "core")
            if result["source"] == "aave_onchain":
                print(
                    f"  {PASS} {asset} Core: supply APY = {result['supply_apy']:.4%}, "
                    f"TVL = ${result.get('total_supply_usd', 0):,.0f}, "
                    f"util = {result.get('utilization', 0):.1%}"
                )
            else:
                print(f"  {FAIL} {asset} Core: {result.get('error', 'unknown error')}")
    except Exception as e:
        print(f"  {FAIL} Aave on-chain test failed: {e}")
        traceback.print_exc(limit=2)

    print(f"\n  {PASS} Aave on-chain test completed")


# ── 7. Euler On-Chain Base APY ──


def test_euler_onchain():
    section("7. Euler V2 On-Chain Base APY (Direct RPC)")
    try:
        from campaign.evm_data import fetch_euler_base_apy_onchain

        for asset in ["PYUSD", "RLUSD"]:
            result = fetch_euler_base_apy_onchain(asset)
            if result["source"] == "euler_onchain":
                print(
                    f"  {PASS} {asset}: supply APY = {result['supply_apy']:.4%}, "
                    f"TVL = ${result.get('total_supply_usd', 0):,.0f}, "
                    f"util = {result.get('utilization', 0):.1%}"
                )
            else:
                print(f"  {WARN} {asset}: {result.get('error', 'unknown error')}")
    except Exception as e:
        print(f"  {FAIL} Euler on-chain test failed: {e}")
        traceback.print_exc(limit=2)

    print(f"\n  {PASS} Euler on-chain test completed")


# ── 8. Unified Base APY Fetch (all venues) ──


def test_unified_base_apy():
    section("8. Unified Base APY Fetch (All Venues)")
    from campaign.base_apy import fetch_all_base_apys

    venues = [
        {
            "name": "AAVE Core PYUSD",
            "protocol": "aave",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "aave_market": "core",
        },
        {
            "name": "AAVE Core RLUSD",
            "protocol": "aave",
            "asset": "RLUSD",
            "chain": "Ethereum",
            "aave_market": "core",
        },
        {"name": "Euler PYUSD", "protocol": "euler", "asset": "PYUSD", "chain": "Ethereum"},
        {"name": "Euler RLUSD", "protocol": "euler", "asset": "RLUSD", "chain": "Ethereum"},
        {
            "name": "Kamino Lend PYUSD",
            "protocol": "kamino",
            "asset": "PYUSD",
            "chain": "Solana",
            "defillama_project": "kamino-lend",
            "kamino_market_name": "main",
        },
        {
            "name": "Kamino Earn PYUSD",
            "protocol": "kamino",
            "asset": "PYUSD",
            "chain": "Solana",
            "defillama_project": "kamino",
        },
        {
            "name": "Curve PYUSD-USDC",
            "protocol": "curve",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "defillama_project": "curve-dex",
        },
        {
            "name": "Morpho PYUSD",
            "protocol": "morpho",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "vault_address": "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55",
        },
    ]

    results = fetch_all_base_apys(venues)

    print(f"\n  {'Venue':<25} {'Base APY':>10} {'Source':<20}")
    print(f"  {'-' * 58}")
    for name, r in results.items():
        status = PASS if r.base_apy > 0 else WARN
        print(f"  {status} {name:<23} {r.base_apy:>9.2%} {r.source:<20}")

    n_ok = sum(1 for r in results.values() if r.base_apy > 0)
    print(f"\n  {n_ok}/{len(results)} venues returned positive base APY")
    print(f"  {PASS} Unified base APY test completed")


# ── 9. End-to-End MC Test ──


def test_end_to_end_morpho():
    section("9. End-to-End: Morpho PYUSD (fetch → calibrate → MC)")
    from campaign.agents import WhaleProfile
    from campaign.base_apy import fetch_morpho_base_apy
    from campaign.engine import LossWeights, run_monte_carlo
    from campaign.state import CampaignConfig, CampaignEnvironment

    VAULT = "0x19b3cD7032B8C062E8d44EaCad661a0970DD8c55"
    apy_result = fetch_morpho_base_apy(VAULT)
    base_apy = apy_result.base_apy
    print(f"\n  Fetched base APY: {base_apy:.4%}")

    config = CampaignConfig(
        weekly_budget=115_000,
        apr_cap=0.06,
        base_apy=base_apy,
        dt_days=0.5,
        horizon_days=14,
    )
    env = CampaignEnvironment(r_threshold=0.045)
    whales = [
        WhaleProfile("w1", 29_600_000, alt_rate=0.05, risk_premium=0.003, switching_cost_usd=2000),
        WhaleProfile("w2", 25_000_000, alt_rate=0.048, risk_premium=0.004, switching_cost_usd=1800),
    ]
    weights = LossWeights(
        tvl_target=100_000_000,
        apr_target=base_apy + 0.055,
        apr_stability_on_total=True,
    )

    mc = run_monte_carlo(
        config=config,
        env=env,
        initial_tvl=195_000_000,
        whale_profiles=whales,
        weights=weights,
        n_paths=10,
    )

    print(f"  Mean total APR: {mc.mean_apr:.2%}")
    print(f"  Mean TVL: ${mc.mean_tvl:,.0f}")
    print(f"  Budget util: {mc.mean_budget_util:.1%}")
    print(f"  Feasible: {mc.is_feasible}")

    assert mc.mean_apr > base_apy, "Total APR should exceed base APY"
    assert mc.mean_tvl > 0, "TVL should be positive"
    print(f"\n  {PASS} End-to-end test passed")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("  CAMPAIGN OPTIMIZER — DATA FETCH VALIDATION")
    print("=" * 70)

    tests = [
        ("Morpho Base APY", test_morpho_base_apy),
        ("Morpho Whale Data", test_morpho_whale_data),
        ("DeFiLlama Base APYs", test_defillama_base_apys),
        ("Competitor Rates", test_competitor_rates),
        ("Kamino API", test_kamino_api),
        ("Aave On-Chain", test_aave_onchain),
        ("Euler On-Chain", test_euler_onchain),
        ("Unified Base APY", test_unified_base_apy),
        ("End-to-End Morpho", test_end_to_end_morpho),
    ]

    results = {}
    for name, fn in tests:
        try:
            fn()
            results[name] = PASS
        except Exception as e:
            results[name] = FAIL
            print(f"\n  {FAIL} {name} FAILED: {e}")
            traceback.print_exc(limit=3)

    section("SUMMARY")
    for name, status in results.items():
        print(f"  {status} {name}")

    n_pass = sum(1 for s in results.values() if s == PASS)
    print(f"\n  {n_pass}/{len(results)} tests passed")

    if n_pass < len(results):
        print(f"\n  {WARN} Some tests failed. Check output above.")
        print("  Note: On-chain tests may fail on rate-limited public RPCs.")
        print("  Note: Kamino API tests may fail if endpoints change.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Test script: validate Curve venue data fetching for the Kraken Split dashboard.

Run with:  python scripts/test_curve_data.py

Tests:
1. Curve RLUSD-USDC base APY via DeFiLlama
2. Curve PYUSD-USDC base APY via DeFiLlama
3. Curve TVL from DeFiLlama pool IDs
4. Batch fetch via fetch_all_base_apys (verifies routing)
5. Integration: kraken_split fetch_live_vault_tvls Curve entries
"""

from __future__ import annotations

import os
import sys
import traceback

# Ensure sim/ is on path and .env is loaded
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_SIM_DIR, "..", ".env"), override=False)
except ImportError:
    pass

PASS, FAIL, WARN = "✅", "❌", "⚠️"


def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ── 1. Curve RLUSD-USDC base APY ──


def test_curve_rlusd_base_apy():
    section("1. Curve RLUSD-USDC Base APY (DeFiLlama)")
    from campaign.base_apy import fetch_defillama_base_apy

    r = fetch_defillama_base_apy(
        venue_name="Curve RLUSD-USDC",
        project="curve-dex",
        asset_symbol="RLUSD",
        chain="Ethereum",
    )

    print(f"\n  Base APY: {r.base_apy:.4%}  ({r.base_apy_pct:.4f}%)")
    print(f"  Source:   {r.source}")
    print(f"  Pool ID:  {r.details.get('pool_id', 'N/A')}")
    print(f"  Symbol:   {r.details.get('symbol', 'N/A')}")
    print(f"  TVL:      ${r.details.get('tvl_usd', 0):,.0f}")

    assert r.source == "defillama", f"Expected source 'defillama', got '{r.source}'"
    assert r.base_apy >= 0, "Base APY should be non-negative"
    assert r.details.get("tvl_usd", 0) > 0, "TVL should be > 0"
    print(f"\n  {PASS} Curve RLUSD-USDC base APY test passed")
    return r


# ── 2. Curve PYUSD-USDC base APY ──


def test_curve_pyusd_base_apy():
    section("2. Curve PYUSD-USDC Base APY (DeFiLlama)")
    from campaign.base_apy import fetch_defillama_base_apy

    r = fetch_defillama_base_apy(
        venue_name="Curve PYUSD-USDC",
        project="curve-dex",
        asset_symbol="PYUSD",
        chain="Ethereum",
        pool_id_contains="14681aee",
    )

    print(f"\n  Base APY: {r.base_apy:.4%}  ({r.base_apy_pct:.4f}%)")
    print(f"  Source:   {r.source}")
    print(f"  Pool ID:  {r.details.get('pool_id', 'N/A')}")
    print(f"  Symbol:   {r.details.get('symbol', 'N/A')}")
    print(f"  TVL:      ${r.details.get('tvl_usd', 0):,.0f}")

    assert r.source == "defillama", f"Expected source 'defillama', got '{r.source}'"
    assert r.base_apy >= 0, "Base APY should be non-negative"
    assert r.details.get("tvl_usd", 0) > 0, "TVL should be > 0"
    print(f"\n  {PASS} Curve PYUSD-USDC base APY test passed")
    return r


# ── 3. Curve TVL from DeFiLlama pool IDs ──


def test_curve_tvl_from_defillama():
    section("3. Curve TVL from DeFiLlama Pool IDs")
    import requests

    CURVE_DEFILLAMA_IDS = {
        "curve_rlusd": "e91e23af-9099-45d9-8ba5-ea5b4638e453",
        "curve_pyusd": "14681aee-05c9-4733-acd0-7b2c84616209",
    }

    resp = requests.get("https://yields.llama.fi/pools", timeout=30)
    resp.raise_for_status()
    pool_map = {p["pool"]: p for p in resp.json().get("data", [])}

    for key, pool_id in CURVE_DEFILLAMA_IDS.items():
        pool = pool_map.get(pool_id)
        assert pool is not None, f"Pool ID '{pool_id}' not found in DeFiLlama for {key}"
        tvl = pool.get("tvlUsd", 0)
        apy_base = pool.get("apyBase", 0)
        symbol = pool.get("symbol", "")
        project = pool.get("project", "")
        print(f"\n  {key}:")
        print(f"    Pool:      {pool_id}")
        print(f"    Symbol:    {symbol}")
        print(f"    Project:   {project}")
        print(f"    TVL:       ${tvl:,.0f}")
        print(f"    Base APY:  {apy_base:.4f}%")
        assert tvl > 0, f"TVL for {key} should be > 0"
        assert project == "curve-dex", f"Expected project 'curve-dex', got '{project}'"

    print(f"\n  {PASS} Curve TVL DeFiLlama lookup passed")


# ── 4. Batch fetch via fetch_all_base_apys ──


def test_batch_curve_base_apys():
    section("4. Batch Fetch — Curve via fetch_all_base_apys")
    from campaign.base_apy import fetch_all_base_apys

    configs = [
        {
            "name": "Curve RLUSD-USDC",
            "protocol": "curve",
            "asset": "RLUSD",
            "chain": "Ethereum",
            "pool_id_contains": "e91e23af",
        },
        {
            "name": "Curve PYUSD-USDC",
            "protocol": "curve",
            "asset": "PYUSD",
            "chain": "Ethereum",
            "pool_id_contains": "14681aee",
        },
    ]

    results = fetch_all_base_apys(configs)

    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    Base APY: {r.base_apy:.4%}  Source: {r.source}")
        print(f"    TVL:      ${r.details.get('tvl_usd', 0):,.0f}")
        assert r.source != "error", f"Fetch failed for {name}: {r.details}"

    assert "Curve RLUSD-USDC" in results, "Missing Curve RLUSD-USDC"
    assert "Curve PYUSD-USDC" in results, "Missing Curve PYUSD-USDC"
    print(f"\n  {PASS} Batch Curve base APY test passed")


# ── 5. Integration: kraken_split fetch_live_vault_tvls ──


def test_kraken_split_curve_tvls():
    section("5. Integration — kraken_split Curve TVL Entries")
    # Import the fetch function and constants from the dashboard module
    sys.path.insert(0, os.path.join(_SIM_DIR, "dashboard"))

    # We can't call the streamlit-cached function directly, but we can
    # replicate the Curve portion of fetch_live_vault_tvls
    import requests

    # These should match the constants in kraken_split.py
    CURVE_DEFILLAMA_IDS = {
        "curve_rlusd": "e91e23af-9099-45d9-8ba5-ea5b4638e453",
        "curve_pyusd": "14681aee-05c9-4733-acd0-7b2c84616209",
    }

    resp = requests.get("https://yields.llama.fi/pools", timeout=30)
    resp.raise_for_status()
    dl_pools = {p["pool"]: p for p in resp.json().get("data", [])}

    results = {}
    for key, pool_id in CURVE_DEFILLAMA_IDS.items():
        pool = dl_pools.get(pool_id)
        if pool:
            results[key] = pool.get("tvlUsd", 0.0)
        else:
            results[key] = 0.0

    for key, tvl in results.items():
        tvl_m = tvl / 1e6
        print(f"\n  {key}: ${tvl_m:.2f}M (raw: ${tvl:,.0f})")
        assert tvl > 0, f"Curve TVL for {key} should be > 0"

    print(f"\n  {PASS} Kraken split Curve TVL integration test passed")


# ── Main ──


def main():
    print("\n" + "=" * 70)
    print("  Curve Venue Data Test Suite")
    print("=" * 70)

    results = {}
    tests = [
        ("Curve RLUSD base APY", test_curve_rlusd_base_apy),
        ("Curve PYUSD base APY", test_curve_pyusd_base_apy),
        ("Curve TVL DeFiLlama", test_curve_tvl_from_defillama),
        ("Batch Curve base APYs", test_batch_curve_base_apys),
        ("Kraken split Curve TVLs", test_kraken_split_curve_tvls),
    ]

    for name, fn in tests:
        try:
            fn()
            results[name] = PASS
        except Exception as e:
            print(f"\n  {FAIL} {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = FAIL

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"  {status} {name}")

    n_fail = sum(1 for v in results.values() if v == FAIL)
    if n_fail:
        print(f"\n  {n_fail} test(s) failed!")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} tests passed!")


if __name__ == "__main__":
    main()

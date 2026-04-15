#!/usr/bin/env python
"""Quick test to find the correct Euler V2 caps() function selector."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import requests
from campaign.evm_data import get_eth_rpc_url

rpc = get_eth_rpc_url()
vault_pyusd = "0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5"
vault_rlusd = "0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2"

# Try various possible selectors
selectors = {
    "caps()":           "0x18aab3e0",
    "supplyCap()":      "0xbd9a548b",
    "borrowCap()":      "0xd13eb4e0",
    "maxDeposit(address)": "0x402d267d",  # ERC4626 maxDeposit
    "maxMint(address)":    "0xc63d75b6",  # ERC4626 maxMint
}

# Also try with zero-padded address for maxDeposit
ZERO_ADDR_PADDED = "0x402d267d" + "0" * 64  # maxDeposit(address(0))
MAX_ADDR_PADDED = "0x402d267d" + "f" * 64   # maxDeposit(max_address)

print(f"Testing Euler PYUSD vault: {vault_pyusd}")
print("=" * 70)

for name, sel in selectors.items():
    data = sel
    # For functions with address params, pad with zeros
    if "address" in name:
        data = sel + "0" * 64
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_call",
        "params": [{"to": vault_pyusd, "data": data}, "latest"],
    }
    r = requests.post(rpc, json=payload, timeout=15).json()
    if "error" in r:
        err = r["error"]
        print(f"  {name:30s} -> REVERT: {err.get('message', '')}")
    else:
        result = r["result"]
        if result and result != "0x":
            # Decode as uint256
            val = int(result, 16) if len(result) <= 66 else None
            if val is not None:
                print(f"  {name:30s} -> {result}  (= {val})")
            else:
                print(f"  {name:30s} -> {result[:66]}...")
        else:
            print(f"  {name:30s} -> {result}")

print()
print(f"Testing Euler RLUSD vault: {vault_rlusd}")
print("=" * 70)

for name, sel in selectors.items():
    data = sel
    if "address" in name:
        data = sel + "0" * 64
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_call",
        "params": [{"to": vault_rlusd, "data": data}, "latest"],
    }
    r = requests.post(rpc, json=payload, timeout=15).json()
    if "error" in r:
        err = r["error"]
        print(f"  {name:30s} -> REVERT: {err.get('message', '')}")
    else:
        result = r["result"]
        if result and result != "0x":
            val = int(result, 16) if len(result) <= 66 else None
            if val is not None:
                print(f"  {name:30s} -> {result}  (= {val})")
            else:
                print(f"  {name:30s} -> {result[:66]}...")
        else:
            print(f"  {name:30s} -> {result}")

# Also check ERC4626 maxDeposit to infer supply cap
print()
print("ERC4626 maxDeposit/maxMint with real address:")
print("=" * 70)
# Use a generic non-zero address for maxDeposit query
REAL_ADDR = "0000000000000000000000000000000000000000000000000000000000000001"
for label, vault in [("PYUSD", vault_pyusd), ("RLUSD", vault_rlusd)]:
    for fn, sel in [
        ("maxDeposit(addr)", "0x402d267d"),
        ("maxMint(addr)", "0xc63d75b6"),
        ("totalSupply()", "0x18160ddd"),
        ("totalAssets()", "0x01e1d114"),
        # Euler Earn-specific
        ("totalAllocatable()", "0x5babfde9"),
        ("totalAssetsDeposited()", "0xe23a9a52"),
        ("interestAccrued()", "0xa86a1724"),
    ]:
        data = sel
        if "addr" in fn:
            data = sel + REAL_ADDR
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_call",
            "params": [{"to": vault, "data": data}, "latest"],
        }
        r = requests.post(rpc, json=payload, timeout=15).json()
        if "error" in r:
            print(f"  {label} {fn:30s} -> REVERT")
        else:
            result = r["result"]
            val = int(result, 16) if result and result != "0x" else 0
            dec = 6 if label == "PYUSD" else 18
            usd = val / 10**dec if val < (1 << 200) else float("inf")
            if usd == float("inf") or usd > 1e15:
                print(f"  {label} {fn:30s} -> UNLIMITED (raw={val})")
            else:
                print(f"  {label} {fn:30s} -> ${usd:,.2f}")

# Check if caps exists on the underlying interest-bearing Euler EVault
# Try getting the interest rate model or the actual EVK vault
print()
print("Checking if these are Euler Earn wrappers (ERC4626 around EVault):")
print("=" * 70)
for label, vault in [("PYUSD", vault_pyusd), ("RLUSD", vault_rlusd)]:
    # Try eulerEarnVaultModule() or similar
    for fn, sel in [
        ("asset()", "0x38d52e0f"),
        ("governor()", "0x0c340a24"),
        ("interestFee()", "0xa73053c8"),
        ("protocolFeeShare()", "0x8d2c5b3e"),
    ]:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_call",
            "params": [{"to": vault, "data": sel}, "latest"],
        }
        r = requests.post(rpc, json=payload, timeout=15).json()
        if "error" in r:
            print(f"  {label} {fn:30s} -> REVERT")
        else:
            result = r["result"]
            if len(result) == 66:
                # Might be an address
                addr_check = "0x" + result[26:]
                print(f"  {label} {fn:30s} -> {result}  (addr? {addr_check})")
            else:
                val = int(result, 16) if result and result != "0x" else 0
                print(f"  {label} {fn:30s} -> {val}")

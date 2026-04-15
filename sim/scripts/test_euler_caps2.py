#!/usr/bin/env python
"""
Test: Euler V2 supply cap discovery via on-chain methods.

These Sentora/Euler Earn vaults don't support caps() directly.
Try: maxDeposit(address) + totalAssets() to infer supply cap headroom.
Also test EVC-based lookup.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import json, requests
from campaign.evm_data import _eth_call, _decode_uint256, get_eth_rpc_url

rpc = get_eth_rpc_url()
PYUSD_VAULT = "0xba98fC35C9dfd69178AD5dcE9FA29c64554783b5"
RLUSD_VAULT = "0xaF5372792a29dC6b296d6FFD4AA3386aff8f9BB2"

# ERC4626 maxDeposit uses receiver address — try with common addresses
ADDRS_TO_TEST = [
    ("0x0...1", "0000000000000000000000000000000000000000000000000000000000000001"),
    ("Kraken hot wallet", "000000000000000000000000" + "DA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf"[2:].lower()),
    ("Random EOA", "000000000000000000000000" + "d8dA6BF26964aF9D7eEd9e03E53415D37aA96045"[2:].lower()),
]

print("=" * 70)
print("  maxDeposit Test with Different Addresses")
print("=" * 70)

for label, vault, dec in [("PYUSD", PYUSD_VAULT, 6), ("RLUSD", RLUSD_VAULT, 18)]:
    total_assets = _decode_uint256(_eth_call(rpc, vault, "0x01e1d114"))
    tvl_usd = total_assets / 10**dec
    print(f"\n{label} vault ({vault[:10]}...): totalAssets = ${tvl_usd:,.2f}")
    
    for addr_label, addr_padded in ADDRS_TO_TEST:
        data = "0x402d267d" + addr_padded
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_call",
            "params": [{"to": vault, "data": data}, "latest"],
        }
        r = requests.post(rpc, json=payload, timeout=15).json()
        if "error" in r:
            print(f"  maxDeposit({addr_label}): REVERT")
        else:
            result = r["result"]
            val = int(result, 16) if result and result != "0x" else 0
            usd = val / 10**dec
            if val > (1 << 200):
                print(f"  maxDeposit({addr_label}): UNLIMITED (type(uint256).max)")
            elif val == 0:
                print(f"  maxDeposit({addr_label}): 0 (AT CAP or deposits disabled)")
            else:
                print(f"  maxDeposit({addr_label}): ${usd:,.2f} remaining")
                inferred_cap = tvl_usd + usd
                print(f"    => Inferred supply cap: ${inferred_cap:,.2f}")

# Try the Euler V2 EVK perspective/lens contract
# The EulerVaultLens on mainnet
LENS_CANDIDATES = [
    "0x46739Cad3cc10dEc2a6C3D01843b705B94B73A04",  # known EulerVaultLens
    "0x42Cd409Dc1C1E4667C33D40cA12EaD5b3f8E7e81",  # alternate lens
]

print("\n" + "=" * 70)
print("  Attempting Euler Lens Contract Queries")
print("=" * 70)

# getVaultInfoFull(address vault) selector
# Let's try different lens functions
for lens in LENS_CANDIDATES:
    print(f"\nTrying lens: {lens}")
    # Try a simple call to verify the contract exists
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "eth_getCode",
        "params": [lens, "latest"],
    }
    r = requests.post(rpc, json=payload, timeout=15).json()
    code = r.get("result", "0x")
    if code == "0x" or len(code) < 10:
        print(f"  No contract at this address")
        continue
    print(f"  Contract found (code length: {len(code)} hex chars)")

"""
Dune Analytics SQL query templates for campaign simulation data.

Each query is parameterized with:
- {{token_address}}: aToken/eToken/vault share address
- {{min_amount}}: Minimum transfer size in token units (e.g. 500000 for $500K stables)
- {{days}}: Lookback period in days
- {{chain}}: Network name (ethereum)

These templates are used by sync.py to execute queries via the Dune API.
"""

# ============================================================================
# WHALE FLOWS: Large ERC20 transfers for aToken/eToken/vault share tokens
# ============================================================================
# Detects deposits (mint from 0x0) and withdrawals (burn to 0x0)
# for lending protocol share tokens. Captures whale-sized movements.

WHALE_FLOWS_QUERY = """
-- Whale deposit/withdrawal flows for lending protocol share tokens
-- Mints (from 0x0) = deposits into protocol
-- Burns (to 0x0) = withdrawals from protocol
-- Regular transfers tracked as secondary signal

WITH transfers AS (
    SELECT
        block_time,
        "from" AS from_address,
        "to" AS to_address,
        CAST(value AS DOUBLE) / POWER(10, {{decimals}}) AS amount,
        tx_hash
    FROM {{chain}}.erc20_{{chain}}.evt_Transfer
    WHERE contract_address = LOWER('{{token_address}}')
      AND block_time >= NOW() - INTERVAL '{{days}}' DAY
      AND CAST(value AS DOUBLE) / POWER(10, {{decimals}}) >= {{min_amount}}
),
classified AS (
    SELECT
        block_time,
        CASE
            WHEN from_address = 0x0000000000000000000000000000000000000000
                THEN to_address
            ELSE from_address
        END AS whale_address,
        amount,
        CASE
            WHEN from_address = 0x0000000000000000000000000000000000000000
                THEN 'deposit'
            WHEN to_address = 0x0000000000000000000000000000000000000000
                THEN 'withdrawal'
            ELSE 'transfer'
        END AS direction,
        tx_hash
    FROM transfers
)
SELECT
    block_time,
    whale_address,
    amount,
    direction,
    tx_hash
FROM classified
ORDER BY block_time DESC
LIMIT 2000
"""


# ============================================================================
# MERCENARY DETECTION: Rapid in/out patterns within N days
# ============================================================================
# Identifies addresses that deposit AND withdraw within a short window,
# indicating yield-chasing / mercenary behavior.

MERCENARY_DETECTION_QUERY = """
-- Detect mercenary capital: addresses with deposit + withdrawal within 7 days
WITH flows AS (
    SELECT
        block_time,
        CASE
            WHEN "from" = 0x0000000000000000000000000000000000000000
                THEN "to"
            ELSE "from"
        END AS address,
        CAST(value AS DOUBLE) / POWER(10, {{decimals}}) AS amount,
        CASE
            WHEN "from" = 0x0000000000000000000000000000000000000000
                THEN 'deposit'
            WHEN "to" = 0x0000000000000000000000000000000000000000
                THEN 'withdrawal'
            ELSE 'transfer'
        END AS direction,
        tx_hash
    FROM {{chain}}.erc20_{{chain}}.evt_Transfer
    WHERE contract_address = LOWER('{{token_address}}')
      AND block_time >= NOW() - INTERVAL '{{days}}' DAY
      AND CAST(value AS DOUBLE) / POWER(10, {{decimals}}) >= {{min_amount}}
),
deposits AS (
    SELECT address, MIN(block_time) AS first_deposit, SUM(amount) AS total_deposited
    FROM flows WHERE direction = 'deposit'
    GROUP BY address
),
withdrawals AS (
    SELECT address, MIN(block_time) AS first_withdrawal, SUM(amount) AS total_withdrawn
    FROM flows WHERE direction = 'withdrawal'
    GROUP BY address
)
SELECT
    d.address,
    d.total_deposited,
    w.total_withdrawn,
    d.first_deposit,
    w.first_withdrawal,
    DATE_DIFF('day', d.first_deposit, w.first_withdrawal) AS days_held
FROM deposits d
INNER JOIN withdrawals w ON d.address = w.address
WHERE DATE_DIFF('day', d.first_deposit, w.first_withdrawal) <= 7
  AND DATE_DIFF('day', d.first_deposit, w.first_withdrawal) >= 0
ORDER BY d.total_deposited DESC
LIMIT 500
"""


# ============================================================================
# HISTORICAL APR SUPPLEMENT: On-chain rate data (Aave/Euler specific)
# ============================================================================
# Optional supplement to DeFiLlama data — direct on-chain rate queries.

HISTORICAL_APR_QUERY = """
-- Historical supply rates from Aave/Euler events
-- This is a fallback when DeFiLlama history is insufficient
SELECT
    block_time,
    CAST(value AS DOUBLE) / POWER(10, 27) AS supply_rate_ray,
    tx_hash
FROM {{chain}}.logs
WHERE contract_address = LOWER('{{pool_address}}')
  AND topic0 = 0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a  -- ReserveDataUpdated
  AND block_time >= NOW() - INTERVAL '{{days}}' DAY
ORDER BY block_time DESC
LIMIT 5000
"""


def render_query(template: str, **params) -> str:
    """
    Render a SQL template with parameters.

    Replaces {{key}} with the provided value. Simple string substitution
    (no SQL injection concerns — these run against Dune's API, not a live DB).

    Args:
        template: SQL template with {{param}} placeholders
        **params: Key-value pairs to substitute

    Returns:
        Rendered SQL string
    """
    sql = template
    for key, value in params.items():
        sql = sql.replace(f"{{{{{key}}}}}", str(value))
    return sql

"""Wake-up detection: find spends of outputs that had gone dormant.

The detector is pure with respect to the outside world. It receives a
transaction, the height at which it was mined, and an age threshold, and
decides whether that transaction spent any long-dormant output. Keeping
the chain client out of this module makes the rule testable without network.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

from .models import Tx, WakeUp

BLOCKS_PER_YEAR = 52_560  # 10-minute target block time


def blocks_to_years(blocks: int) -> float:
    return blocks / BLOCKS_PER_YEAR


def detect_wakeups(
    tx: Tx,
    spend_height: int,
    dormant_after_blocks: int,
    min_spent_sats: int = 0,
    observed_at: Optional[float] = None,
) -> list[WakeUp]:
    """Return one WakeUp per dormant input spent by `tx`.

    An input qualifies when its prevout has a known mining height, that
    height is at least `dormant_after_blocks` behind the spend, and the
    value clears the noise floor. Inputs whose prevout height is unknown
    are skipped rather than guessed at.
    """
    stamp = time.time() if observed_at is None else observed_at
    wakeups: list[WakeUp] = []

    for tx_input in tx.inputs:
        height = tx_input.prevout_block_height
        value = tx_input.prevout_value_sats
        if height is None or value is None:
            continue
        age = spend_height - height
        if age < dormant_after_blocks:
            continue
        if value < min_spent_sats:
            continue
        wakeups.append(
            WakeUp(
                txid=tx.txid,
                spend_block_height=spend_height,
                spent=tx_input.outpoint,
                value_sats=value,
                address=tx_input.prevout_address,
                dormant_blocks=age,
                dormant_years=blocks_to_years(age),
                script_type=tx_input.prevout_script_type,
                observed_at=stamp,
            )
        )

    return wakeups


def scan_transactions(
    transactions: Iterable[tuple[Tx, int]],
    dormant_after_blocks: int,
    min_spent_sats: int = 0,
    observed_at: Optional[float] = None,
) -> list[WakeUp]:
    """Run detection over many (tx, spend_height) pairs."""
    found: list[WakeUp] = []
    for tx, height in transactions:
        found.extend(
            detect_wakeups(
                tx,
                height,
                dormant_after_blocks,
                min_spent_sats=min_spent_sats,
                observed_at=observed_at,
            )
        )
    return found
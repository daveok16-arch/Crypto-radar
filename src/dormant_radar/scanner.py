"""Scan orchestration: walk recent blocks, detect wake-ups, score, persist.

A scan is a function, not a loop, so it can be driven by a one-shot CLI
call, a test, or the background worker without duplicating logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .chain import MempoolClient
from .cluster import OwnershipGraph
from .config import Settings
from .detector import detect_wakeups
from .anomaly import AnomalyDetector
from .ensemble import combine
from .models import WakeUp
from .price import PriceContext
from .store import Store

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    tip_height: int = 0
    scanned_from: int = 0
    scanned_to: int = 0
    blocks_examined: int = 0
    transactions_examined: int = 0
    wakeups: list[WakeUp] = field(default_factory=list)
    new_wakeups: int = 0
    errors: list[str] = field(default_factory=list)
    # The ownership graph actually used, so callers can reuse this connection
    # for backfill instead of opening a second one to the same database.
    graph: Optional[OwnershipGraph] = None


def scan_once(
    settings: Settings,
    client: Optional[MempoolClient] = None,
    store: Optional[Store] = None,
    price: Optional[PriceContext] = None,
    price_multiple: Optional[float] = None,
    detector: Optional[AnomalyDetector] = None,
    graph: Optional[OwnershipGraph] = None,
    load_model: bool = True,
) -> ScanResult:
    """Scan the most recent window of blocks for dormant-output spends.

    Transactions are fed into the ownership graph *before* their wake-ups are
    scored, so a spend is judged against everything we have already learned
    about who controls those addresses.
    """
    client = client or MempoolClient(settings)
    store = store or Store(settings.db_path)
    price = price or PriceContext()

    if graph is None and settings.use_clustering:
        graph = OwnershipGraph(settings.cluster_path)

    if detector is None and load_model and settings.use_neural:
        detector = AnomalyDetector.load(settings.model_path)
        if detector is None:
            logger.info(
                "no anomaly model at %s; scoring with symbolic rules only "
                "(run `dormant-radar train` to enable the neural layer)",
                settings.model_path,
            )

    result = ScanResult()

    tip = client.tip_height()
    result.tip_height = tip

    last_scanned = store.get_state("last_scanned_height")
    if isinstance(last_scanned, int) and last_scanned < tip:
        start = last_scanned + 1
    else:
        start = tip - settings.scan_window_blocks + 1
    start = max(start, 1)

    # Never walk backward through history in one poll; bound the window.
    end = min(tip, start + settings.scan_window_blocks - 1)
    result.scanned_from, result.scanned_to = start, end

    if price_multiple is None:
        price_multiple = price.price_multiple()

    for height in range(start, end + 1):
        block_hash = client.block_hash(height)
        if not block_hash:
            result.errors.append(f"no block hash at height {height}")
            continue
        result.blocks_examined += 1
        txids = client.block_txids(block_hash)[: settings.max_txs_per_block]
        for txid in txids:
            try:
                tx = client.transaction(txid)
            except Exception as exc:  # a single bad tx must not kill the scan
                result.errors.append(f"{txid}: {exc}")
                continue
            if tx is None:
                continue
            result.transactions_examined += 1

            # Learn ownership from this transaction before judging it.
            if graph is not None:
                try:
                    graph.observe(tx)
                except Exception as exc:  # clustering must never kill a scan
                    result.errors.append(f"cluster {txid}: {exc}")

            found = detect_wakeups(
                tx,
                spend_height=height,
                dormant_after_blocks=settings.dormant_after_blocks,
                min_spent_sats=settings.min_spent_sats,
            )
            for wakeup in found:
                ownership = None
                if graph is not None:
                    ownership = graph.summary_for(wakeup.address, tx, wakeup.value_sats)
                hybrid = combine(
                    wakeup,
                    tx=tx,
                    price_multiple=price_multiple,
                    detector=detector,
                    ownership=ownership,
                )
                wakeup.cause = hybrid.cause
                wakeup.neural = hybrid.as_dict()
                if ownership is not None:
                    wakeup.ownership = ownership.as_dict()
                result.wakeups.append(wakeup)

    result.new_wakeups = store.add_wakeups(result.wakeups)
    store.set_state("last_scanned_height", end)
    store.set_state("tip_height", tip)
    result.graph = graph

    logger.info(
        "scan complete: blocks %d-%d, %d txs, %d wake-ups (%d new)",
        start,
        end,
        result.transactions_examined,
        len(result.wakeups),
        result.new_wakeups,
    )
    return result
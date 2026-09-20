"""Address-history backfill for the ownership graph.

The problem this solves
-----------------------
Clustering learns only from what it has scanned. A fresh database therefore
understates ownership: an address that spent alongside others five years ago
looks isolated today, because we never saw that transaction. Every hour of
running improves coverage, so the earliest hours are the weakest — exactly
backwards from when an operator wants an answer.

Backfill walks an address's *historical* transactions, so co-spend partners
are discovered immediately rather than after days of continuous scanning.

Why this is affordable
----------------------
`/address/{addr}/txs` payloads embed the spent output's address under
`vin[].prevout`. One request therefore yields every co-spend partner of every
transaction, with no per-input blowup. Pagination is via `?after_txid=`, and
the endpoint returns non-overlapping pages.

Budget, not completeness
------------------------
Backfilling an address fully could mean hundreds of pages. That is unacceptable
inside a scan cycle, and a bounded sweep that never finishes is worse than one
that always makes progress. So backfill is:
- **budgeted** on both addresses and pages per run,
- **fair**, processing the least-recently-processed address first so no address
  can starve the rest,
- **incremental**, recording per-address progress so each run resumes rather
  than restarts,
- **non-blocking**, so a failure here never affects a scan.

The honest limitation: a heavily-used address is never fully backfilled at a
low page budget, and the `txs_seen` count says how far we actually got.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .chain import ChainError, MempoolClient
from .cluster import OwnershipGraph
from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class BackfillReport:
    addresses_considered: int = 0
    addresses_processed: int = 0
    addresses_completed: int = 0
    pages_fetched: int = 0
    transactions_seen: int = 0
    errors: list[str] = field(default_factory=list)


class Backfiller:
    """Walks address histories into the ownership graph, within a budget."""

    def __init__(
        self,
        settings: Settings,
        client: Optional[MempoolClient] = None,
        graph: Optional[OwnershipGraph] = None,
    ):
        self.settings = settings
        self.client = client or MempoolClient(settings)
        self.graph = graph

    def run(self, graph: OwnershipGraph, dry_run: bool = False) -> BackfillReport:
        """Backfill up to the configured budget. Never raises on chain errors."""
        report = BackfillReport()
        addresses = graph.backfill_targets(self.settings.backfill_max_addresses_per_run)
        report.addresses_considered = len(addresses)

        pages_left = self.settings.backfill_max_pages_per_run

        for address in addresses:
            if report.addresses_processed >= self.settings.backfill_max_addresses_per_run:
                break
            if pages_left <= 0:
                logger.info("backfill page budget exhausted for this run")
                break

            report.addresses_processed += 1
            pages_used, txs_seen, completed, errors = self._backfill_address(
                graph, address, pages_left, dry_run=dry_run
            )
            pages_left -= pages_used
            report.pages_fetched += pages_used
            report.transactions_seen += txs_seen
            report.errors.extend(errors)
            if completed:
                report.addresses_completed += 1

        if report.addresses_processed:
            logger.info(
                "backfill: %d address(es), %d page(s), %d tx(s), %d complete",
                report.addresses_processed,
                report.pages_fetched,
                report.transactions_seen,
                report.addresses_completed,
            )
        return report

    def _backfill_address(
        self,
        graph: OwnershipGraph,
        address: str,
        page_budget: int,
        dry_run: bool,
    ) -> tuple[int, int, bool, list[str]]:
        """Walk one address's history.

        Returns (pages_used, transactions_seen, completed, errors).
        """
        # Resume from where a previous budgeted run stopped, rather than
        # restarting from the newest transaction every time.
        _completed_before, cursor = graph.backfill_progress(address)

        pages = 0
        txs_seen = 0
        errors: list[str] = []
        completed = False

        while pages < page_budget:
            try:
                page, cursor = self.client.address_txs(
                    address, after_txid=cursor, limit=50
                )
            except ChainError as exc:
                # A single unreachable address must not abort the sweep.
                errors.append(f"{address}: {exc}")
                break

            pages += 1
            if not page:
                completed = True
                cursor = None
                break
            txs_seen += len(page)

            for tx in page:
                if dry_run:
                    continue
                try:
                    graph.observe(tx)
                except Exception as exc:  # observation must never abort
                    errors.append(f"{tx.txid}: {exc}")

            if cursor is None:
                completed = True
                break

        if not dry_run:
            graph.mark_backfilled(address, txs_seen, completed=completed, cursor=cursor)
        return pages, txs_seen, completed, errors
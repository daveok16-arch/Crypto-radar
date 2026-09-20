"""Common-input-ownership address clustering.

The economic inference
----------------------
Every input of a Bitcoin transaction must be signed. If two addresses appear
as inputs to the same transaction, one party controlled both spend keys, so
they share an owner. That is common-input-ownership (CIO), and it is the only
ownership inference this module makes.

What this module deliberately does NOT do
-----------------------------------------
- **It does not identify entities.** Saying "this cluster is Binance" needs
  labelled data we do not have. A cluster is an anonymous set of co-owned
  addresses, nothing more.
- **It does not fuse clusters through CoinJoins.** A CoinJoin transaction
  co-spends inputs from *different* owners by design. Naive CIO merges them
  all into one giant false cluster, which then propagates transitively and
  corrupts everything downstream. These transactions are detected and
  skipped.
- **It does not feed the neural model.** Cluster size depends on how much of
  the chain we have scanned, so it would shift under the autoencoder as
  coverage grows. It is used symbolically, where it stays interpretable.

Error asymmetry, which drives every conservative choice here
-----------------------------------------------------------
A false *merge* is permanent and contagious: it grows a cluster transitively
and silently attributes one owner's behaviour to another. A missed merge only
loses information. So every ambiguous case resolves to "do not merge".
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from .models import Tx

# --- CoinJoin exclusion constants ---------------------------------------
# These are heuristic and conservative. Both an equal-sized-output pattern
# AND a matching input count are required, because either alone produces
# many false positives on ordinary payments.
DUST_SATS = 10_000
# Kept deliberately low. Missing a CoinJoin produces a false merge, which is
# permanent and transitive; wrongly excluding an ordinary payment only loses
# one merge. So the threshold errs toward excluding.
MIN_OUTPUTS_FOR_COINJOIN = 4
MIN_EQUAL_OUTPUTS = 3
# Repeated outputs must account for at least this share of all outputs.
COINJOIN_DOMINANCE = 0.5

SCHEMA = """
CREATE TABLE IF NOT EXISTS cluster_members (
    address TEXT PRIMARY KEY,
    parent TEXT NOT NULL,
    size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS observed_txs (
    txid TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS backfilled_addresses (
    address TEXT PRIMARY KEY,
    backfilled_at REAL NOT NULL,
    txs_seen INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    last_cursor TEXT
);
"""


@dataclass
class ClusterDecision:
    """What the graph did with one transaction, and why."""

    txid: str
    merged: bool
    reason: str
    addresses: int
    already_seen: bool = False


@dataclass
class OwnershipSummary:
    """Ownership facts about a wake-up, with provenance.

    `known` is the honest part: when we have not yet seen the transaction's
    inputs, there is no cluster to speak of and callers must fall back to
    weaker evidence rather than assume a cluster of one.
    """

    known: bool
    cluster_size: int
    cluster_id: Optional[str]
    owned_output_share: float
    is_self_transfer: bool

    def as_dict(self) -> dict:
        return {
            "known": self.known,
            "cluster_size": self.cluster_size,
            "cluster_id": self.cluster_id,
            "owned_output_share": round(self.owned_output_share, 4),
            "is_self_transfer": self.is_self_transfer,
        }


def is_likely_coinjoin(tx: Tx) -> tuple[bool, str]:
    """Conservative CoinJoin detection.

    Returns (is_coinjoin, reason). Requires both a dominant set of equally
    sized outputs and an input count in a plausible range for a co-spend,
    because each signal alone fires on ordinary payments.
    """
    outputs = [v for v in tx.output_values_sats if v > DUST_SATS]
    if len(outputs) < MIN_OUTPUTS_FOR_COINJOIN:
        return False, "few outputs"

    counts = Counter(outputs)
    repeated = sum(n for v, n in counts.items() if n >= MIN_EQUAL_OUTPUTS)
    if repeated < MIN_EQUAL_OUTPUTS:
        return False, "no repeated equal outputs"

    dominance = repeated / len(outputs)
    if dominance < COINJOIN_DOMINANCE:
        return False, f"equal outputs not dominant ({dominance:.2f})"

    # A co-spend needs several participants; one input cannot be a CoinJoin.
    if len(tx.inputs) < MIN_EQUAL_OUTPUTS:
        return False, "too few inputs to be a co-spend"

    # Structural discriminator against consolidation. In a CoinJoin each
    # participant contributes roughly one input and receives roughly two
    # outputs (their equal output plus change), so outputs are at least as
    # numerous as inputs. A consolidation funnels many inputs into a couple of
    # outputs, so it has more inputs than outputs.
    if len(tx.inputs) > len(outputs):
        return False, "more inputs than outputs (consolidation-like)"

    return True, f"dominant equal outputs {repeated}/{len(outputs)} with {len(tx.inputs)} inputs"


class OwnershipGraph:
    """Persistent union-find over addresses.

    The canonical representative of a cluster is always the lexicographically
    smallest address in it. That makes the identifier a pure function of the
    cluster's contents, so it is identical no matter what order transactions
    happened to be processed in — including across a restart or a rescan.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent_dir = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent_dir, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- union-find ------------------------------------------------------
    #
    # Invariant: every non-root address points *directly* at its cluster root.
    # A merge repoints the absorbed root and all of its direct children, so the
    # tree never grows deeper than one level. That keeps lookups O(1) and makes
    # the `size` column trustworthy.

    def _root_of(self, address: str) -> Optional[str]:
        """Resolve a known address to its root. Returns None if unknown."""
        row = self._conn.execute(
            "SELECT parent FROM cluster_members WHERE address = ?", (address,)
        ).fetchone()
        return row[0] if row else None

    def _ensure(self, address: str) -> str:
        """Create a singleton cluster for an address if it is new."""
        self._conn.execute(
            "INSERT OR IGNORE INTO cluster_members (address, parent, size) "
            "VALUES (?, ?, 1)",
            (address, address),
        )
        return self._root_of(address) or address

    def find(self, address: str) -> Optional[str]:
        """Public lookup. Never creates state for unknown addresses."""
        return self._root_of(address)

    def _union(self, left: str, right: str) -> None:
        """Merge two clusters, keeping the lexicographically smallest root."""
        root_left = self._ensure(left)
        root_right = self._ensure(right)
        if root_left == root_right:
            return
        keep, absorbed = (
            (root_left, root_right) if root_left < root_right else (root_right, root_left)
        )

        keep_size = self._size_of(keep)
        absorbed_size = self._size_of(absorbed)

        # Repoint the absorbed root and every address that pointed at it.
        self._conn.execute(
            "UPDATE cluster_members SET parent = ?, size = 1 WHERE parent = ?",
            (keep, absorbed),
        )
        self._conn.execute(
            "UPDATE cluster_members SET size = ? WHERE address = ?",
            (keep_size + absorbed_size, keep),
        )

    def _size_of(self, root: str) -> int:
        row = self._conn.execute(
            "SELECT size FROM cluster_members WHERE address = ?", (root,)
        ).fetchone()
        return int(row[0]) if row else 0

    # -- observation -----------------------------------------------------

    def has_seen(self, txid: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM observed_txs WHERE txid = ?", (txid,)
        ).fetchone()
        return row is not None

    def observe(self, tx: Tx) -> ClusterDecision:
        """Feed one transaction into the ownership graph.

        Idempotent: the same txid is never applied twice, so a rescan cannot
        inflate a cluster or double-count a merge.
        """
        if self.has_seen(tx.txid):
            return ClusterDecision(tx.txid, False, "already observed", 0, already_seen=True)

        coinjoin, reason = is_likely_coinjoin(tx)
        addresses = [
            i.prevout_address for i in tx.inputs if i.prevout_address
        ]
        distinct = sorted(set(addresses))

        # Register every input address as a known singleton even when we merge
        # nothing. Seeing an address is real information: it means `summary_for`
        # can honestly answer "known, unmerged" instead of "unknown", and a
        # later ordinary co-spend can still join it.
        for address in distinct:
            self._ensure(address)

        if coinjoin:
            # Record that we saw it, but merge nothing. Co-spent inputs here
            # belong to different owners by construction.
            self._conn.execute(
                "INSERT OR IGNORE INTO observed_txs (txid) VALUES (?)", (tx.txid,)
            )
            self._conn.commit()
            return ClusterDecision(tx.txid, False, f"coinjoin excluded: {reason}", len(distinct))

        if len(distinct) < 2:
            self._conn.execute(
                "INSERT OR IGNORE INTO observed_txs (txid) VALUES (?)", (tx.txid,)
            )
            self._conn.commit()
            return ClusterDecision(tx.txid, False, "single input address", len(distinct))

        anchor = distinct[0]
        for other in distinct[1:]:
            self._union(anchor, other)

        self._conn.execute(
            "INSERT OR IGNORE INTO observed_txs (txid) VALUES (?)", (tx.txid,)
        )
        self._conn.commit()
        return ClusterDecision(
            tx.txid, True, f"co-spend of {len(distinct)} addresses", len(distinct)
        )

    def observe_all(self, txs: Iterable[Tx]) -> list[ClusterDecision]:
        return [self.observe(tx) for tx in txs]

    # -- queries ---------------------------------------------------------

    def cluster_size(self, address: str) -> int:
        root = self._root_of(address)
        return self._size_of(root) if root else 0

    def cluster_id(self, address: str) -> Optional[str]:
        return self._root_of(address)

    def summary_for(
        self,
        source_address: Optional[str],
        tx: Optional[Tx],
        value_sats: int,
    ) -> OwnershipSummary:
        """Describe ownership for a wake-up, honestly reporting provenance.

        Returns `known=False` when there is nothing to say, so callers can
        fall back to weaker evidence instead of treating an unseen address as
        an isolated one.
        """
        if not source_address or tx is None:
            return OwnershipSummary(False, 0, None, 0.0, False)

        # We only know an address's owner if we have actually seen it before.
        root = self._root_of(source_address)
        if root is None:
            return OwnershipSummary(False, 0, None, 0.0, False)

        size = self._size_of(root)

        owned_value = 0.0
        total_value = 0.0
        for address, value in zip(tx.output_addresses, tx.output_values_sats):
            total_value += value
            if address and self._root_of(address) == root:
                owned_value += value

        share = (owned_value / total_value) if total_value > 0 else 0.0
        return OwnershipSummary(
            known=True,
            cluster_size=size,
            cluster_id=root,
            owned_output_share=share,
            is_self_transfer=owned_value > 0,
        )

    def stats(self) -> dict:
        clusters = self._conn.execute(
            "SELECT COUNT(DISTINCT parent) FROM cluster_members"
        ).fetchone()[0]
        addresses = self._conn.execute(
            "SELECT COUNT(*) FROM cluster_members"
        ).fetchone()[0]
        observed = self._conn.execute(
            "SELECT COUNT(*) FROM observed_txs"
        ).fetchone()[0]
        largest = self._conn.execute(
            "SELECT COALESCE(MAX(size), 0) FROM cluster_members"
        ).fetchone()[0]
        backfilled = self._conn.execute(
            "SELECT COUNT(*) FROM backfilled_addresses"
        ).fetchone()[0]
        return {
            "addresses": int(addresses),
            "clusters": int(clusters),
            "observed_transactions": int(observed),
            "largest_cluster": int(largest),
            "backfilled_addresses": int(backfilled),
        }

    # -- backfill bookkeeping --------------------------------------------

    def needs_backfill(self, address: str) -> bool:
        """True when we have seen this address but never walked its history.

        Deliberately not "is the address known": an address only becomes worth
        a history walk once it has appeared as a spend input, which is the
        moment its co-spend partners start to matter.
        """
        row = self._conn.execute(
            "SELECT 1 FROM backfilled_addresses WHERE address = ?", (address,)
        ).fetchone()
        return row is None

    def mark_backfilled(
        self,
        address: str,
        txs_seen: int = 0,
        completed: bool = False,
        cursor: str | None = None,
    ) -> None:
        """Record progress on one address's history walk.

        The cursor is stored so a walk resumes where it stopped rather than
        restarting. Without it, an address with a long history would hit the
        page budget, restart from the newest transaction on the next turn, and
        never reach the end — a bounded sweep that never finishes.
        """
        import time

        existing = self._conn.execute(
            "SELECT txs_seen, completed, last_cursor FROM backfilled_addresses "
            "WHERE address = ?",
            (address,),
        ).fetchone()
        if existing:
            txs_seen += int(existing["txs_seen"])
            completed = completed or bool(existing["completed"])
            if completed:
                cursor = None

        self._conn.execute(
            "INSERT OR REPLACE INTO backfilled_addresses "
            "(address, backfilled_at, txs_seen, completed, last_cursor) "
            "VALUES (?, ?, ?, ?, ?)",
            (address, time.time(), txs_seen, int(completed), cursor),
        )
        self._conn.commit()

    def backfill_progress(self, address: str) -> tuple[bool, str | None]:
        """Return (completed, cursor) for an address, for resumption."""
        row = self._conn.execute(
            "SELECT completed, last_cursor FROM backfilled_addresses WHERE address = ?",
            (address,),
        ).fetchone()
        if not row:
            return False, None
        return bool(row["completed"]), row["last_cursor"]

    def backfilled_count(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM backfilled_addresses"
            ).fetchone()[0]
        )

    def backfill_targets(self, limit: int) -> list[str]:
        """Addresses to walk next, least-recently-backfilled first.

        Only incomplete addresses are returned. `completed` is distinct from
        "has been touched": an address whose history is too long to finish in
        one budgeted run must stay eligible so the walk can continue, whereas a
        fully walked address should never be re-fetched.

        Oldest-first ordering means no address can be starved by a stream of new
        arrivals, and because the timestamps and cursors are durable, an
        interrupted sweep resumes rather than restarting.
        """
        rows = self._conn.execute(
            """
            SELECT m.address AS address
            FROM cluster_members AS m
            JOIN backfilled_addresses AS b ON b.address = m.address
            WHERE m.size = 1 AND b.completed = 0
            ORDER BY b.backfilled_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        pending = [row["address"] for row in rows]

        if len(pending) < limit:
            # Then addresses never walked at all.
            fresh = self._conn.execute(
                """
                SELECT m.address AS address
                FROM cluster_members AS m
                LEFT JOIN backfilled_addresses AS b ON b.address = m.address
                WHERE m.size = 1 AND b.address IS NULL
                ORDER BY m.address ASC
                LIMIT ?
                """,
                (limit - len(pending),),
            ).fetchall()
            pending.extend(row["address"] for row in fresh)
        return pending
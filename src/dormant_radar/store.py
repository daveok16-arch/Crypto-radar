"""SQLite persistence for detected wake-ups and scan progress.

The store is intentionally small: one table for wake-ups and one key/value
row for how far the scanner has processed. SQLite keeps the whole thing
runnable with no external service.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Iterable, Optional

from .models import WakeUp

SCHEMA = """
CREATE TABLE IF NOT EXISTS wakeups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txid TEXT NOT NULL,
    spent_outpoint TEXT NOT NULL,
    spend_block_height INTEGER NOT NULL,
    value_sats INTEGER NOT NULL,
    address TEXT,
    dormant_blocks INTEGER NOT NULL,
    dormant_years REAL NOT NULL,
    script_type TEXT,
    observed_at REAL NOT NULL,
    cause_hypothesis TEXT,
    cause_confidence REAL,
    payload TEXT NOT NULL,
    UNIQUE (txid, spent_outpoint)
);
CREATE INDEX IF NOT EXISTS idx_wakeups_height ON wakeups (spend_block_height);
CREATE INDEX IF NOT EXISTS idx_wakeups_value ON wakeups (value_sats);

CREATE TABLE IF NOT EXISTS scan_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerted_outpoints (
    outpoint TEXT PRIMARY KEY,
    alerted_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- wake-ups --------------------------------------------------------

    def add_wakeups(self, wakeups: Iterable[WakeUp]) -> int:
        """Insert wake-ups, ignoring ones already recorded. Returns the count added."""
        added = 0
        for wakeup in wakeups:
            payload = json.dumps(wakeup.as_dict())
            hypothesis = wakeup.cause.hypothesis if wakeup.cause else None
            confidence = wakeup.cause.confidence if wakeup.cause else None
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO wakeups (
                    txid, spent_outpoint, spend_block_height, value_sats,
                    address, dormant_blocks, dormant_years, script_type,
                    observed_at, cause_hypothesis, cause_confidence, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wakeup.txid,
                    str(wakeup.spent),
                    wakeup.spend_block_height,
                    wakeup.value_sats,
                    wakeup.address,
                    wakeup.dormant_blocks,
                    wakeup.dormant_years,
                    wakeup.script_type,
                    wakeup.observed_at,
                    hypothesis,
                    confidence,
                    payload,
                ),
            )
            added += cursor.rowcount
        self._conn.commit()
        return added

    def list_wakeups(
        self,
        limit: int = 50,
        min_value_sats: int = 0,
        since_height: Optional[int] = None,
        hypothesis: Optional[str] = None,
    ) -> list[dict]:
        clauses = ["value_sats >= ?"]
        params: list = [min_value_sats]
        if since_height is not None:
            clauses.append("spend_block_height >= ?")
            params.append(since_height)
        if hypothesis is not None:
            clauses.append("cause_hypothesis = ?")
            params.append(hypothesis)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT payload FROM wakeups
            WHERE {' AND '.join(clauses)}
            ORDER BY spend_block_height DESC, value_sats DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def stats(self) -> dict:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(value_sats), 0) AS total_value,
                   COALESCE(MAX(spend_block_height), 0) AS latest_height
            FROM wakeups
            """
        ).fetchone()
        by_cause = {
            r["cause_hypothesis"] or "unscored": r["n"]
            for r in self._conn.execute(
                """
                SELECT cause_hypothesis, COUNT(*) AS n FROM wakeups
                GROUP BY cause_hypothesis
                """
            ).fetchall()
        }
        return {
            "total_wakeups": row["total"],
            "total_value_sats": row["total_value"],
            "total_value_btc": round(row["total_value"] / 100_000_000, 8),
            "latest_height": row["latest_height"],
            "by_hypothesis": by_cause,
            "last_scanned_height": self.get_state("last_scanned_height"),
        }

    # -- scan state ------------------------------------------------------

    def set_state(self, key: str, value: object) -> None:
        self._conn.execute(
            "INSERT INTO scan_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def get_state(self, key: str) -> object:
        row = self._conn.execute(
            "SELECT value FROM scan_state WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else None

    # -- alert deduplication ---------------------------------------------

    def was_alerted(self, outpoint: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM alerted_outpoints WHERE outpoint = ?", (outpoint,)
        ).fetchone()
        return row is not None

    def mark_alerted(self, outpoint: str) -> None:
        import time

        self._conn.execute(
            "INSERT OR REPLACE INTO alerted_outpoints (outpoint, alerted_at) "
            "VALUES (?, ?)",
            (outpoint, time.time()),
        )
        self._conn.commit()

    def alerted_count(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM alerted_outpoints"
            ).fetchone()[0]
        )
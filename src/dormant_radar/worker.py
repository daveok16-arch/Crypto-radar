"""Background polling worker.

Each cycle re-reads the tip and scans any unprocessed blocks. A persistent
cursor in the store means a restart resumes rather than rescans.

After each scan, new wake-ups are evaluated against the alert policy and
delivered through the configured channels. Alerts are deduplicated in the
database, so a restart cannot re-send an alert the operator has already seen.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from .backfill import Backfiller
from .chain import ChainError
from .cluster import OwnershipGraph
from .config import Settings
from .notifier import Notifier
from .notify import build_channels
from .redact import install_redaction, register_many
from .scanner import scan_once
from .store import Store

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.store = Store(self.settings.db_path)
        self._stop = threading.Event()

        # Register channel secrets before any logging happens, so the
        # redaction filter can scrub them from the very first message.
        register_many(
            [
                self.settings.telegram_bot_token,
                self.settings.smtp_password,
                *self.settings.telegram_chat_ids,
            ]
        )

        channels = build_channels(self.settings)
        if channels:
            logger.info(
                "alert channels: %s",
                ", ".join(channel.name for channel in channels),
            )
        elif self.settings.alerts_enabled:
            logger.warning(
                "alerts are enabled but no channel is fully configured; "
                "nothing will be delivered"
            )

        self.notifier = self.settings.build_notifier(self.store, channels)

        if self.settings.alerts_enabled:
            logger.info(
                "alerting mode: %s",
                "digest every %.0fs"
                % self.settings.alert_digest_interval_seconds
                if self.settings.alert_digest_mode
                else "immediate",
            )

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        logger.info(
            "worker started; poll every %.0fs, dormant threshold %d blocks",
            self.settings.poll_interval_seconds,
            self.settings.dormant_after_blocks,
        )
        while not self._stop.is_set():
            graph = None
            try:
                scan_result = scan_once(self.settings, store=self.store)
                graph = scan_result.graph
            except ChainError as exc:
                logger.error("scan failed: %s", exc)
            except Exception:  # keep the worker alive across unexpected faults
                logger.exception("unexpected error during scan")

            # Backfill after scanning, using the scan's own graph connection.
            # Budgeted, so it cannot stall the next poll.
            if self.settings.backfill_enabled and graph is not None:
                try:
                    Backfiller(self.settings).run(graph)
                except Exception:
                    logger.exception("unexpected error during backfill")

            try:
                self.notifier.run()
            except Exception:  # alerting must never kill the hunter
                logger.exception("unexpected error while alerting")

            self._stop.wait(self.settings.poll_interval_seconds)
        logger.info("worker stopped")
        self.store.close()


def run_worker(settings: Settings | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    install_redaction()
    worker = Worker(settings)

    def _handle(signum, frame):  # pragma: no cover - signal path
        logger.info("signal %s received, shutting down", signum)
        worker.stop()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    worker.run_forever()
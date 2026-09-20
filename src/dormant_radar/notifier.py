"""The alerting loop: evaluate new wake-ups, deduplicate, deliver.

Two delivery modes:

**Immediate** (default): each matching wake-up is sent as its own message.

**Digest**: matching wake-ups accumulate and one summary message is sent per
window (hourly by default). This is what keeps a channel survivable — a busy
hour becomes one message rather than thirty.

Safeguards, each fixing a specific failure:

**Dedup persistence.** Alerted outpoints are recorded in the database, not in
memory. A worker restart must not re-send yesterday's whale movement.

**Queue inferred from durable state.** The digest queue is not a separate
buffer. It is derived from wake-ups that match the policy and are not yet in
`alerted_outpoints`. A crash therefore loses nothing: the next run recomputes
the same set. A separate in-memory queue would have dropped alerts on restart.

**Marked delivered only after a channel accepts.** In digest mode this happens
at send time, not queue time. Marking at queue time would mean a crash between
queueing and sending lost those alerts permanently.

**Rate limiting.** In immediate mode, excess is held back and retried, never
dropped. In digest mode the window itself bounds volume, so the per-item cap
does not apply.

**Isolation.** A failing channel must not break a scan, and one bad alert must
not stop the rest.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .alerting import Alert, AlertPolicy, evaluate
from .notify import Channel, format_digest
from .store import Store

logger = logging.getLogger(__name__)


@dataclass
class AlertRun:
    considered: int = 0
    matched: int = 0
    delivered: int = 0
    suppressed_duplicates: int = 0
    suppressed_ratelimit: int = 0
    channel_failures: int = 0
    alerts: list[Alert] = field(default_factory=list)
    # Digest-specific
    digest_sent: bool = False
    digest_pending: int = 0
    digest_seconds_until_next: float = 0.0


class Notifier:
    """Turns stored wake-ups into delivered alerts, exactly once each.

    In digest mode, `run()` queues (evaluates and counts) and sends a summary
    only when the window has elapsed.
    """

    def __init__(
        self,
        store: Store,
        policy: AlertPolicy,
        channels: list[Channel],
        enabled: bool = True,
        digest_mode: bool = False,
        digest_interval_seconds: float = 3600.0,
        digest_max_items: int = 25,
    ):
        self.store = store
        self.policy = policy
        self.channels = channels
        self.enabled = enabled
        self.digest_mode = digest_mode
        self.digest_interval_seconds = digest_interval_seconds
        self.digest_max_items = digest_max_items

    def _recent_deliveries(self) -> list[float]:
        raw = self.store.get_state("alert_delivery_times") or []
        if not isinstance(raw, list):
            return []
        now = time.time()
        # Only the last hour matters for the rate limit.
        return [float(t) for t in raw if now - float(t) < 3600]

    def _record_delivery(self, timestamps: list[float]) -> None:
        self.store.set_state("alert_delivery_times", timestamps[-200:])

    def _pending(self, limit: int = 200) -> list[Alert]:
        """Evaluate recent wake-ups and return those not yet alerted.

        This is the digest queue, recomputed from durable state on every run so
        a restart cannot lose queued alerts.
        """
        pending: list[Alert] = []
        for wakeup in self.store.list_wakeups(limit=limit):
            alert = evaluate(wakeup, self.policy)
            if alert is None:
                continue
            if self.store.was_alerted(alert.dedupe_key):
                continue
            pending.append(alert)
        return pending

    def run(self, limit: int = 200) -> AlertRun:
        """Evaluate the most recent wake-ups and deliver as the mode dictates."""
        result = AlertRun()
        if not self.enabled or not self.channels:
            return result

        candidates = self.store.list_wakeups(limit=limit)
        result.considered = len(candidates)

        if self.digest_mode:
            return self._run_digest(result)

        delivered_times = self._recent_deliveries()

        for wakeup in candidates:
            alert = evaluate(wakeup, self.policy)
            if alert is None:
                continue
            result.matched += 1

            key = alert.dedupe_key
            if self.store.was_alerted(key):
                result.suppressed_duplicates += 1
                continue

            if len(delivered_times) >= self.policy.max_alerts_per_hour:
                # Held back, not lost: it stays un-alerted and will be retried
                # on the next run once the window frees up.
                result.suppressed_ratelimit += 1
                logger.info("alert rate limit reached; holding %s", key)
                break

            if self._deliver(alert, result):
                self.store.mark_alerted(key)
                delivered_times.append(time.time())
                result.delivered += 1
                result.alerts.append(alert)

        self._record_delivery(delivered_times)
        if result.delivered:
            logger.info(
                "alerts: %d delivered, %d duplicates skipped, %d rate-limited",
                result.delivered,
                result.suppressed_duplicates,
                result.suppressed_ratelimit,
            )
        return result

    def _run_digest(self, result: AlertRun) -> AlertRun:
        """Accumulate matches and send one summary when the window elapses."""
        pending = self._pending()
        result.matched = len(pending)
        result.digest_pending = len(pending)

        last = self.store.get_state("last_digest_at")
        now = time.time()
        if isinstance(last, (int, float)):
            elapsed = now - float(last)
        else:
            # Never digest before: start the clock rather than firing instantly,
            # so a fresh deployment does not send a summary of historical rows.
            self.store.set_state("last_digest_at", now)
            result.digest_seconds_until_next = self.digest_interval_seconds
            logger.info("digest window started; first summary in %.0fs", self.digest_interval_seconds)
            return result

        remaining = self.digest_interval_seconds - elapsed
        if remaining > 0:
            result.digest_seconds_until_next = remaining
            return result

        if not pending:
            # Sending "0 alerts" every hour is precisely the noise that gets a
            # channel muted. Silence is the normal state.
            self.store.set_state("last_digest_at", now)
            result.digest_seconds_until_next = self.digest_interval_seconds
            logger.info("digest window elapsed with nothing to report; staying silent")
            return result

        body = format_digest(
            pending,
            window_seconds=self.digest_interval_seconds,
            max_items=self.digest_max_items,
        )
        subject = f"Dormant Radar digest: {len(pending)} alert(s)"

        sent_any = False
        for channel in self.channels:
            try:
                if channel.send_text(body, subject=subject):
                    sent_any = True
            except Exception as exc:  # a channel must never break the run
                result.channel_failures += 1
                logger.warning("channel %s raised: %s", channel.name, exc)

        if sent_any:
            # Marked only now, after a channel accepted, so a crash between
            # queueing and sending cannot lose these alerts.
            for alert in pending:
                self.store.mark_alerted(alert.dedupe_key)
            self.store.set_state("last_digest_at", now)
            result.digest_sent = True
            result.delivered = len(pending)
            result.alerts = pending
            result.digest_seconds_until_next = self.digest_interval_seconds
            logger.info("digest sent covering %d alert(s)", len(pending))
        else:
            # Window is NOT advanced: the same items are retried on the next run
            # instead of being silently skipped.
            result.channel_failures += 1
            result.digest_seconds_until_next = 0.0
            logger.warning(
                "no channel accepted the digest; %d alert(s) remain pending",
                len(pending),
            )
        return result

    def _deliver(self, alert: Alert, result: AlertRun) -> bool:
        """Send one alert to every channel, isolating failures."""
        sent_any = False
        for channel in self.channels:
            try:
                if channel.send(alert):
                    sent_any = True
            except Exception as exc:  # a channel must never break the run
                result.channel_failures += 1
                logger.warning("channel %s raised: %s", channel.name, exc)
        if not sent_any:
            result.channel_failures += 1
            logger.warning("no channel accepted alert for %s; will retry", alert.dedupe_key)
        return sent_any
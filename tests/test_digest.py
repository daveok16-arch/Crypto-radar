"""Digest mode tests.

The properties that matter, each guarding a specific failure:

1. Nothing is marked delivered until a channel accepts — a crash between
   queueing and sending must not lose alerts.
2. An empty window sends nothing — a "0 alerts" message every hour is exactly
   the noise that gets a channel muted.
3. The window is not advanced on delivery failure, so pending alerts are
   retried rather than silently skipped.
4. One message per window, not one per alert.
"""

import time

import pytest

from dormant_radar.alerting import AlertPolicy
from dormant_radar.notifier import Notifier
from dormant_radar.notify import format_digest
from dormant_radar.store import Store


def store_wakeup(store, outpoint, value=20_000_000_000, years=12.0, address="bc1qex"):
    import json

    wakeup = {
        "txid": "tt" * 32,
        "spent_outpoint": outpoint,
        "spend_block_height": 900_000,
        "value_sats": value,
        "value_btc": value / 100_000_000,
        "address": address,
        "dormant_blocks": int(years * 52_560),
        "dormant_years": years,
        "script_type": "v0_p2wpkh",
        "observed_at": 1.0,
        "cause": {
            "hypothesis": "holding",
            "confidence": 0.2,
            "distribution": {"holding": 0.6},
            "rationale": ["long dormancy"],
        },
    }
    store._conn.execute(
        """INSERT INTO wakeups (txid, spent_outpoint, spend_block_height, value_sats,
           address, dormant_blocks, dormant_years, script_type, observed_at,
           cause_hypothesis, cause_confidence, payload)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            wakeup["txid"], outpoint, 900_000, value, address, 0, years,
            "v0_p2wpkh", 1.0, "holding", 0.2, json.dumps(wakeup),
        ),
    )
    store._conn.commit()


class FakeChannel:
    """Implements send_text, which is what digest mode uses."""

    def __init__(self, name="fake", succeed=True):
        self.name = name
        self.succeed = succeed
        self.messages: list[tuple[str, str]] = []

    def send_text(self, text, subject=""):
        if not self.succeed:
            return False
        self.messages.append((subject, text))
        return True

    def send(self, alert):
        return self.send_text("single", "single")


POLICY = AlertPolicy(min_value_sats=5_000_000_000, min_dormant_years=8.0)


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "digest.db"))
    yield s
    s.close()


def make_notifier(store, channels, interval=3600.0):
    return Notifier(
        store,
        POLICY,
        channels,
        enabled=True,
        digest_mode=True,
        digest_interval_seconds=interval,
    )


def test_first_run_starts_window_without_sending(store):
    """A fresh deployment must not immediately summarise historical rows."""
    store_wakeup(store, "o1" + "0" * 62 + ":0")
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])
    result = notifier.run()

    assert result.digest_sent is False
    assert channel.messages == []
    assert result.digest_pending == 1
    assert result.digest_seconds_until_next > 0


def test_window_elapsed_sends_one_summary(store):
    for i in range(3):
        store_wakeup(store, f"o{i}" + "0" * 62 + ":0")
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])

    notifier.run()  # starts the window
    store.set_state("last_digest_at", time.time() - 4000)  # force expiry
    result = notifier.run()

    assert result.digest_sent is True
    assert len(channel.messages) == 1, "must be one message, not one per alert"
    assert result.delivered == 3
    assert "3 alert(s)" in channel.messages[0][0]


def test_nothing_marked_before_delivery(store):
    """Crash safety: pending must remain pending until a send succeeds."""
    store_wakeup(store, "o1" + "0" * 62 + ":0")
    failing = FakeChannel(succeed=False)
    notifier = make_notifier(store, [failing])

    notifier.run()  # start window
    store.set_state("last_digest_at", time.time() - 4000)
    result = notifier.run()

    assert result.digest_sent is False
    assert store.alerted_count() == 0, "nothing may be marked before delivery"
    # Still pending for the next attempt.
    assert result.digest_pending == 1


def test_failed_digest_does_not_advance_window(store):
    store_wakeup(store, "o1" + "0" * 62 + ":0")
    failing = FakeChannel(succeed=False)
    notifier = make_notifier(store, [failing])
    notifier.run()
    store.set_state("last_digest_at", time.time() - 4000)
    notifier.run()

    pending = notifier._pending()
    assert len(pending) == 1, "alert must be retried, not skipped"

    # A later successful run does deliver it.
    good = FakeChannel()
    retry = make_notifier(store, [good])
    result = retry.run()
    assert result.digest_sent is True
    assert store.alerted_count() == 1


def test_empty_window_stays_silent(store):
    """No alerts means no message. This is the whole point of the mode."""
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])
    notifier.run()
    store.set_state("last_digest_at", time.time() - 4000)
    result = notifier.run()

    assert result.digest_sent is False
    assert channel.messages == []
    # Window still advances, so it does not fire continuously.
    assert store.get_state("last_digest_at") is not None


def test_digest_not_sent_twice_for_same_alerts(store):
    store_wakeup(store, "o1" + "0" * 62 + ":0")
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])
    notifier.run()
    store.set_state("last_digest_at", time.time() - 4000)
    notifier.run()
    assert len(channel.messages) == 1

    store.set_state("last_digest_at", time.time() - 4000)
    result = notifier.run()
    assert result.digest_sent is False
    assert len(channel.messages) == 1


def test_digest_mode_ignores_per_item_rate_limit(store):
    """The window bounds volume, so the per-item cap must not hold items back."""
    for i in range(5):
        store_wakeup(store, f"o{i}" + "0" * 62 + ":0")
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])
    notifier.policy = AlertPolicy(
        min_value_sats=1, min_dormant_years=None, max_alerts_per_hour=1
    )
    notifier.run()
    store.set_state("last_digest_at", time.time() - 4000)
    result = notifier.run()
    assert result.delivered == 5


def test_disabled_notifier_stays_silent(store):
    store_wakeup(store, "o1" + "0" * 62 + ":0")
    channel = FakeChannel()
    notifier = make_notifier(store, [channel])
    notifier.enabled = False
    notifier.run()
    store.set_state("last_digest_at", time.time() - 4000)
    assert notifier.run().digest_sent is False
    assert channel.messages == []


# --- digest rendering --------------------------------------------------


def make_alert(value_sats, years=10.0, address="bc1qa", known=True,
               cluster=3, self_transfer=False):
    from dormant_radar.alerting import Alert

    wakeup = {
        "txid": "tt" * 32,
        "spent_outpoint": "oo" * 32 + ":0",
        "value_sats": value_sats,
        "value_btc": value_sats / 100_000_000,
        "dormant_years": years,
        "address": address,
        "cause": {"hypothesis": "holding", "rationale": ["because"]},
        "ownership": {
            "known": known,
            "cluster_size": cluster,
            "is_self_transfer": self_transfer,
        },
    }
    return Alert(wakeup=wakeup, reasons=[f"value {wakeup['value_btc']} BTC"], severity="high")


def test_empty_digest_is_refused():
    """Nonexistence of an empty digest is enforced, not merely conventional."""
    with pytest.raises(ValueError):
        format_digest([], window_seconds=3600)


def test_digest_orders_by_value_descending():
    """Largest movements first, so truncation drops the least interesting."""
    body = format_digest(
        [make_alert(1_000_000_000), make_alert(90_000_000_000), make_alert(5_000_000_000)],
        window_seconds=3600,
    )
    first = body.split("1. ", 1)[1].split(" BTC")[0]
    assert first == "900.0"


def test_digest_states_total_and_window():
    body = format_digest([make_alert(5_000_000_000)], window_seconds=7200)
    assert "1 alert(s)" in body
    assert "2.0h" in body
    assert "50.00 BTC" in body


def test_digest_truncates_with_a_truthful_notice():
    """Beyond the cap, the count stays honest and the omission is disclosed."""
    alerts = [make_alert(1_000_000_000) for _ in range(30)]
    body = format_digest(alerts, window_seconds=3600, max_items=5)
    assert "30 alert(s)" in body, "header must report the true total"
    assert "and 25 more not shown" in body


def test_digest_reports_unknown_ownership_honestly():
    body = format_digest([make_alert(1_000_000_000, known=False)], window_seconds=3600)
    assert "ownership unknown" in body


def test_digest_marks_self_transfer():
    body = format_digest(
        [make_alert(1_000_000_000, self_transfer=True)], window_seconds=3600
    )
    assert "self-transfer" in body


def test_digest_omits_rationale_by_default():
    body = format_digest([make_alert(1_000_000_000)], window_seconds=3600)
    assert "note:" not in body

    with_notes = format_digest(
        [make_alert(1_000_000_000)], window_seconds=3600, include_rationale=True
    )
    assert "note: because" in with_notes


def test_digest_is_a_single_message_even_for_many_alerts():
    alerts = [make_alert(1_000_000_000) for _ in range(10)]
    body = format_digest(alerts, window_seconds=3600)
    assert isinstance(body, str)
    assert body.count("Dormant Radar digest") == 1
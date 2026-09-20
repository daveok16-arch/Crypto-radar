"""Notifier tests: dedup across restarts, rate limiting, channel isolation.

Channels here are hand-written fakes implementing the real `send` contract.
A fake is appropriate because delivery is a network side effect; the dedup,
rate-limit, and isolation logic under test is real and uses real SQLite.
"""

import pytest

from dormant_radar.alerting import Alert, AlertPolicy
from dormant_radar.notifier import Notifier
from dormant_radar.notify import format_alert
from dormant_radar.store import Store


def store_wakeup(store, outpoint="oo" * 32 + ":0", value=20_000_000_000, years=12.0):
    wakeup = {
        "txid": "tt" * 32,
        "spent_outpoint": outpoint,
        "spend_block_height": 900_000,
        "value_sats": value,
        "value_btc": value / 100_000_000,
        "address": "bc1qexample",
        "dormant_blocks": int(years * 52_560),
        "dormant_years": years,
        "script_type": "v0_p2wpkh",
        "observed_at": 1.0,
        "cause": {
            "hypothesis": "holding",
            "confidence": 0.2,
            "distribution": {"lost": 0.1, "holding": 0.6, "structural": 0.3},
            "rationale": ["because of dormancy"],
        },
    }
    store._conn.execute(
        """INSERT INTO wakeups (txid, spent_outpoint, spend_block_height, value_sats,
           address, dormant_blocks, dormant_years, script_type, observed_at,
           cause_hypothesis, cause_confidence, payload)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            wakeup["txid"], wakeup["spent_outpoint"], 900_000, value, "bc1qexample",
            0, years, "v0_p2wpkh", 1.0, "holding", 0.2,
            __import__("json").dumps(wakeup),
        ),
    )
    store._conn.commit()


class FakeChannel:
    def __init__(self, name="fake", succeed=True):
        self.name = name
        self.succeed = succeed
        self.sent: list[Alert] = []

    def send(self, alert):
        if not self.succeed:
            return False
        self.sent.append(alert)
        return True


class ExplodingChannel:
    name = "boom"

    def send(self, alert):
        raise RuntimeError("channel blew up")


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "notify.db"))
    yield s
    s.close()


POLICY = AlertPolicy(min_value_sats=5_000_000_000, min_dormant_years=8.0)


def test_delivers_matching_alert(store):
    store_wakeup(store)
    channel = FakeChannel()
    result = Notifier(store, POLICY, [channel]).run()
    assert result.delivered == 1
    assert len(channel.sent) == 1


def test_ignores_non_matching_alert(store):
    store_wakeup(store, value=1_000, years=1.0)
    channel = FakeChannel()
    result = Notifier(store, POLICY, [channel]).run()
    assert result.delivered == 0
    assert channel.sent == []


def test_disabled_notifier_sends_nothing(store):
    store_wakeup(store)
    channel = FakeChannel()
    result = Notifier(store, POLICY, [channel], enabled=False).run()
    assert result.delivered == 0
    assert channel.sent == []


def test_dedup_survives_restart(tmp_path):
    """The critical property: a worker restart must not re-alert old events."""
    db_path = str(tmp_path / "restart.db")

    first_store = Store(db_path)
    store_wakeup(first_store)
    channel = FakeChannel()
    Notifier(first_store, POLICY, [channel]).run()
    assert len(channel.sent) == 1
    first_store.close()

    # New process, same database.
    second_store = Store(db_path)
    try:
        second_channel = FakeChannel()
        result = Notifier(second_store, POLICY, [second_channel]).run()
        assert result.delivered == 0
        assert result.suppressed_duplicates == 1
        assert second_channel.sent == []
    finally:
        second_store.close()


def test_distinct_outpoints_both_alert(store):
    store_wakeup(store, outpoint="o1" + "0" * 62 + ":0")
    store_wakeup(store, outpoint="o2" + "0" * 62 + ":1")
    channel = FakeChannel()
    result = Notifier(store, POLICY, [channel]).run()
    assert result.delivered == 2
    assert len(channel.sent) == 2


def test_rate_limit_holds_excess_and_does_not_mark_it_delivered(store):
    for i in range(5):
        store_wakeup(store, outpoint=f"r{i}" + "0" * 62 + ":0")
    channel = FakeChannel()
    policy = AlertPolicy(min_value_sats=1, min_dormant_years=None, max_alerts_per_hour=2)
    result = Notifier(store, policy, [channel]).run()

    assert result.delivered == 2
    assert result.suppressed_ratelimit >= 1
    # Held-back events must remain eligible for a later run.
    assert store.alerted_count() == 2
    assert len(channel.sent) == 2


def test_failed_delivery_is_retried_not_lost(store):
    """If no channel accepts it, the event must stay eligible."""
    store_wakeup(store)
    failing = FakeChannel(succeed=False)
    result = Notifier(store, POLICY, [failing]).run()
    assert result.delivered == 0
    assert store.alerted_count() == 0

    good = FakeChannel()
    second = Notifier(store, POLICY, [good]).run()
    assert second.delivered == 1


def test_one_bad_channel_does_not_block_the_others(store):
    store_wakeup(store)
    good = FakeChannel()
    result = Notifier(store, POLICY, [ExplodingChannel(), good]).run()
    assert result.delivered == 1
    assert len(good.sent) == 1
    assert result.channel_failures == 1


def test_multiple_channels_all_receive(store):
    store_wakeup(store)
    a, b = FakeChannel("a"), FakeChannel("b")
    result = Notifier(store, POLICY, [a, b]).run()
    assert result.delivered == 1
    assert len(a.sent) == 1 and len(b.sent) == 1


def test_no_channels_means_no_work(store):
    store_wakeup(store)
    result = Notifier(store, POLICY, []).run()
    assert result.delivered == 0
    assert result.considered == 0


# --- message formatting ------------------------------------------------


def test_format_alert_includes_facts_and_reasons(store):
    store_wakeup(store)
    alert = Alert(
        wakeup=store.list_wakeups()[0],
        reasons=["value 200.00 BTC clears the threshold"],
        severity="high",
    )
    text = format_alert(alert)
    assert "Dormant wallet wake-up" in text
    assert "200.0" in text
    assert "Why this alerted" in text
    assert "Reasoning" in text


def test_format_alert_can_omit_reasoning(store):
    """Operators may not want analysis leaving the machine."""
    store_wakeup(store)
    alert = Alert(wakeup=store.list_wakeups()[0], reasons=["x"], severity="high")
    text = format_alert(alert, include_rationale=False)
    assert "because of dormancy" not in text
    assert "Why this alerted" in text


def test_format_alert_reports_unknown_ownership_honestly(store):
    store_wakeup(store)
    alert = Alert(wakeup=store.list_wakeups()[0], reasons=["x"])
    assert "Ownership     : unknown" in format_alert(alert)
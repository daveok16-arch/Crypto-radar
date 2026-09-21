"""Portable state tests.

The bug these guard against, demonstrated before the fix: on a stateless cloud
runner each run gets an empty filesystem, so SQLite-backed dedup forgets, and
the same wake-up is alerted on every single run. Three runs produced three
duplicate deliveries. Users mute a channel that does that, permanently.

So the central test is `test_dedup_survives_a_fresh_pod`: run the notifier
repeatedly with nothing shared between invocations except the state backend.
"""

import json

import pytest

from dormant_radar.alerting import AlertPolicy
from dormant_radar.notifier import Notifier
from dormant_radar.state import LocalState, RadarState, open_state
from dormant_radar.store import Store


def make_wakeup(outpoint, value=600_000_000_000, years=12.0):
    return {
        "txid": "tt" * 32,
        "spent_outpoint": outpoint,
        "spend_block_height": 900_000,
        "value_sats": value,
        "value_btc": value / 100_000_000,
        "address": "bc1qx",
        "dormant_blocks": int(years * 52_560),
        "dormant_years": years,
        "script_type": "v0_p2wpkh",
        "observed_at": 1.0,
        "cause": {"hypothesis": "holding", "confidence": 0.3,
                  "distribution": {}, "rationale": []},
    }


def seed(store, outpoint, **kwargs):
    w = make_wakeup(outpoint, **kwargs)
    store._conn.execute(
        """INSERT INTO wakeups (txid, spent_outpoint, spend_block_height, value_sats,
           address, dormant_blocks, dormant_years, script_type, observed_at,
           cause_hypothesis, cause_confidence, payload)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (w["txid"], outpoint, 900_000, w["value_sats"], "bc1qx", 0, w["dormant_years"],
         "v0_p2wpkh", 1.0, "holding", 0.3, json.dumps(w)),
    )
    store._conn.commit()


class FakeChannel:
    name = "fake"

    def __init__(self, succeed=True):
        self.succeed = succeed
        self.sent = []

    def send(self, alert):
        if self.succeed:
            self.sent.append(alert)
        return self.succeed

    def send_text(self, text, subject=""):
        return self.succeed


POLICY = AlertPolicy(min_value_sats=100_000_000, min_dormant_years=5.0)


# --- the behaviour that matters ----------------------------------------


def test_dedup_survives_a_fresh_pod(tmp_path):
    """The regression test for duplicate alerts on a stateless runner.

    Nothing is shared between runs but the state backend — the database is a
    brand new file each time, exactly as a fresh pod gives.
    """
    state_path = str(tmp_path / "state.json")
    delivered = []

    for run_number in range(1, 4):
        store = Store(str(tmp_path / f"run{run_number}.db"))  # empty every run
        seed(store, "oo" * 32 + ":0")
        channel = FakeChannel()

        notifier = Notifier(store, POLICY, [channel])
        notifier.install_state(LocalState(state_path))
        result = notifier.run()
        notifier.snapshot_state()

        delivered.append(result.delivered)
        store.close()

    assert delivered == [1, 0, 0], f"first run should alert once, later runs not: {delivered}"


def test_without_state_the_bug_reproduces(tmp_path):
    """Documents the failure being fixed, so the guard is not mistaken for noise.

    With no shared state, every run re-alerts. This is the observable behaviour
    that makes a channel unusable, and it is why the state backend exists.
    """
    delivered = []
    for run_number in range(1, 4):
        store = Store(str(tmp_path / f"nostate{run_number}.db"))
        seed(store, "oo" * 32 + ":0")
        channel = FakeChannel()
        result = Notifier(store, POLICY, [channel]).run()
        delivered.append(result.delivered)
        store.close()

    assert delivered == [1, 1, 1], "without shared state, duplicates are expected"


def test_new_events_still_alert_across_runs(tmp_path):
    """Dedup must not become 'alert once ever'."""
    state_path = str(tmp_path / "state.json")

    store = Store(str(tmp_path / "a.db"))
    seed(store, "a" * 64 + ":0")
    n = Notifier(store, POLICY, [FakeChannel()])
    n.install_state(LocalState(state_path))
    assert n.run().delivered == 1
    n.snapshot_state()
    store.close()

    store = Store(str(tmp_path / "b.db"))
    seed(store, "b" * 64 + ":1")   # a different outpoint
    channel = FakeChannel()
    n = Notifier(store, POLICY, [channel])
    n.install_state(LocalState(state_path))
    result = n.run()
    assert result.delivered == 1, "a genuinely new event must still alert"
    assert channel.sent[0].dedupe_key == "b" * 64 + ":1"
    store.close()


def test_state_records_the_scan_cursor(tmp_path):
    state_path = str(tmp_path / "state.json")
    store = Store(str(tmp_path / "c.db"))
    n = Notifier(store, POLICY, [FakeChannel()])
    n.install_state(LocalState(state_path))
    n.snapshot_state(scanned_to=967_900)
    store.close()

    assert LocalState(state_path).load().last_scanned_height == 967_900


def test_failed_delivery_is_not_recorded_as_alerted(tmp_path):
    """A suppressed send must not be deduped away — it has to retry."""
    state_path = str(tmp_path / "state.json")
    store = Store(str(tmp_path / "d.db"))
    seed(store, "oo" * 32 + ":0")

    failing = FakeChannel(succeed=False)
    n = Notifier(store, POLICY, [failing])
    n.install_state(LocalState(state_path))
    assert n.run().delivered == 0
    n.snapshot_state()
    assert LocalState(state_path).load().alerted_outpoints == []
    store.close()


def test_alerted_list_is_bounded(tmp_path):
    """An unbounded list would grow forever on a long deployment."""
    state_path = str(tmp_path / "state.json")
    store = Store(str(tmp_path / "e.db"))
    n = Notifier(store, POLICY, [FakeChannel()])
    n.install_state(LocalState(state_path))
    for i in range(5100):
        n._mark_alerted(f"outpoint{i}", commit=False)
    n.store.commit()
    assert len(n.alerted_outpoints) <= 5000
    store.close()


# --- state backend behaviour -------------------------------------------


def test_radar_state_roundtrip():
    state = RadarState(last_scanned_height=100, alerted_outpoints=["a:0", "b:1"])
    restored = RadarState.from_dict(state.as_dict())
    assert restored.last_scanned_height == 100
    assert restored.alerted_outpoints == ["a:0", "b:1"]


def test_unknown_version_is_treated_as_empty():
    """A schema we do not understand must not be silently trusted."""
    restored = RadarState.from_dict({"version": 99, "alerted_outpoints": ["a"]})
    assert restored.alerted_outpoints == []


def test_missing_state_file_is_first_run(tmp_path):
    state = LocalState(str(tmp_path / "absent.json")).load()
    assert state.last_scanned_height is None
    assert state.alerted_outpoints == []


def test_corrupt_state_file_does_not_crash(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not json")
    state = LocalState(str(path)).load()
    assert state.alerted_outpoints == []


def test_open_state_prefers_kv_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOMATION_KV_TOKEN", "tok")
    monkeypatch.setenv("AUTOMATION_API_URL", "https://example.invalid/api/automation")
    assert open_state(str(tmp_path / "s.json")).name == "kv"


def test_open_state_falls_back_to_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOMATION_KV_TOKEN", raising=False)
    monkeypatch.delenv("AUTOMATION_API_URL", raising=False)
    backend = open_state(str(tmp_path / "s.json"))
    assert backend.name == "local"


def test_kv_state_reads_and_writes(monkeypatch):
    """Exercised against a fake HTTP layer so no network is involved."""
    from dormant_radar import state as state_module

    stored = {}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        if request.method == "PUT":
            stored["value"] = json.loads(request.data.decode())
            return FakeResponse({})
        if "value" in stored:
            return FakeResponse({"value": stored["value"]})
        import urllib.error

        raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(state_module.urllib.request, "urlopen", fake_urlopen)

    backend = state_module.KVState(base_url="https://example.invalid/api/automation", token="t")
    assert backend.load().alerted_outpoints == []          # first run

    backend.save(RadarState(last_scanned_height=42, alerted_outpoints=["x:0"]))
    loaded = backend.load()
    assert loaded.last_scanned_height == 42
    assert loaded.alerted_outpoints == ["x:0"]


def test_kv_state_survives_network_failure(monkeypatch):
    """A KV outage must degrade to re-scanning, never crash the run."""
    from dormant_radar import state as state_module

    def boom(request, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(state_module.urllib.request, "urlopen", boom)
    backend = state_module.KVState(base_url="https://example.invalid", token="t")
    assert backend.load().alerted_outpoints == []
    backend.save(RadarState())  # must not raise
"""Store tests: real SQLite, no mocks."""

import os

from dormant_radar.models import CauseScore, Outpoint, WakeUp
from dormant_radar.store import Store


def make_wakeup(txid="aa" * 32, height=900_000, value=500_000_000, years=6.0, cause=None):
    return WakeUp(
        txid=txid,
        spend_block_height=height,
        spent=Outpoint("bb" * 32, 0),
        value_sats=value,
        address="bc1qexample",
        dormant_blocks=int(years * 52_560),
        dormant_years=years,
        script_type="p2wpkh",
        observed_at=1_700_000_000.0,
        cause=cause,
    )


def make_cause(hypothesis="holding"):
    return CauseScore(
        hypothesis=hypothesis,
        confidence=0.4,
        distribution={"lost": 0.2, "holding": 0.6, "structural": 0.2},
        rationale=["because"],
    )


def test_add_and_list_roundtrip(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    try:
        added = store.add_wakeups([make_wakeup(cause=make_cause())])
        assert added == 1
        events = store.list_wakeups()
        assert len(events) == 1
        assert events[0]["txid"] == "aa" * 32
        assert events[0]["cause"]["hypothesis"] == "holding"
        assert events[0]["value_btc"] == 5.0
    finally:
        store.close()


def test_duplicate_wakeup_is_ignored(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    try:
        wakeup = make_wakeup()
        assert store.add_wakeups([wakeup]) == 1
        assert store.add_wakeups([wakeup]) == 0
        assert store.stats()["total_wakeups"] == 1
    finally:
        store.close()


def test_filters(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    try:
        store.add_wakeups(
            [
                make_wakeup(txid="a1" * 32, height=800_000, value=50_000_000, cause=make_cause("lost")),
                make_wakeup(txid="a2" * 32, height=900_000, value=500_000_000, cause=make_cause("holding")),
            ]
        )
        assert len(store.list_wakeups(min_value_sats=100_000_000)) == 1
        assert len(store.list_wakeups(since_height=850_000)) == 1
        assert len(store.list_wakeups(hypothesis="lost")) == 1
        assert store.list_wakeups(hypothesis="structural") == []
    finally:
        store.close()


def test_stats(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    try:
        store.add_wakeups([make_wakeup(cause=make_cause("holding"))])
        stats = store.stats()
        assert stats["total_wakeups"] == 1
        assert stats["total_value_btc"] == 5.0
        assert stats["latest_height"] == 900_000
        assert stats["by_hypothesis"] == {"holding": 1}
    finally:
        store.close()


def test_scan_state_roundtrip(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    try:
        assert store.get_state("last_scanned_height") is None
        store.set_state("last_scanned_height", 900_000)
        assert store.get_state("last_scanned_height") == 900_000
        store.set_state("last_scanned_height", 900_010)
        assert store.get_state("last_scanned_height") == 900_010
    finally:
        store.close()


def test_creates_missing_parent_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "radar.db"
    store = Store(str(nested))
    try:
        assert os.path.exists(nested)
    finally:
        store.close()
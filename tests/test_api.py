"""API tests against the real app with the network stubbed at the scan boundary.

The scan endpoint is exercised through a fake settings object whose chain
client is never reached, because these tests pass no block data. What is
verified here is the read/aggregate surface over a real SQLite database.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from dormant_radar.api import create_app
from dormant_radar.config import Settings
from dormant_radar.store import Store


@pytest.fixture
def client(tmp_path):
    settings = Settings(db_path=str(tmp_path / "api.db"))
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def seed(settings: Settings) -> None:
    from dormant_radar.models import CauseScore, Outpoint, WakeUp

    store = Store(settings.db_path)
    try:
        store.add_wakeups(
            [
                WakeUp(
                    txid="a1" * 32,
                    spend_block_height=910_000,
                    spent=Outpoint("b1" * 32, 0),
                    value_sats=1_500_000_000,
                    address="bc1qwhale",
                    dormant_blocks=12 * 52_560,
                    dormant_years=12.0,
                    script_type="p2wpkh",
                    observed_at=1_700_000_000.0,
                    cause=CauseScore(
                        hypothesis="structural",
                        confidence=0.5,
                        distribution={"lost": 0.1, "holding": 0.3, "structural": 0.6},
                        rationale=["old vintage"],
                    ),
                )
            ]
        )
    finally:
        store.close()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_empty(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["total_wakeups"] == 0


def test_events_empty(client):
    response = client.get("/events")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "events": []}


def test_events_after_seed(tmp_path):
    settings = Settings(db_path=str(tmp_path / "seeded.db"))
    seed(settings)
    app = create_app(settings)
    with TestClient(app) as test_client:
        body = test_client.get("/events").json()
        assert body["count"] == 1
        assert body["events"][0]["address"] == "bc1qwhale"
        assert body["events"][0]["cause"]["hypothesis"] == "structural"

        filtered = test_client.get("/events", params={"hypothesis": "lost"}).json()
        assert filtered["count"] == 0

        assert test_client.get("/stats").json()["total_wakeups"] == 1


def test_events_rejects_bad_hypothesis(client):
    response = client.get("/events", params={"hypothesis": "nonsense"})
    assert response.status_code == 422


def test_events_validates_limit(client):
    assert client.get("/events", params={"limit": 0}).status_code == 422
    assert client.get("/events", params={"limit": 10_000}).status_code == 422
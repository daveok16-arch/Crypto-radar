"""Price helper tests using a fake HTTP session (no network)."""

from dormant_radar.price import PriceContext


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_price_multiple_computes_ratio(monkeypatch):
    series = [[i, 100.0] for i in range(100)] + [[100, 200.0]]
    monkeypatch.setattr(
        "dormant_radar.price.requests.get",
        lambda *a, **k: FakeResponse({"prices": series}),
    )
    multiple = PriceContext().price_multiple()
    assert multiple is not None
    # Last point is 200 against a mean near 100, so clearly above 1.
    assert multiple > 1.5


def test_price_multiple_returns_none_on_failure(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr("dormant_radar.price.requests.get", boom)
    assert PriceContext().price_multiple() is None


def test_price_multiple_returns_none_on_short_series(monkeypatch):
    monkeypatch.setattr(
        "dormant_radar.price.requests.get",
        lambda *a, **k: FakeResponse({"prices": [[1, 100.0]]}),
    )
    assert PriceContext().price_multiple() is None
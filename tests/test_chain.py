"""Chain client tests.

The session is a hand-written fake rather than a mock library. It records
the requested URL and returns recorded mempool.space payloads, so the real
parsing code runs. Payloads below are trimmed copies of live responses and
exist to lock in two bugs found during development: `/block-height/{h}`
returns plain text, and the spent outpoint lives on the input (`vin.txid`),
not inside `prevout`.
"""

from dormant_radar.chain import ChainError, MempoolClient
from dormant_radar.config import Settings

# Trimmed from a live mempool.space response. Note txid/vout sit on the
# input and the coinbase input has no prevout.
LIVE_TX = {
    "txid": "f1b170e2adf252b52b7b3e058ee851ac1c91648db6fdd3cb3856429f7d238d89",
    "fee": 1410,
    "status": {"confirmed": True, "block_height": 967762},
    "vin": [
        {
            "is_coinbase": True,
            "prevout": None,
            "txid": None,
            "vout": None,
        },
        {
            "is_coinbase": False,
            "txid": "1c7f8debf0cf8a5d5b84dc128cbb04d211f2608a480e199438533a31c72f827a",
            "vout": 1,
            "prevout": {
                "scriptpubkey_type": "v0_p2wpkh",
                "scriptpubkey_address": "bc1qryhgpmfv03qjhhp2dj8nw8g4ewg08jzmgy3cyx",
                "value": 40432148,
            },
        },
    ],
    "vout": [
        {"value": 40000000, "scriptpubkey_address": "bc1qdest"},
        {"value": 430738, "scriptpubkey_address": "bc1qchange"},
    ],
}


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.text = body if isinstance(body, str) else ""

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("unexpected status")


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[str] = []
        self.headers: dict = {}

    def get(self, url, timeout=None):
        self.calls.append(url)
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                if isinstance(response, Exception):
                    raise response
                return response
        return FakeResponse(None, status_code=404)


def client_with(routes) -> tuple[MempoolClient, FakeSession]:
    session = FakeSession(routes)
    return MempoolClient(Settings(max_retries=1), session=session), session


def test_block_hash_parses_plain_text():
    client, _ = client_with(
        {"/block-height/100": FakeResponse("0000abcd" + "0" * 56)}
    )
    assert client.block_hash(100) == "0000abcd" + "0" * 56


def test_block_hash_missing_returns_none():
    client, _ = client_with({})
    assert client.block_hash(100) is None


def test_tip_height_requires_int():
    client, _ = client_with({"/blocks/tip/height": FakeResponse("not-an-int")})
    try:
        client.tip_height()
    except ChainError:
        return
    raise AssertionError("expected ChainError for non-int tip height")


def test_transaction_parses_vin_outpoint_and_prevout():
    client, _ = client_with(
        {
            "/tx/f1b170e2adf252b52b7b3e058ee851ac1c91648db6fdd3cb3856429f7d238d89": FakeResponse(
                LIVE_TX
            ),
            "/tx/1c7f8debf0cf8a5d5b84dc128cbb04d211f2608a480e199438533a31c72f827a": FakeResponse(
                {"status": {"confirmed": True, "block_height": 967760}}
            ),
        }
    )
    tx = client.transaction(LIVE_TX["txid"])
    assert tx is not None
    # The coinbase input must be dropped, leaving exactly the real spend.
    assert len(tx.inputs) == 1
    spend = tx.inputs[0]
    assert str(spend.outpoint) == (
        "1c7f8debf0cf8a5d5b84dc128cbb04d211f2608a480e199438533a31c72f827a:1"
    )
    assert spend.prevout_value_sats == 40432148
    assert spend.prevout_address == "bc1qryhgpmfv03qjhhp2dj8nw8g4ewg08jzmgy3cyx"
    assert spend.prevout_block_height == 967760
    assert tx.output_addresses == ["bc1qdest", "bc1qchange"]
    assert tx.total_output_sats == 40430738


def test_transaction_missing_returns_none():
    client, _ = client_with({})
    assert client.transaction("deadbeef") is None


def test_retries_then_succeeds(monkeypatch):
    import dormant_radar.chain as chain_module

    monkeypatch.setattr(chain_module.time, "sleep", lambda _s: None)
    session = FakeSession({"/blocks/tip/height": FakeResponse(None, status_code=429)})

    calls = {"n": 0}

    def flaky_get(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(None, status_code=429)
        return FakeResponse(967762)

    session.get = flaky_get
    client = MempoolClient(Settings(max_retries=2), session=session)
    assert client.tip_height() == 967762
    assert calls["n"] == 2


def test_gives_up_after_max_retries(monkeypatch):
    import dormant_radar.chain as chain_module

    monkeypatch.setattr(chain_module.time, "sleep", lambda _s: None)
    session = FakeSession({"/blocks/tip/height": FakeResponse(None, status_code=500)})
    client = MempoolClient(Settings(max_retries=2), session=session)
    try:
        client.tip_height()
    except ChainError as exc:
        assert "giving up" in str(exc)
        return
    raise AssertionError("expected ChainError")


def test_prevout_height_is_cached():
    client, session = client_with(
        {
            "/tx/aa": FakeResponse({"status": {"confirmed": True, "block_height": 5}}),
        }
    )
    assert client.block_height_for_tx("aa") == 5
    before = len(session.calls)
    assert client.block_height_for_tx("aa") == 5
    assert len(session.calls) == before
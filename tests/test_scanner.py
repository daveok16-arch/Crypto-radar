"""Scanner tests.

A hand-written in-memory chain stands in for mempool.space. This is a real
collaborator implementing the same interface, not a mock: the scanner logic,
detector, scorer, and SQLite writes all execute for real. A fake is used
only because live block contents are nondeterministic and tests must not
touch the network.
"""

from dormant_radar.config import Settings
from dormant_radar.detector import detect_wakeups
from dormant_radar.models import Outpoint, Tx, TxInput
from dormant_radar.scanner import scan_once
from dormant_radar.store import Store

ONE_YEAR = 52_560
DORMANT = 4 * ONE_YEAR


class FakeChain:
    """Deterministic stand-in for MempoolClient."""

    def __init__(self, tip: int, blocks: dict[str, list[str]], txs: dict[str, Tx]):
        self.tip = tip
        self.blocks = blocks
        self.txs = txs

    def tip_height(self) -> int:
        return self.tip

    def block_hash(self, height: int) -> str | None:
        return f"hash-{height}" if f"hash-{height}" in self.blocks else None

    def block_txids(self, block_hash: str) -> list[str]:
        return self.blocks.get(block_hash, [])

    def transaction(self, txid: str) -> Tx | None:
        return self.txs.get(txid)


def make_old_spend_tx(txid: str, prevout_height: int, value: int) -> Tx:
    return Tx(
        txid=txid,
        block_height=None,
        inputs=[
            TxInput(
                outpoint=Outpoint("f0" * 32, 0),
                prevout_value_sats=value,
                prevout_address="bc1qdormant",
                prevout_block_height=prevout_height,
                prevout_script_type="p2wpkh",
            )
        ],
    )


def build_settings(tmp_path, window=2):
    return Settings(
        db_path=str(tmp_path / "scan.db"),
        scan_window_blocks=window,
        model_path=str(tmp_path / "no-model.npz"),
        cluster_path=str(tmp_path / "clusters.db"),
    )


def test_scan_detects_and_persists(tmp_path):
    settings = build_settings(tmp_path)
    tip = 900_000
    chain = FakeChain(
        tip=tip,
        blocks={
            f"hash-{tip}": ["tx-old"],
            f"hash-{tip - 1}": ["tx-new"],
        },
        txs={
            "tx-old": make_old_spend_tx("tx-old", prevout_height=100, value=2_000_000_000),
            "tx-new": make_old_spend_tx("tx-new", prevout_height=tip - 2, value=2_000_000_000),
        },
    )
    store = Store(settings.db_path)
    try:
        result = scan_once(settings, client=chain, store=store)
        assert result.blocks_examined == 2
        assert result.transactions_examined == 2
        assert len(result.wakeups) == 1
        assert result.wakeups[0].txid == "tx-old"
        assert result.new_wakeups == 1
        assert store.stats()["total_wakeups"] == 1
        assert store.get_state("last_scanned_height") == tip
    finally:
        store.close()


def test_scan_resumes_from_cursor(tmp_path):
    settings = build_settings(tmp_path)
    store = Store(settings.db_path)
    try:
        store.set_state("last_scanned_height", 899_990)
        chain = FakeChain(
            tip=899_992,
            blocks={f"hash-{h}": [] for h in (899_991, 899_992)},
            txs={},
        )
        result = scan_once(settings, client=chain, store=store)
        assert result.scanned_from == 899_991
        assert result.scanned_to == 899_992
    finally:
        store.close()


def test_scan_survives_bad_transaction(tmp_path):
    settings = build_settings(tmp_path)
    tip = 900_000

    class ExplodingChain(FakeChain):
        def transaction(self, txid):
            if txid == "boom":
                raise RuntimeError("exploded")
            return super().transaction(txid)

    chain = ExplodingChain(
        tip=tip,
        blocks={f"hash-{tip}": ["boom", "tx-old"]},
        txs={"tx-old": make_old_spend_tx("tx-old", 100, 1_000_000_000)},
    )
    store = Store(settings.db_path)
    try:
        result = scan_once(settings, client=chain, store=store)
        assert len(result.wakeups) == 1
        assert any("exploded" in e for e in result.errors)
        assert result.new_wakeups == 1
    finally:
        store.close()


def test_rescan_does_not_duplicate(tmp_path):
    settings = build_settings(tmp_path)
    tip = 900_000
    chain = FakeChain(
        tip=tip,
        blocks={f"hash-{tip}": ["tx-old"]},
        txs={"tx-old": make_old_spend_tx("tx-old", 100, 1_000_000_000)},
    )
    store = Store(settings.db_path)
    try:
        scan_once(settings, client=chain, store=store)
        store.set_state("last_scanned_height", None)
        result = scan_once(settings, client=chain, store=store)
        assert len(result.wakeups) == 1
        assert result.new_wakeups == 0
        assert store.stats()["total_wakeups"] == 1
    finally:
        store.close()


def test_scan_records_ownership_on_events(tmp_path):
    """End-to-end: ownership learned from the graph must reach the stored event.

    The first transaction merges two addresses into a cluster; the second
    spends from that cluster back to a member. The stored event must report a
    known cluster and a self-transfer, which the address-only check could only
    do when the exact address repeats.
    """
    from dormant_radar.cluster import OwnershipGraph

    settings = build_settings(tmp_path, window=2)
    tip = 900_000

    cluster_setup = make_old_spend_tx("setup", prevout_height=1, value=10_000)
    # Co-spend two addresses so they join one cluster.
    cluster_setup.inputs.append(
        TxInput(
            outpoint=Outpoint("f9" * 32, 0),
            prevout_value_sats=10_000,
            prevout_address="bc1qsecond",
            prevout_block_height=1,
            prevout_script_type="p2wpkh",
        )
    )

    spend = make_old_spend_tx("tx-old", prevout_height=100, value=1_000_000_000)
    spend.inputs[0].prevout_address = "bc1qdormant"
    # Value returns to the *other* address in the cluster. Addresses and values
    # are parallel arrays and must be set together.
    spend.output_addresses = ["bc1qsecond"]
    spend.output_values_sats = [1_000_000_000]

    chain = FakeChain(
        tip=tip,
        blocks={f"hash-{tip - 1}": ["setup"], f"hash-{tip}": ["tx-old"]},
        txs={"setup": cluster_setup, "tx-old": spend},
    )
    graph = OwnershipGraph(settings.cluster_path)
    store = Store(settings.db_path)
    try:
        result = scan_once(settings, client=chain, store=store, graph=graph)
        assert result.new_wakeups == 1
        event = store.list_wakeups()[0]
        assert event["ownership"] is not None
        assert event["ownership"]["known"] is True
        assert event["ownership"]["cluster_size"] == 2
        assert event["ownership"]["is_self_transfer"] is True
    finally:
        store.close()
        graph.close()


def test_scan_works_without_clustering(tmp_path):
    """Clustering is optional; disabling it must not break a scan."""
    settings = Settings(
        db_path=str(tmp_path / "scan.db"),
        scan_window_blocks=1,
        model_path=str(tmp_path / "no-model.npz"),
        use_clustering=False,
    )
    tip = 900_000
    chain = FakeChain(
        tip=tip,
        blocks={f"hash-{tip}": ["tx-old"]},
        txs={"tx-old": make_old_spend_tx("tx-old", 100, 1_000_000_000)},
    )
    store = Store(settings.db_path)
    try:
        result = scan_once(settings, client=chain, store=store)
        assert result.new_wakeups == 1
        assert "ownership" not in store.list_wakeups()[0]
    finally:
        store.close()


def test_scan_attaches_neural_payload_when_model_present(tmp_path):
    """End-to-end: a trained model must reach the stored event.

    Trains a real model on synthetic ordinary spends, then runs a scan whose
    fake chain serves one dormant spend, and checks the hybrid neural fields
    are present in what gets persisted.
    """
    import numpy as np

    from dormant_radar.anomaly import AnomalyDetector
    from dormant_radar.features import extract_features
    from dormant_radar.trainer import TrainingSet, train_autoencoder

    model_path = str(tmp_path / "model.npz")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        tx = make_old_spend_tx(f"t{i}", prevout_height=1, value=100_000_000)
        wakeup = detect_wakeups(
            tx, spend_height=900_000, dormant_after_blocks=1, min_spent_sats=0
        )[0]
        rows.append(extract_features(wakeup, tx=tx))
    train_autoencoder(TrainingSet(matrix=np.vstack(rows)), model_path=model_path, epochs=60)
    detector = AnomalyDetector.load(model_path)
    assert detector is not None

    settings = Settings(
        db_path=str(tmp_path / "scan.db"),
        scan_window_blocks=1,
        model_path=model_path,
    )
    tip = 900_000
    chain = FakeChain(
        tip=tip,
        blocks={f"hash-{tip}": ["tx-old"]},
        txs={"tx-old": make_old_spend_tx("tx-old", 100, 1_000_000_000)},
    )
    store = Store(settings.db_path)
    try:
        result = scan_once(settings, client=chain, store=store, detector=detector)
        assert result.new_wakeups == 1
        event = store.list_wakeups()[0]
        assert event["neural"] is not None
        assert event["neural"]["anomaly"] is not None
        assert "rule_distribution" in event["neural"]
    finally:
        store.close()
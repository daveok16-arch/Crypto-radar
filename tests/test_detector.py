"""Detector tests: the aged-output spend rule."""

from dormant_radar.detector import blocks_to_years, detect_wakeups, scan_transactions
from dormant_radar.models import Outpoint, Tx, TxInput

ONE_YEAR = 52_560


def make_tx(txid="aa" * 32, prevout_height=100, value=500_000_000, address="bc1qexample"):
    return Tx(
        txid=txid,
        block_height=None,
        inputs=[
            TxInput(
                outpoint=Outpoint("bb" * 32, 0),
                prevout_value_sats=value,
                prevout_address=address,
                prevout_block_height=prevout_height,
                prevout_script_type="witness_v0_keyhash",
            )
        ],
    )


def test_detects_spend_older_than_threshold():
    tx = make_tx(prevout_height=100)
    found = detect_wakeups(tx, spend_height=100 + 5 * ONE_YEAR, dormant_after_blocks=4 * ONE_YEAR)
    assert len(found) == 1
    wakeup = found[0]
    assert wakeup.txid == tx.txid
    assert wakeup.dormant_blocks == 5 * ONE_YEAR
    assert round(wakeup.dormant_years, 1) == 5.0
    assert wakeup.address == "bc1qexample"
    assert wakeup.value_sats == 500_000_000


def test_ignores_spend_below_threshold():
    tx = make_tx(prevout_height=100)
    found = detect_wakeups(tx, spend_height=100 + 2 * ONE_YEAR, dormant_after_blocks=4 * ONE_YEAR)
    assert found == []


def test_ignores_dust_below_value_floor():
    tx = make_tx(prevout_height=100, value=1000)
    found = detect_wakeups(
        tx,
        spend_height=100 + 5 * ONE_YEAR,
        dormant_after_blocks=4 * ONE_YEAR,
        min_spent_sats=100_000_000,
    )
    assert found == []


def test_ignores_unknown_prevout_height():
    tx = Tx(
        txid="cc" * 32,
        block_height=None,
        inputs=[TxInput(outpoint=Outpoint("dd" * 32, 1), prevout_value_sats=10**9)],
    )
    assert detect_wakeups(tx, spend_height=900_000, dormant_after_blocks=1) == []


def test_multiple_dormant_inputs_produce_multiple_wakeups():
    tx = Tx(
        txid="ee" * 32,
        block_height=None,
        inputs=[
            TxInput(Outpoint("f1" * 32, 0), 200_000_000, "addr-a", 1_000, "p2wpkh"),
            TxInput(Outpoint("f2" * 32, 1), 300_000_000, "addr-b", 400_000, "p2pkh"),
            TxInput(Outpoint("f3" * 32, 2), 400_000_000, "addr-c", 999_000, "p2wpkh"),
        ],
    )
    found = detect_wakeups(tx, spend_height=1_000_000, dormant_after_blocks=10 * ONE_YEAR)
    assert [w.address for w in found] == ["addr-a", "addr-b"]


def test_scan_transactions_aggregates():
    pair = (make_tx(prevout_height=1), 900_000)
    found = scan_transactions([pair, pair], dormant_after_blocks=ONE_YEAR, observed_at=42.0)
    assert len(found) == 2
    assert all(w.observed_at == 42.0 for w in found)


def test_blocks_to_years():
    assert blocks_to_years(ONE_YEAR) == 1.0
    assert round(blocks_to_years(4 * ONE_YEAR), 2) == 4.0
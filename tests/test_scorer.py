"""Scorer tests: cause attribution must be inspectable and directional."""

from dormant_radar.detector import blocks_to_years
from dormant_radar.models import Outpoint, Tx, TxInput, WakeUp
from dormant_radar.scorer import HYPOTHESES, score_cause

ONE_YEAR = 52_560


def make_wakeup(years=5.0, address="bc1qowner", value=500_000_000):
    blocks = int(years * ONE_YEAR)
    return WakeUp(
        txid="aa" * 32,
        spend_block_height=900_000,
        spent=Outpoint("bb" * 32, 0),
        value_sats=value,
        address=address,
        dormant_blocks=blocks,
        dormant_years=blocks_to_years(blocks),
        script_type="p2wpkh",
        observed_at=1_700_000_000.0,
    )


def test_distribution_is_valid_probability_simplex():
    result = score_cause(make_wakeup())
    assert set(result.distribution) == set(HYPOTHESES)
    assert all(0.0 <= p <= 1.0 for p in result.distribution.values())
    assert abs(sum(result.distribution.values()) - 1.0) < 1e-6


def test_hypothesis_is_argmax():
    result = score_cause(make_wakeup())
    assert result.hypothesis == max(result.distribution, key=result.distribution.get)


def test_spend_weakens_loss_hypothesis():
    result = score_cause(make_wakeup())
    assert result.distribution["lost"] < result.distribution["holding"]


def test_self_transfer_pushes_toward_holding():
    wakeup = make_wakeup(address="bc1qowner")
    plain = Tx(txid="cc" * 32, block_height=900_000)
    self_move = Tx(
        txid="cc" * 32,
        block_height=900_000,
        output_addresses=["bc1qowner", "bc1qowner"],
    )
    baseline = score_cause(wakeup, tx=plain)
    moved = score_cause(wakeup, tx=self_move)
    assert moved.distribution["holding"] > baseline.distribution["holding"]
    assert any("returns to the originating address" in r for r in moved.rationale)


def test_many_inputs_push_toward_structural():
    wakeup = make_wakeup()
    consensus = Tx(
        txid="dd" * 32,
        block_height=900_000,
        inputs=[TxInput(Outpoint(f"{i:02x}" * 32, 0)) for i in range(12)],
    )
    result = score_cause(wakeup, tx=consensus)
    assert result.distribution["structural"] > score_cause(wakeup).distribution["structural"]


def test_hot_market_favours_holding():
    cold = score_cause(make_wakeup(), price_multiple=1.0)
    hot = score_cause(make_wakeup(), price_multiple=3.0)
    assert hot.distribution["holding"] > cold.distribution["holding"]


def test_very_old_vintage_shifts_toward_structural():
    young = score_cause(make_wakeup(years=5.0))
    old = score_cause(make_wakeup(years=12.0))
    assert old.distribution["structural"] > young.distribution["structural"]


def test_rationale_is_always_populated():
    result = score_cause(make_wakeup())
    assert result.rationale
    assert all(isinstance(r, str) and r for r in result.rationale)


def test_confidence_in_unit_range():
    result = score_cause(make_wakeup(), tx=Tx(txid="ee" * 32, block_height=1))
    assert 0.0 <= result.confidence <= 1.0


# --- ownership evidence (from clustering) ------------------------------


def test_cluster_self_transfer_favours_holding():
    from dormant_radar.cluster import OwnershipSummary

    baseline = score_cause(make_wakeup())
    owned = score_cause(
        make_wakeup(),
        ownership=OwnershipSummary(True, 3, "addrroot", 1.0, True),
    )
    assert owned.distribution["holding"] > baseline.distribution["holding"]
    assert any("ownership cluster" in r for r in owned.rationale)


def test_cluster_self_transfer_detected_across_different_address():
    """Clustering must catch self-transfers that the same-address check misses.

    The value returns to a *different* address owned by the same cluster, so
    address equality alone would see nothing.
    """
    from dormant_radar.cluster import OwnershipSummary

    tx = Tx(
        txid="f0" * 32,
        block_height=900_000,
        output_values_sats=[500_000_000],
        output_addresses=["bc1qdifferentalias"],
    )
    wakeup = make_wakeup(address="bc1qoriginal")
    without = score_cause(wakeup, tx=tx)
    with_cluster = score_cause(
        wakeup, tx=tx, ownership=OwnershipSummary(True, 4, "root", 1.0, True)
    )
    assert with_cluster.distribution["holding"] > without.distribution["holding"]


def test_external_disposal_not_treated_as_self_transfer():
    from dormant_radar.cluster import OwnershipSummary

    result = score_cause(
        make_wakeup(),
        ownership=OwnershipSummary(True, 3, "root", 0.0, False),
    )
    assert any("genuine disposal" in r for r in result.rationale)


def test_large_cluster_favours_structural():
    from dormant_radar.cluster import OwnershipSummary

    small = score_cause(make_wakeup(), ownership=OwnershipSummary(True, 2, "r", 0.0, False))
    large = score_cause(
        make_wakeup(), ownership=OwnershipSummary(True, 500, "r", 0.0, False)
    )
    assert large.distribution["structural"] > small.distribution["structural"]
    assert any("custodial or institutional" in r for r in large.rationale)


def test_unknown_ownership_falls_back_to_same_address_rationale():
    from dormant_radar.cluster import OwnershipSummary

    tx = Tx(
        txid="ab" * 32,
        block_height=900_000,
        output_values_sats=[500_000_000],
        output_addresses=["bc1qowner"],
    )
    result = score_cause(
        make_wakeup(address="bc1qowner"),
        tx=tx,
        ownership=OwnershipSummary(False, 0, None, 0.0, False),
    )
    assert any("clustering unavailable" in r for r in result.rationale)
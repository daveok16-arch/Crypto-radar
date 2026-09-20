"""Address clustering tests.

Two properties matter most and are tested hardest:

1. **Determinism** — the cluster identifier must depend only on which
   addresses are grouped, never on transaction processing order or a restart.
   Otherwise the same wallet could report different cluster sizes across runs.

2. **CoinJoin exclusion** — a false merge is permanent and spreads
   transitively, so co-spends must be refused when they look collaborative.
"""

import pytest

from dormant_radar.cluster import (
    OwnershipGraph,
    is_likely_coinjoin,
)
from dormant_radar.models import Outpoint, Tx, TxInput


def make_tx(txid, spenders, outputs, fee=1000):
    """Build a transaction from input addresses and (address, value) outputs."""
    return Tx(
        txid=txid,
        block_height=1,
        inputs=[
            TxInput(Outpoint(f"{i:02x}" * 32, 0), 100_000, addr, 900_000, "v0_p2wpkh")
            for i, addr in enumerate(spenders)
        ],
        output_values_sats=[v for _, v in outputs],
        output_addresses=[a for a, _ in outputs],
        fee_sats=fee,
    )


@pytest.fixture
def graph(tmp_path):
    g = OwnershipGraph(str(tmp_path / "clusters.db"))
    yield g
    g.close()


# --- union-find correctness --------------------------------------------


def test_unknown_address_has_no_cluster(graph):
    assert graph.find("unknown") is None
    assert graph.cluster_size("unknown") == 0
    assert graph.cluster_id("unknown") is None


def test_coinjoin_single_spend_creates_cluster_of_two(graph):
    graph.observe(make_tx("t1", ["addrA", "addrB"], [("out", 50_000)]))
    assert graph.find("addrA") == graph.find("addrB")
    assert graph.cluster_size("addrA") == 2


def test_transitivity_across_transactions(graph):
    """A-B then B-C must place all three in one cluster."""
    graph.observe(make_tx("t1", ["addrA", "addrB"], [("out", 1)]))
    graph.observe(make_tx("t2", ["addrB", "addrC"], [("out", 1)]))
    assert graph.find("addrA") == graph.find("addrB") == graph.find("addrC")
    assert graph.cluster_size("addrA") == 3


def test_merge_combines_sizes_correctly(graph):
    graph.observe(make_tx("t1", ["a", "b"], [("out", 1)]))
    graph.observe(make_tx("t2", ["c", "d"], [("out", 1)]))
    assert graph.cluster_size("a") == 2
    graph.observe(make_tx("t3", ["b", "c"], [("out", 1)]))
    assert graph.cluster_size("a") == 4
    assert graph.cluster_size("d") == 4


def test_separate_clusters_stay_separate(graph):
    graph.observe(make_tx("t1", ["a", "b"], [("out", 1)]))
    graph.observe(make_tx("t2", ["c", "d"], [("out", 1)]))
    assert graph.find("a") != graph.find("c")
    assert graph.cluster_size("a") == 2


def test_every_member_agrees_on_root_after_merge(graph):
    """No member may be left pointing at a stale root."""
    graph.observe(make_tx("t1", ["m", "n"], [("out", 1)]))
    graph.observe(make_tx("t2", ["o", "p"], [("out", 1)]))
    graph.observe(make_tx("t3", ["n", "o"], [("out", 1)]))
    roots = {graph.find(a) for a in ("m", "n", "o", "p")}
    assert len(roots) == 1
    assert graph.cluster_size("p") == 4


# --- determinism -------------------------------------------------------


def test_cluster_id_is_order_independent(tmp_path):
    """Same merges in a different order must yield the same identifier.

    Root is the lexicographically smallest address, so the identifier is a
    function of set membership rather than processing history.
    """
    forward = OwnershipGraph(str(tmp_path / "f.db"))
    reverse = OwnershipGraph(str(tmp_path / "r.db"))
    try:
        forward.observe(make_tx("t1", ["bbb", "ccc"], [("out", 1)]))
        forward.observe(make_tx("t2", ["aaa", "bbb"], [("out", 1)]))

        reverse.observe(make_tx("t2", ["aaa", "bbb"], [("out", 1)]))
        reverse.observe(make_tx("t1", ["bbb", "ccc"], [("out", 1)]))

        assert forward.find("ccc") == reverse.find("ccc") == "aaa"
        assert forward.cluster_size("aaa") == reverse.cluster_size("aaa") == 3
    finally:
        forward.close()
        reverse.close()


def test_clusters_survive_reopen(tmp_path):
    path = str(tmp_path / "persist.db")
    first = OwnershipGraph(path)
    first.observe(make_tx("t1", ["a", "b"], [("out", 1)]))
    first.close()

    second = OwnershipGraph(path)
    try:
        assert second.find("a") == second.find("b")
        assert second.cluster_size("a") == 2
        assert second.has_seen("t1")
    finally:
        second.close()


# --- idempotency -------------------------------------------------------


def test_reobserving_same_tx_does_not_change_state(graph):
    tx = make_tx("t1", ["a", "b"], [("out", 1)])
    first = graph.observe(tx)
    assert first.merged is True
    assert graph.cluster_size("a") == 2

    second = graph.observe(tx)
    assert second.already_seen is True
    assert second.merged is False
    assert graph.cluster_size("a") == 2


def test_single_address_spend_merges_nothing(graph):
    decision = graph.observe(make_tx("t1", ["solo"], [("out", 1)]))
    assert decision.merged is False
    assert decision.reason == "single input address"


# --- coinjoin detection ------------------------------------------------


def test_equal_output_coinjoin_is_detected():
    tx = make_tx(
        "cj",
        ["p1", "p2", "p3", "p4", "p5"],
        [(f"o{i}", 1_000_000) for i in range(5)] + [("change", 77_777)],
    )
    is_cj, reason = is_likely_coinjoin(tx)
    assert is_cj is True
    assert "equal outputs" in reason


def test_equal_output_coinjoin_merges_nothing(graph):
    """The critical safety property: a CoinJoin must not fuse its participants."""
    tx = make_tx(
        "cj",
        ["p1", "p2", "p3", "p4", "p5"],
        [(f"o{i}", 1_000_000) for i in range(5)] + [("change", 77_777)],
    )
    decision = graph.observe(tx)
    assert decision.merged is False
    assert decision.reason.startswith("coinjoin excluded")
    # Participants are registered as known singletons, but never fused.
    roots = {graph.find(a) for a in ("p1", "p2", "p3", "p4", "p5")}
    assert len(roots) == 5
    assert graph.cluster_size("p1") == 1


def test_ordinary_payment_is_not_flagged_as_coinjoin():
    """Round-number payments must not be mistaken for collaborative spends."""
    tx = make_tx("pay", ["a", "b"], [("recipient", 1_000_000), ("change", 500_000)])
    is_cj, _ = is_likely_coinjoin(tx)
    assert is_cj is False


def test_payment_with_equal_outputs_but_two_inputs_not_flagged():
    """Equal outputs alone are not enough; a co-spend needs many participants."""
    tx = make_tx(
        "payroll",
        ["wallet", "wallet2"],
        [(f"w{i}", 1_000_000) for i in range(6)],
    )
    is_cj, reason = is_likely_coinjoin(tx)
    assert is_cj is False
    assert reason == "too few inputs to be a co-spend"


def test_consolidation_with_round_outputs_not_flagged():
    """Many inputs funnelled into few outputs is consolidation, not a co-spend."""
    tx = make_tx(
        "consol",
        [f"a{i}" for i in range(20)],
        [("big", 1_000_000), ("big2", 1_000_000), ("big3", 1_000_000),
         ("big4", 1_000_000), ("big5", 1_000_000), ("big6", 1_000_000)],
    )
    is_cj, reason = is_likely_coinjoin(tx)
    assert is_cj is False
    assert "consolidation-like" in reason


def test_dust_outputs_ignored_in_coinjoin_analysis():
    # 3 repeated 1_000_000 outputs among 4 non-dust outputs (a 1_000_000 change
    # plus 3 equal participant outputs). Dust is excluded from the analysis.
    tx = make_tx(
        "cj",
        ["p1", "p2", "p3"],
        [(f"o{i}", 1_000_000) for i in range(3)]
        + [("change", 1_500_000)]
        + [("dust", 100), ("dust2", 200)],
    )
    is_cj, _ = is_likely_coinjoin(tx)
    assert is_cj is True


# --- ownership summary -------------------------------------------------


def test_summary_unknown_for_unseen_address(graph):
    tx = make_tx("t1", ["x"], [("y", 100)])
    summary = graph.summary_for("never-seen", tx, 100)
    assert summary.known is False
    assert summary.cluster_id is None
    assert summary.is_self_transfer is False


def test_summary_detects_self_transfer_by_cluster_membership(graph):
    """Value returning to a *different* address in the same cluster is a self-transfer."""
    graph.observe(make_tx("setup", ["addrA", "addrB"], [("z", 1)]))

    tx = Tx(
        txid="spend",
        block_height=2,
        output_values_sats=[100_000],
        output_addresses=["addrB"],
    )
    summary = graph.summary_for("addrA", tx, 100_000)
    assert summary.known is True
    assert summary.cluster_size == 2
    assert summary.is_self_transfer is True
    assert summary.owned_output_share == 1.0


def test_summary_distinguishes_external_payment(graph):
    graph.observe(make_tx("setup", ["addrA", "addrB"], [("z", 1)]))
    tx = Tx(
        txid="spend",
        block_height=2,
        output_values_sats=[100_000],
        output_addresses=["someExchange"],
    )
    summary = graph.summary_for("addrA", tx, 100_000)
    assert summary.known is True
    assert summary.is_self_transfer is False
    assert summary.owned_output_share == 0.0


def test_summary_partial_self_transfer_share(graph):
    graph.observe(make_tx("setup", ["addrA", "addrB"], [("z", 1)]))
    tx = Tx(
        txid="spend",
        block_height=2,
        output_values_sats=[75_000, 25_000],
        output_addresses=["addrB", "external"],
    )
    summary = graph.summary_for("addrA", tx, 100_000)
    assert summary.owned_output_share == pytest.approx(0.75)


def test_summary_serialises(graph):
    graph.observe(make_tx("setup", ["addrA", "addrB"], [("z", 1)]))
    tx = Tx(
        txid="spend",
        block_height=2,
        output_values_sats=[100_000],
        output_addresses=["addrB"],
    )
    payload = graph.summary_for("addrA", tx, 100_000).as_dict()
    assert set(payload) == {
        "known",
        "cluster_size",
        "cluster_id",
        "owned_output_share",
        "is_self_transfer",
    }


def test_stats(graph):
    graph.observe(make_tx("t1", ["a", "b"], [("out", 1)]))
    graph.observe(make_tx("t2", ["c"], [("out", 1)]))
    stats = graph.stats()
    # Both of t2's addresses are registered as singletons, so the merged pair
    # plus the lone spend counts three addresses in two clusters.
    assert stats["addresses"] == 3
    assert stats["clusters"] == 2
    assert stats["observed_transactions"] == 2
    assert stats["largest_cluster"] == 2


def test_coinjoin_participants_remain_singletons(graph):
    """Registering participants as singletons must not create one big cluster."""
    tx = make_tx(
        "cj",
        ["p1", "p2", "p3", "p4", "p5"],
        [(f"o{i}", 1_000_000) for i in range(5)] + [("change", 77_777)],
    )
    graph.observe(tx)
    roots = {graph.find(a) for a in ("p1", "p2", "p3", "p4", "p5")}
    assert all(root is not None for root in roots)
    assert len(roots) == 5
    assert graph.stats()["largest_cluster"] == 1
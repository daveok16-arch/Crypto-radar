"""Backfill tests.

The properties that matter:

1. **Discovery** — historical co-spends must actually reach the graph, which is
   the entire point: an address that looks isolated today gains its cluster.
2. **Budget** — the run must respect its page and address caps, so it can never
   stall a scan cycle.
3. **Fairness** — least-recently-processed first, so no address starves.
4. **Resumability** — per-address progress is durable, so an interrupted sweep
   resumes rather than restarting.
5. **Isolation** — a chain error on one address must not abort the sweep.
"""

import pytest

from dormant_radar.backfill import Backfiller
from dormant_radar.config import Settings
from dormant_radar.models import Outpoint, Tx, TxInput


def make_tx(txid, spenders, outputs=(("out", 1000),)):
    return Tx(
        txid=txid,
        block_height=1,
        inputs=[
            TxInput(Outpoint(f"{i:02x}" * 32, 0), 100_000, addr, 900_000, "v0_p2wpkh")
            for i, addr in enumerate(spenders)
        ],
        output_values_sats=[v for _, v in outputs],
        output_addresses=[a for a, _ in outputs],
        fee_sats=1000,
    )


class FakeChain:
    """Serves canned address pages, recording what was requested."""

    def __init__(self, pages):
        # address -> list of pages; each page is a list of Tx
        self.pages = pages
        self.requests: list[tuple[str, str | None]] = []
        self.fail_addresses: set[str] = set()

    def address_txs(self, address, after_txid=None, limit=50):
        self.requests.append((address, after_txid))
        if address in self.fail_addresses:
            from dormant_radar.chain import ChainError

            raise ChainError(f"simulated failure for {address}")
        pages = self.pages.get(address, [])
        index = 0
        if after_txid is not None:
            # Find the page whose last txid matches the cursor.
            for i, page in enumerate(pages):
                if page and page[-1].txid == after_txid:
                    index = i + 1
                    break
            else:
                return [], None
        if index >= len(pages):
            return [], None
        page = pages[index]
        next_cursor = page[-1].txid if len(page) >= limit else None
        return page, next_cursor


def settings(tmp_path, addresses=10, pages=20):
    return Settings(
        cluster_path=str(tmp_path / "clusters.db"),
        backfill_enabled=True,
        backfill_max_addresses_per_run=addresses,
        backfill_max_pages_per_run=pages,
    )


@pytest.fixture
def graph(tmp_path):
    from dormant_radar.cluster import OwnershipGraph

    g = OwnershipGraph(str(tmp_path / "clusters.db"))
    yield g
    g.close()


def seed_address(graph, address):
    """Put a lone address into the graph, as a scan would."""
    from dormant_radar.cluster import ClusterDecision

    graph._ensure(address)


def test_backfill_discovers_historical_cluster(tmp_path, graph):
    """The core promise: history merges addresses that today look unrelated."""
    seed_address(graph, "addrA")
    assert graph.cluster_size("addrA") == 1

    chain = FakeChain(
        {
            "addrA": [[make_tx("hist1", ["addrA", "addrB", "addrC"], )]]
        }
    )
    s = settings(tmp_path)
    report = Backfiller(s, client=chain).run(graph)

    assert report.transactions_seen == 1
    assert graph.cluster_size("addrA") == 3
    assert graph.find("addrA") == graph.find("addrC")


def test_backfill_follows_pagination(tmp_path, graph):
    """A long history must be walked across pages, not just the first."""
    seed_address(graph, "addrA")
    page = [make_tx(f"t{i}", ["addrA"]) for i in range(50)]
    chain = FakeChain({"addrA": [page]})
    s = settings(tmp_path)

    # limit=50 means a full page yields a cursor, so a second call is needed.
    report = Backfiller(s, client=chain).run(graph)
    assert report.pages_fetched == 2  # the page, then the empty terminator
    assert len(chain.requests) == 2


def test_backfill_respects_page_budget(tmp_path, graph):
    """The budget is a hard cap: this must never run unbounded."""
    seed_address(graph, "addrA")
    page = [make_tx(f"t{i}", ["addrA"]) for i in range(50)]
    chain = FakeChain({"addrA": [page, page, page, page]})
    s = settings(tmp_path, pages=2)

    report = Backfiller(s, client=chain).run(graph)
    assert report.pages_fetched <= 2
    assert len(chain.requests) <= 2


def test_backfill_respects_address_budget(tmp_path, graph):
    for i in range(10):
        seed_address(graph, f"addr{i}")
    chain = FakeChain({f"addr{i}": [[]] for i in range(10)})
    s = settings(tmp_path, addresses=3)

    report = Backfiller(s, client=chain).run(graph)
    assert report.addresses_processed <= 3
    assert len({addr for addr, _ in chain.requests}) <= 3


def test_backfill_records_progress_for_resumption(tmp_path, graph):
    seed_address(graph, "addrA")
    chain = FakeChain({"addrA": [[]]})
    Backfiller(settings(tmp_path), client=chain).run(graph)

    assert graph.backfilled_count() == 1
    assert graph.stats()["backfilled_addresses"] == 1


def test_backfill_skips_already_backfilled_addresses(tmp_path, graph):
    """A second run must not re-walk what it already finished."""
    seed_address(graph, "addrA")
    chain = FakeChain({"addrA": [[]]})
    s = settings(tmp_path)

    Backfiller(s, client=chain).run(graph)
    first_requests = len(chain.requests)

    Backfiller(s, client=chain).run(graph)
    assert len(chain.requests) == first_requests, "should not re-request"


def test_long_history_resumes_instead_of_restarting(tmp_path, graph):
    """The bug this guards: restarting a budgeted walk never reaches the end.

    With a one-page budget and a multi-page history, each run must continue
    from the previous cursor. Restarting from the newest transaction would
    re-fetch the same first page forever and never complete.
    """
    seed_address(graph, "addrA")
    page1 = [make_tx(f"a{i}", ["addrA"]) for i in range(50)]
    page2 = [make_tx(f"b{i}", ["addrA"]) for i in range(50)]
    page3 = [make_tx(f"c{i}", ["addrA"]) for i in range(3)]  # short page ends it
    chain = FakeChain({"addrA": [page1, page2, page3]})
    s = settings(tmp_path, pages=1)

    first = Backfiller(s, client=chain).run(graph)
    assert first.addresses_completed == 0
    assert chain.requests[0][1] is None, "first walk starts at the newest tx"

    second = Backfiller(s, client=chain).run(graph)
    assert chain.requests[1][1] == page1[-1].txid, "second walk resumes from the cursor"
    assert second.addresses_completed == 0

    third = Backfiller(s, client=chain).run(graph)
    assert chain.requests[2][1] == page2[-1].txid
    assert third.addresses_completed == 1, "third run reaches the end"

    # Now that it is complete, it must not be walked again.
    requests_before = len(chain.requests)
    Backfiller(s, client=chain).run(graph)
    assert len(chain.requests) == requests_before


def test_newly_discovered_addresses_are_queued_for_later_runs(tmp_path, graph):
    """Backfill is breadth-first: discoveries wait for the next run.

    Walking an address discovers its co-spend partners. Those should join the
    queue for a later run rather than being walked immediately, so one busy
    address cannot consume the entire budget in a single pass.
    """
    seed_address(graph, "addrA")
    chain = FakeChain(
        {"addrA": [[make_tx("t1", ["addrA", "addrB", "addrC"])]], "addrB": [[]], "addrC": [[]]}
    )
    s = settings(tmp_path, addresses=1)

    first = Backfiller(s, client=chain).run(graph)
    assert first.addresses_processed == 1
    assert {addr for addr, _ in chain.requests} == {"addrA"}

    # The partners are now eligible.
    targets = graph.backfill_targets(10)
    assert "addrB" in targets and "addrC" in targets


def test_completed_address_is_not_retargeted(tmp_path, graph):
    seed_address(graph, "addrA")
    chain = FakeChain({"addrA": [[]]})
    Backfiller(settings(tmp_path), client=chain).run(graph)
    assert graph.backfill_targets(10) == []


def test_backfill_is_fair_across_runs(tmp_path, graph):
    """Least-recently-processed first, so no address starves."""
    for name in ("a", "b", "c"):
        seed_address(graph, name)
    chain = FakeChain({name: [[]] for name in ("a", "b", "c")})
    s = settings(tmp_path, addresses=1)

    Backfiller(s, client=chain).run(graph)
    first = {addr for addr, _ in chain.requests}
    Backfiller(s, client=chain).run(graph)
    second = {addr for addr, _ in chain.requests} - first

    assert first and second, "each run should cover a different address"
    assert first != second


def test_chain_error_on_one_address_does_not_abort(tmp_path, graph):
    for name in ("a", "b"):
        seed_address(graph, name)
    chain = FakeChain({"a": [[]], "b": [[]]})
    chain.fail_addresses = {"a"}
    s = settings(tmp_path)

    report = Backfiller(s, client=chain).run(graph)
    assert report.errors
    assert any("a" in err for err in report.errors)
    # b was still attempted.
    assert report.addresses_processed >= 1


def test_observation_error_does_not_abort(tmp_path, graph):
    seed_address(graph, "addrA")
    chain = FakeChain({"addrA": [[make_tx("t1", ["addrA", "addrB"])]]})

    class ExplodingGraph:
        """Graph whose observe raises once, to prove isolation."""

        def __init__(self, real):
            self.real = real
            self.calls = 0

        def observe(self, tx):
            self.calls += 1
            raise RuntimeError("observe failed")

        def backfill_targets(self, limit):
            return self.real.backfill_targets(limit)

        def backfill_progress(self, address):
            return self.real.backfill_progress(address)

        def mark_backfilled(self, *a, **k):
            return self.real.mark_backfilled(*a, **k)

    report = Backfiller(settings(tmp_path), client=chain).run(ExplodingGraph(graph))
    assert any("observe failed" in err for err in report.errors)


def test_dry_run_fetches_but_records_nothing(tmp_path, graph):
    seed_address(graph, "addrA")
    chain = FakeChain({"addrA": [[make_tx("t1", ["addrA", "addrB"])]]})

    report = Backfiller(settings(tmp_path), client=chain).run(graph, dry_run=True)
    assert report.transactions_seen == 1
    assert graph.backfilled_count() == 0, "dry run must not mark progress"
    assert graph.cluster_size("addrA") == 1, "dry run must not merge"


def test_no_targets_is_a_clean_no_op(tmp_path, graph):
    chain = FakeChain({})
    report = Backfiller(settings(tmp_path), client=chain).run(graph)
    assert report.addresses_processed == 0
    assert chain.requests == []
# AGENTS.md

Repository memory for agents working on Dormant Radar.

## What this is

Detects spends of long-dormant Bitcoin outputs and attributes a probable cause
(lost / holding / structural). Python 3.10+, src layout, FastAPI + SQLite, with
a NumPy-only neural anomaly layer and a common-input-ownership graph.

## Commands

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m pytest                 # 195 tests, no network required
python -m dormant_radar.cli train --blocks 6 --epochs 500
python -m dormant_radar.cli anomaly
python -m dormant_radar.cli scan # live scan of recent blocks
python -m dormant_radar.cli clusters
python -m dormant_radar.cli hunt --dry-run   # preview what would alert
python -m dormant_radar.cli backfill --dry-run
python -m dormant_radar.cli alerts
python -m dormant_radar.cli serve --port 8000
```

## Conventions

- Chain access goes through `MempoolClient` in `chain.py`. Do not call
  `requests` directly from detectors or the scanner.
- `detector.py` is pure: it takes a `Tx`, a spend height, and a threshold, and
  returns `WakeUp`s. Keep network and persistence out of it so the rule stays
  unit-testable.
- Cause attribution lives only in `scorer.py` and must stay inspectable:
  evidence changes log-odds in `PRIOR`, and every evidence step appends a
  human-readable rationale. Never replace it with an opaque model without
  keeping the distribution and rationale output.
- Tests avoid the network entirely. Use `FakeSession` / `FakeChain` (real
  collaborators implementing the same interface); do not add mock libraries.
- Persistence is idempotent: `add_wakeups` uses `INSERT OR IGNORE` on
  `(txid, spent_outpoint)`, so rescans never duplicate.

## Ownership clustering rules (read before touching cluster.py)

- **Error asymmetry drives the design.** A false merge is permanent and
  transitive, silently fusing one owner's behaviour into another's. A missed
  merge only loses information. Every ambiguous case must resolve to "do not
  merge".
- **Never remove the CoinJoin exclusion.** CoinJoins co-spend inputs from
  different owners by design, so naive CIO merges strangers. Detection combines
  a dominant set of equal outputs with a co-spend-like input/output ratio
  (outputs >= inputs). Both signals are required because either alone fires on
  ordinary payments. The thresholds are intentionally loose: missing a CoinJoin
  is the dangerous direction.
- **Cluster ids must stay order-independent.** The root is the lexicographically
  smallest address, so an id depends only on set membership, never processing
  order or restart history. `test_cluster_id_is_order_independent` asserts it.
- **Every non-root address must point directly at the root.** Merges repoint
  the absorbed root and its direct children, keeping the tree one level deep
  and the `size` column correct. Test transitivity after every merge.
- **`observe` must stay idempotent.** It is keyed on txid, so rescans cannot
  inflate a cluster.
- **Do not identify entities.** No labelled data exists; a cluster is an
  anonymous set of co-owned addresses. Say "suggests", never "is".
- **Do not feed cluster size into the neural feature vector.** Coverage grows
  with scanning, so it would shift under the autoencoder. Clustering is
  consumed only by `scorer.py`, with provenance.
- **Report unknown as unknown.** `OwnershipSummary.known` is False until the
  address has been observed. Never treat an unseen address as isolated.

## Alerting rules (read before touching alerting/notify/notifier/redact)

- **Never log a Telegram URL.** The bot token is in the URL *path*, and
  `requests` puts the URL in its exception messages. Always pass exception text
  through `redact()`. `install_redaction()` on the root logger is the backstop,
  not a substitute.
- **Register secrets before the first log line.** `Worker.__init__` calls
  `register_many` before anything else so the filter can scrub from the start.
- **Dedup must stay in SQLite.** Keyed on the spent outpoint in
  `alerted_outpoints`. In-memory dedup would re-alert on every restart.
- **Mark alerted only after a channel accepts.** Otherwise a total delivery
  failure silently loses an event.
- **Rate-limited alerts must stay unmarked and be retried.** Never drop them.
- **Default to silence.** Alert thresholds are strict because a channel that
  fires constantly gets muted, which destroys its value permanently.
- **Do not add an address-watchlist mode.** Watching specific addresses leaks
  the operator's interest set to the API provider on every poll; scanning all
  blocks leaks nothing. This asymmetry is a deliberate design choice.
- **Never describe the channel as private.** Telegram and email are third
  parties. Say so in docs and code comments.

## Digest mode rules (read before touching notifier.py)

- **Never mark alerted at queue time.** Marking happens only after a channel
  accepts the digest. Marking earlier means a crash between queueing and
  sending loses those alerts permanently.
- **Never send an empty digest.** `format_digest` raises on an empty list and
  `_run_digest` returns early when nothing is pending. A "0 alerts" message
  every hour is the noise that gets a channel muted.
- **Never advance the digest window on delivery failure.** The window must
  stay expired so the same alerts are retried rather than skipped for an hour.
- **Never buffer the queue in memory.** Pending alerts are recomputed from
  `alerted_outpoints` each run, so a restart loses nothing.
- **Keep the per-item rate limit out of digest mode.** The window bounds volume;
  running both creates two mechanisms doing one job.
- **Keep truncation honest.** If items exceed `MAX_ITEMS`, the header must still
  report the true total and the body must say how many were omitted.
- **Channel bodies must go through `truncate_for_channel`.** Telegram rejects
  bodies over 4096 chars with a 400, losing the whole message.

## Backfill rules (read before touching backfill.py)

- **Keep it budgeted.** Both `BACKFILL_MAX_ADDRESSES_PER_RUN` and
  `BACKFILL_MAX_PAGES_PER_RUN` must be enforced. An unbounded sweep stalls the
  scan cycle, which makes the whole hunter look hung.
- **Persist the cursor and the completed flag separately.** Restarting a walk
  from the newest transaction each run means a long history never reaches the
  end. `test_long_history_resumes_instead_of_restarting` guards this; do not
  weaken it to merely "mark_backfilled was called".
- **Keep it breadth-first.** Addresses discovered during a walk join the queue
  for a later run; one busy address must not consume the whole budget.
- **Never let backfill raise into a scan.** Chain errors and observation errors
  are recorded in `errors` and skipped.
- **Use `graph.backfill_targets()` and `graph.backfill_progress()`.** Do not
  query the graph's tables directly; the layout is private to cluster.py.
- **Backfill needs a graph.** It is skipped when clustering is disabled or when
  the scan did not produce one.

## Neural layer rules (read before touching features/anomaly/ensemble)

- **The anomaly model is unsupervised and must stay that way.** There are no
  ground-truth labels for cause. Do not add a supervised classifier over
  `lost`/`holding`/`structural`; it would learn invented labels and present
  them as findings. The neural layer answers "how unlike ordinary spending is
  this shape?", nothing more.
- **Never add output age to `FEATURE_NAMES`.** Age is the definition of a
  wake-up, so a model trained on short-lived spends would find every input
  maximally anomalous on age alone, making the neural score a noisy restatement
  of the symbolic rule. `test_features.py::test_age_is_excluded_from_features`
  enforces this.
- **The anomaly scale is calibrated on a held-out split**, never on training
  data. Training error can go near zero by memorising, which yields a tiny
  `error_std` and makes every input a thousands-of-sigma outlier. Observed on a
  real run: train loss 0.0003 vs validation 0.0060.
- **The model's saved metadata includes `feature_count`** and `load` raises if
  it does not match `FEATURE_COUNT`. Changing features requires retraining;
  silently loading a stale model produces garbage or crashes.
- **`MAX_NEURAL_ADJUSTMENT` (2.0 log-odds) is a hard cap** and is applied after
  any scaling, not just to the raw magnitude. The neural opinion must never
  override the symbolic evidence base.
- Neural maths is hand-written in `nn.py`. Any change to forward/backward must
  keep `test_nn.py`'s finite-difference gradient check passing. That check has
  already caught a real bug: the l2 gradient originally divided by batch size
  instead of total element count, scaling gradients by the output width.
- Feature indices are positional. If you reorder `FEATURE_NAMES`, update the
  index constants in `extract_features` and run the full suite.

## Hard-won API facts (verify before changing chain.py)

- `GET /block-height/{h}` returns **plain text** (a bare hash), not JSON.
  `_get_text` exists for this; `_get` would raise.
- In a transaction payload, the spent outpoint is `vin[i].txid` and
  `vin[i].vout`. The spent output's details (`value`, `scriptpubkey_address`,
  `scriptpubkey_type`) are under `vin[i].prevout`. Coinbase inputs have
  `is_coinbase: true` and no useful prevout; skip them.
- `GET /tx/{txid}` is needed for `status.block_height`; results are cached in
  `_height_cache` for the process lifetime because it is the hot path.
- mempool.space rate-limits (HTTP 429) readily. Scan windows must stay bounded
  and `MAX_TXS_PER_BLOCK` respects that.
- CoinGecko's free `/coins/bitcoin/market_chart` returns 429 often. Price
  failures must degrade to `None`, never raise into a scan.
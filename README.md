# Dormant Radar

Detect and score dormant Bitcoin wallet wake-ups from public chain data.

A **wake-up** is a spend of a transaction output that had sat untouched for a
long time. Detection finds those spends. Scoring then asks the harder
question: *why* did the wallet sit idle, and what does moving now suggest?

No API key is required. Chain data comes from the public
[mempool.space](https://mempool.space/docs/api/rest) REST API; market context
comes from [CoinGecko](https://docs.coingecko.com/).

## The central honesty constraint

A dormant wallet is an *observation*, not a reason. On-chain data shows exactly
two things: coins arrived, and they did not leave for a long time. It cannot
show whether the owner lost the keys, chose to hold, or is a custodian.

This project therefore reports a probability distribution over causes rather
than a verdict, and every score ships with the reasoning that produced it.

### Backfilling address history

Clustering only knows what it has scanned, so a fresh database understates
ownership: an address that spent alongside others five years ago looks isolated
today. Backfill walks an address's *historical* transactions, so co-spend
partners are found immediately instead of after days of running.

```bash
export BACKFILL_ENABLED=1
export BACKFILL_MAX_ADDRESSES_PER_RUN=10
export BACKFILL_MAX_PAGES_PER_RUN=20
python -m dormant_radar.cli backfill --dry-run     # fetch, record nothing
python -m dormant_radar.cli backfill               # walk history
```

Measured on live data, a single address with a 2-page budget discovered
**20 addresses in its cluster** and 135 other addresses, in 100 transactions.
That is the difference between an answer now and an answer in a week.

Why it is affordable: `/address/{addr}/txs` embeds the spent output's address
under `vin[].prevout`, so one request yields every co-spend partner with no
per-input blowup. Pages do not overlap.

Why it is budgeted rather than complete:

- **Bounded per run** on both addresses and pages, so it can never stall a scan
  cycle. An unbounded sweep is the thing that makes a hunter look hung.
- **Resumable.** The per-address cursor is stored, so a long history continues
  from where it stopped. Restarting from the newest transaction each run would
  never reach the end — a bounded sweep that never finishes. There is a test
  specifically for that failure.
- **Fair.** Incomplete addresses first (oldest first), then never-walked ones,
  so no address is starved by new arrivals.
- **Breadth-first.** Addresses discovered mid-run are queued for a later run
  rather than walked immediately, so one busy address cannot eat the budget.
- **Non-blocking.** A chain error on one address is recorded and skipped; it
  never aborts the sweep or a scan.

The honest limitation: a heavily-used address is never fully walked at a low
page budget. `txs_seen` records how far it actually got.

## Hunting and alerting

The `worker` runs continuously and raises alerts; `hunt` is a single pass, and
`hunt --dry-run` shows what *would* fire without contacting anyone. Always use
`--dry-run` first.

```bash
python -m dormant_radar.cli alerts               # effective policy and channels
python -m dormant_radar.cli hunt --dry-run       # preview, send nothing
python -m dormant_radar.cli hunt                 # one pass, deliver matches
python -m dormant_radar.cli worker               # continuous hunting
```

Alerts go to Telegram, email, or both. Each trigger is independent and
OR-ed, and every alert states *why* it fired:

| Setting | Default | Meaning |
|---|---|---|
| `ALERTS_ENABLED` | `0` | master switch; off, nothing is sent |
| `ALERT_MIN_VALUE_SATS` | `5000000000` | value floor (50 BTC) |
| `ALERT_MIN_DORMANT_YEARS` | `8` | dormancy floor |
| `ALERT_MIN_ANOMALY_SCORE` | unset | anomaly floor; needs a trained model |
| `ALERT_MIN_CLUSTER_SIZE` | unset | ownership-cluster floor |
| `ALERT_ON_SELF_TRANSFER` | `0` | off, because internal moves are usually noise |
| `ALERT_MAX_PER_HOUR` | `20` | flood guard (immediate mode); excess is held for the next run |
| `ALERT_DIGEST_MODE` | `0` | accumulate and send one summary per window |
| `ALERT_DIGEST_INTERVAL_SECONDS` | `3600` | digest window |
| `ALERT_DIGEST_MAX_ITEMS` | `25` | items shown in a digest before truncation notice |
| `ALERT_INCLUDE_RATIONALE` | `1` | set `0` to send facts without the analysis |

Defaults are strict on purpose: silence is the normal state.

### Immediate or digest delivery

**Immediate** (default) sends one message per alert. **Digest** accumulates
matches and sends a single summary per window — hourly by default. Digest is
what keeps a channel survivable: a busy hour becomes one message, not thirty.

```
Dormant Radar digest: 3 alert(s) in 1.0h
Total value: 1756.00 BTC

1. 1240.0 BTC | 11.2y dormant | holding | cluster 3
   1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
   why: value 1240.00 BTC clears the threshold
...
```

```bash
export ALERT_DIGEST_MODE=1
export ALERT_DIGEST_INTERVAL_SECONDS=3600
export ALERT_DIGEST_MAX_ITEMS=25
```

Design decisions worth knowing, because each guards a specific failure:

- **An empty window sends nothing.** A "0 alerts" message every hour is exactly
  the noise that gets a channel muted. Silence stays the normal state.
- **The queue is derived, not buffered.** Pending alerts are recomputed each run
  from wake-ups that match the policy and are not yet marked alerted, so a
  restart cannot lose a queued digest. An in-memory queue would have.
- **Nothing is marked delivered until a channel accepts.** A crash between
  queueing and sending loses nothing.
- **A failed digest does not advance the window,** so the same alerts are
  retried rather than silently skipped for an hour.
- **Ordered by value, and truncation is disclosed.** Beyond
  `ALERT_DIGEST_MAX_ITEMS` the header still reports the true total and prints
  "... and N more not shown", so the count is never misleading.
- **The per-item rate limit does not apply in digest mode** — the window itself
  bounds volume. Two overlapping limiters would just be confusing.

### What is private here, and what is not

Be clear about the distinction, because "privacy" points two ways.

**Whole-chain scanning is the more private design.** An address watchlist
leaks your interest set on every poll — whoever sees the requests learns which
wallets you care about. Scanning every block reveals nothing, because you look
at everything, exactly as everyone else does. There is deliberately no
watchlist mode.

**The alert channel is not confidential.** Telegram is a third party and its
message history lives on its servers; email traverses relays into a mailbox
usually unencrypted at rest. An alert is therefore a durable record, held by
someone else, of which wallets you find interesting. If the alert *content* is
itself sensitive, the honest answer is a local channel — a file or a private
webhook — not a cloud chat app. `ALERT_INCLUDE_RATIONALE=0` reduces what
leaves the machine but does not make the channel private.

**Credentials are scrubbed from everything that leaves the process.** This is
the one leak that is easy to miss and fatal when it happens: the Telegram bot
token travels in the URL *path*:

```
https://api.telegram.org/bot<TOKEN>/sendMessage
```

`requests` embeds the request URL in its exception messages, so a single logged
retry warning writes a live credential to a log file. `redact.py` installs a
logging filter on the root logger, so no handler — ours or a library's — can
emit one. Verified against a real failing call:

```
DEBUG:urllib3.connectionpool:https://api.telegram.org:443 "POST /bot<redacted>/sendMessage HTTP/1.1" 401
```

That line came from urllib3's internals, not our code. The filter caught it
anyway, which is the point of defence in depth.

### Delivery guarantees

- **Deduplication is persisted, not in-memory.** Keyed on the spent outpoint in
  SQLite, so a worker restart cannot re-send an alert you have already seen.
- **Held-back is not dropped.** An alert suppressed by the rate limit stays
  unmarked and is retried next run.
- **A failed delivery is retried.** An outpoint is marked alerted only after a
  channel accepts it, so total delivery failure loses nothing.
- **Channels are isolated.** One channel raising never blocks the others, and
  alerting never breaks a scan.

## Three layers that answer different questions

**1. The symbolic scorer** (`scorer.py`) decides *what kind of cause* is most
likely, using explicit log-odds evidence: dormancy length, market conditions,
ownership, consolidation patterns. It is the decision, and its reasoning is
printed with every event.

**2. The ownership graph** (`cluster.py`) answers *who controlled the coins*,
using common-input-ownership: if two addresses are spent together as inputs,
one party held both keys. This is what distinguishes a holder moving coins
between their own wallets from a genuine sale.

**3. The neural anomaly model** (`nn.py`, `anomaly.py`) reports *how unlike
ordinary spending* the event's shape is. It is a NumPy-only autoencoder trained
exclusively on ordinary spends — short-lived outputs from everyday
transactions. It never sees a cause label.

`ensemble.py` lets the anomaly score nudge the symbolic distribution, but only
within a hard cap of 2.0 log-odds:

```
rule:  holding 0.594, structural 0.370, lost 0.035
                    <- 27.7 sigma anomaly, capped at 2.0 log-odds ->
final: structural 0.691, holding 0.298, lost 0.011
```

Both distributions, the anomaly score, the features that drove it, and the
size of the nudge are all stored, so the adjustment can be audited or rejected.

### How the ownership graph stays honest

Three deliberate limits, because a wrong merge is worse than a missing one: a
false merge is permanent and **transitive** (it silently fuses one owner's
behaviour into another's cluster), while a missed merge only loses information.
Every ambiguous case therefore resolves to "do not merge".

**CoinJoins are excluded.** A CoinJoin co-spends inputs from *different* owners
by design, so naive common-input-ownership merges strangers into one giant
false cluster. These transactions are detected via a dominant set of equally
sized outputs *plus* a co-spend-like input/output ratio, and skipped. In a live
probe of 600 transactions, zero were excluded — the pattern is rare in random
blocks — so the guard is tested with constructed cases rather than assumed.

**It does not identify entities.** Saying "this cluster is Binance" needs
labelled data we do not have. A cluster is an anonymous set of co-owned
addresses. Large clusters raise the `structural` hypothesis, and the rationale
says "suggests", not "is".

**It does not feed the neural model.** Cluster size depends on how much of the
chain has been scanned, so it would shift under the autoencoder as coverage
grows. It is consumed by the symbolic scorer only, where it stays interpretable
and carries provenance.

**Unknown is reported as unknown.** `OwnershipSummary.known` is False until the
address has actually been seen. When it is False the scorer falls back to the
weaker same-address check *and says so in the rationale*, rather than treating
an unseen address as an isolated one.

Cluster identifiers are the lexicographically smallest address in the cluster,
so an identifier is a function of set membership, never of processing order or
restart history. A test asserts this directly.

## Why the neural layer is unsupervised

There are **no ground-truth labels** for why a given wallet was dormant. Nobody
knows, including the owner's heirs in most cases. A supervised network trained
to output `lost`/`holding`/`structural` would be learning from labels that
someone invented, and would present those assumptions back with false
confidence.

Anomaly detection needs no labels: it asks only "how unlike ordinary spending
is this?", which the chain data can actually answer. That is the defensible use
of a network on this problem, so that is what is implemented.

### Two design decisions worth knowing

**Age is deliberately not a feature.** Age *is* the definition of a wake-up, so
a model trained on short-lived spends would flag every input as maximally
anomalous on age alone — making the neural score a noisy restatement of the
symbolic rule. With age excluded, the rule reasons about *how long* and the
model reasons about *what shape*. A regression test enforces this.

**Calibration uses a held-out split.** An autoencoder can drive its training
error near zero by memorising, which leaves an artificially tiny error spread
and makes everything look like a 2000-sigma outlier. The anomaly scale is
therefore measured on validation data: on a live 433-sample run, training loss
was 0.0003 but validation loss 0.0060, and the calibrated spread was 0.049
rather than 0.0006. Without this the scores were meaningless.

## Layout

```
  chain.py      mempool.space client: retries, prevout height cache
  detector.py   pure aged-output spend rule
  scorer.py     symbolic cause attribution as log-odds evidence
  cluster.py    common-input-ownership graph with CoinJoin exclusion
  features.py   spend-shape feature vector (age excluded by design)
  nn.py         dense network + Adam, hand-written, gradient-checked
  anomaly.py    autoencoder wrapper, robust anomaly scoring
  trainer.py    ordinary-spend collection and training with validation split
  ensemble.py   bounded fusion of symbolic and neural opinions
  alerting.py   alert policy: which wake-ups are worth waking a human for
  notify.py     Telegram and email channels, with the privacy caveats
  notifier.py   dedup, rate limiting, and channel isolation
  redact.py     secret scrubbing for every message that leaves the process
  scanner.py    scan orchestration (block walk -> cluster -> detect -> score)
  store.py      SQLite persistence, scan cursor, alert dedup
  price.py      trailing-average price multiple for market context
  api.py        FastAPI read/scan endpoints
  worker.py     continuous hunter: scan, then alert
  cli.py        command line entry points
```

## Install and run

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m dormant_radar.cli train --blocks 6 --epochs 500   # fit the anomaly model
python -m dormant_radar.cli anomaly                          # inspect what it scores as unusual
python -m dormant_radar.cli scan                             # scan recent blocks once
python -m dormant_radar.cli events --limit 20                # list stored wake-ups
python -m dormant_radar.cli clusters                         # ownership graph stats
python -m dormant_radar.cli clusters --address bc1q...        # look up one address
python -m dormant_radar.cli stats                            # aggregates
python -m dormant_radar.cli serve --port 8000                # HTTP API
python -m dormant_radar.cli worker                           # poll continuously
```

The neural layer is optional. With no trained model present, scans run on the
symbolic scorer alone and say so; train one to enable the hybrid.

Run the tests with `python -m pytest`. The suite does not touch the network.

## HTTP endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/events` | stored wake-ups; filters `limit`, `min_value_sats`, `since_height`, `hypothesis` |
| GET | `/stats` | totals, value, per-hypothesis counts, scan cursor |
| POST | `/scan` | run one scan synchronously |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DORMANT_RADAR_DB` | `data/dormant_radar.db` | SQLite path |
| `DORMANT_RADAR_MODEL` | `data/anomaly_model.npz` | trained anomaly model |
| `DORMANT_RADAR_CLUSTERS` | `data/clusters.db` | ownership graph database |
| `USE_NEURAL` | `1` | set to `0` to force symbolic-only scoring |
| `USE_CLUSTERING` | `1` | set to `0` to disable ownership inference |
| `BACKFILL_ENABLED` | `0` | walk address history to sharpen ownership |
| `BACKFILL_MAX_ADDRESSES_PER_RUN` | `10` | addresses walked per cycle |
| `BACKFILL_MAX_PAGES_PER_RUN` | `20` | page budget per cycle |
| `DORMANT_AFTER_BLOCKS` | `210240` | age threshold (~4 years) to count as dormant |
| `SCAN_WINDOW_BLOCKS` | `6` | blocks examined per scan |
| `MIN_SPENT_SATS` | `100000000` | value floor (1 BTC) to ignore dust |
| `MAX_TXS_PER_BLOCK` | `400` | per-block work cap |
| `TELEGRAM_BOT_TOKEN` | unset | Telegram bot credential |
| `TELEGRAM_CHAT_ID` | unset | one id, or several comma-separated |
| `SMTP_HOST` / `SMTP_PORT` | unset / `587` | email delivery |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | unset | email credentials |
| `ALERT_EMAIL_FROM` / `ALERT_EMAIL_TO` | unset | email sender and recipients |
| `POLL_INTERVAL_SECONDS` | `120` | worker cadence |
| `MAX_RETRIES` | `3` | HTTP retries per request |

## Known limitations

- **Cause is inferred, never observed.** Confidence is routinely low; that is
  the correct output when evidence is thin, not a defect.
- **The anomaly score orders, it does not calibrate.** It reliably separates
  ordinary from unusual shapes, but a score of 30 is not a probability and its
  magnitude depends on the training run.
- **The neural adjustment is a bounded heuristic.** Its direction (unusual
  shape favours structural causes) is a modelling assumption, capped at 2.0
  log-odds precisely because it is not proven.
- **Small training sets.** A few hundred samples from a handful of blocks is
  enough for shape separation, not for subtle structure. Retrain over more
  blocks for better coverage.
- **Public-endpoint rate limits.** mempool.space returns HTTP 429 under load.
  The client backs off, but wide scans or training runs should use a local node
  or a keyed provider. CoinGecko's free endpoint rate-limits aggressively; a
  failed lookup degrades to chain-only evidence rather than breaking a scan.
- **Clustering coverage is cumulative unless backfill is on.** Without it, a
  cluster is only as large as the history scanned so far. Backfill fixes the
  cold start but is itself budgeted, so a heavily-used address may be only
  partly walked. `txs_seen` records how far it got.
- **Backfill trusts the address page cursor.** If mempool.space ever returned
  an inconsistent page mid-walk, the stored cursor could skip transactions.
  Pages were verified non-overlapping on live data, but the walk has no
  checksum to detect it.
- **CoinJoin exclusion is heuristic.** It catches the equal-output pattern
  (Whirlpool, JoinMarket); exotic collaborative constructions may slip through
  and over-merge. Treat very large clusters as a hint, not proof.
- **Rate limit is a floor, not a queue.** In immediate mode, excess alerts stay
  pending and arrive on the next run rather than being dropped — but they arrive
  late, so a genuine burst is better served by digest mode.
- **Digest truncation hides detail, not occurrence.** Beyond
  `ALERT_DIGEST_MAX_ITEMS` you are told how many were omitted but not what they
  were. Raise the cap if you need every line.
- **Digest timing is per-process.** Two workers sharing a database would both
  see the same window and could each try to send. Run one worker per database.
- **Bitcoin only** at present.
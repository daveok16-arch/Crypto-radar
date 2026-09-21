"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Settings
from .scanner import scan_once
from .store import Store


def _cmd_scan(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.window is not None:
        settings = Settings(**{**settings.__dict__, "scan_window_blocks": args.window})
    result = scan_once(settings)
    print(
        json.dumps(
            {
                "scanned_from": result.scanned_from,
                "scanned_to": result.scanned_to,
                "blocks_examined": result.blocks_examined,
                "transactions_examined": result.transactions_examined,
                "wakeups_found": len(result.wakeups),
                "new_wakeups": result.new_wakeups,
                "errors": result.errors[:20],
            },
            indent=2,
        )
    )
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    try:
        events = store.list_wakeups(
            limit=args.limit,
            min_value_sats=args.min_value,
            hypothesis=args.hypothesis,
        )
    finally:
        store.close()
    print(json.dumps({"count": len(events), "events": events}, indent=2))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    store = Store(settings.db_path)
    try:
        print(json.dumps(store.stats(), indent=2))
    finally:
        store.close()
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .trainer import collect_ordinary_spends, train_autoencoder

    settings = Settings.from_env()
    if args.blocks is not None:
        settings = Settings(**{**settings.__dict__, "scan_window_blocks": args.blocks})
    print(f"collecting ordinary spends from {args.blocks} recent blocks...")
    training = collect_ordinary_spends(settings, blocks=args.blocks)
    print(f"collected {len(training)} samples; training {args.epochs} epochs...")
    report = train_autoencoder(
        training,
        model_path=settings.model_path,
        epochs=args.epochs,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "samples": report.samples,
                "train_samples": report.train_samples,
                "validation_samples": report.validation_samples,
                "epochs": report.epochs,
                "train_loss": round(report.final_loss, 6),
                "validation_loss": round(report.validation_loss, 6),
                "loss_reduction_pct": round(
                    100 * (1 - report.final_loss / report.initial_loss), 2
                )
                if report.initial_loss
                else 0.0,
                "calibrated_error_mean": round(report.error_mean, 6),
                "calibrated_error_std": round(report.error_std, 6),
                "model_path": report.model_path,
            },
            indent=2,
        )
    )
    return 0


def _cmd_anomaly(args: argparse.Namespace) -> int:
    from .anomaly import AnomalyDetector
    from .features import FeatureContext, extract_features
    from .models import Outpoint, Tx, TxInput, WakeUp
    import time

    settings = Settings.from_env()
    detector = AnomalyDetector.load(settings.model_path)
    if detector is None:
        print(f"no model at {settings.model_path}; run `dormant-radar train` first")
        return 1

    # Score a few illustrative shapes so the model's behaviour is visible.
    cases = [
        ("plain holder sale", 4 * 52_560, 500_000_000, 1),
        ("very old, tiny", 14 * 52_560, 2_000_000, 1),
        ("large sweep", 6 * 52_560, 900_000_000, 40),
    ]
    out = []
    for label, age, value, n_in in cases:
        wakeup = WakeUp(
            txid="ab" * 32,
            spend_block_height=900_000,
            spent=Outpoint("cd" * 32, 0),
            value_sats=value,
            address="bc1qexample",
            dormant_blocks=age,
            dormant_years=age / 52_560,
            script_type="v0_p2wpkh",
            observed_at=time.time(),
        )
        tx = Tx(
            txid="ef" * 32,
            block_height=900_000,
            inputs=[TxInput(Outpoint(f"{i:02x}" * 32, 0), value, "bc1qx", 800_000, "p2wpkh")
                    for i in range(n_in)],
            output_values_sats=[value],
            output_addresses=["bc1qdst"],
        )
        score = detector.score(wakeup, tx=tx, context=FeatureContext())
        out.append({"case": label, **score.as_dict()})
    print(json.dumps(out, indent=2))
    return 0


def _cmd_clusters(args: argparse.Namespace) -> int:
    from .cluster import OwnershipGraph

    settings = Settings.from_env()
    graph = OwnershipGraph(settings.cluster_path)
    try:
        stats = graph.stats()
        payload: dict = {"stats": stats}
        if args.address:
            cluster_id = graph.cluster_id(args.address)
            payload["address"] = args.address
            payload["cluster_id"] = cluster_id
            payload["cluster_size"] = graph.cluster_size(args.address) if cluster_id else 0
        print(json.dumps(payload, indent=2))
    finally:
        graph.close()
    return 0


def _cmd_hunt(args: argparse.Namespace) -> int:
    """One hunting pass: scan, then alert on anything that clears the policy."""
    from .notifier import Notifier
    from .notify import build_channels
    from .redact import install_redaction, register_many
    from .scanner import scan_once
    from .store import Store

    install_redaction()
    settings = Settings.from_env()
    register_many(
        [
            settings.telegram_bot_token,
            settings.smtp_password,
            *settings.telegram_chat_ids,
        ]
    )

    store = Store(settings.db_path)
    try:
        scan_result = scan_once(settings, store=store)

        channels = build_channels(settings)
        notifier = settings.build_notifier(store, channels, force_disabled=args.dry_run)

        alert_result = _run_alerts(notifier, channels, store, settings, dry_run=args.dry_run)

        print(
            json.dumps(
                {
                    "scan": {
                        "blocks_examined": scan_result.blocks_examined,
                        "transactions_examined": scan_result.transactions_examined,
                        "wakeups_found": len(scan_result.wakeups),
                        "new_wakeups": scan_result.new_wakeups,
                        "errors": scan_result.errors[:5],
                    },
                    "alerts": alert_result,
                    "digest_mode": settings.alert_digest_mode,
                    "dry_run": args.dry_run,
                },
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


def _run_alerts(notifier, channels, store, settings, dry_run: bool) -> dict:
    """Preview what would alert, or actually send."""
    from .alerting import evaluate

    if not dry_run:
        result = notifier.run()
        payload = {
            "matched": result.matched,
            "delivered": result.delivered,
            "suppressed_duplicates": result.suppressed_duplicates,
            "suppressed_ratelimit": result.suppressed_ratelimit,
            "channel_failures": result.channel_failures,
            "channels": [c.name for c in channels],
        }
        if notifier.digest_mode:
            payload["digest_sent"] = result.digest_sent
            payload["digest_pending"] = result.digest_pending
            payload["digest_seconds_until_next"] = round(
                result.digest_seconds_until_next, 1
            )
        return payload

    # Dry run: evaluate against the policy without contacting any channel.
    policy = settings.alert_policy()
    matched = []
    for wakeup in store.list_wakeups(limit=200):
        alert = evaluate(wakeup, policy)
        if alert is not None:
            matched.append(
                {
                    "txid": alert.txid,
                    "value_btc": wakeup.get("value_btc"),
                    "reasons": alert.reasons,
                    "severity": alert.severity,
                    "already_alerted": store.was_alerted(alert.dedupe_key),
                }
            )
    payload = {
        "matched": len(matched),
        "delivered": 0,
        "channels": [c.name for c in channels],
        "would_alert": matched,
        "note": "dry run: nothing was sent",
    }
    if settings.alert_digest_mode:
        payload["digest_mode"] = True
        payload["digest_would_send"] = len(matched) > 0
        payload["digest_note"] = (
            "in digest mode these would be summarised into one message "
            "when the window elapses, not sent individually"
        )
    return payload


def _cmd_alerts(args: argparse.Namespace) -> int:
    """Show the effective alert policy and delivery history."""
    from .notify import build_channels

    settings = Settings.from_env()
    store = Store(settings.db_path)
    try:
        policy = settings.alert_policy()
        channels = build_channels(settings)
        print(
            json.dumps(
                {
                    "enabled": settings.alerts_enabled,
                    "channels": [channel.name for channel in channels],
                    "policy": {
                        "min_value_sats": policy.min_value_sats,
                        "min_dormant_years": policy.min_dormant_years,
                        "min_anomaly_score": policy.min_anomaly_score,
                        "min_cluster_size": policy.min_cluster_size,
                        "alert_on_self_transfer": policy.alert_on_self_transfer,
                        "max_alerts_per_hour": policy.max_alerts_per_hour,
                    },
                    "delivery": {
                        "digest_mode": settings.alert_digest_mode,
                        "digest_interval_seconds": settings.alert_digest_interval_seconds,
                        "digest_max_items": settings.alert_digest_max_items,
                        "last_digest_at": store.get_state("last_digest_at"),
                        "include_rationale": settings.alert_include_rationale,
                    },
                    "already_alerted": store.alerted_count(),
                },
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


def _cmd_backfill(args: argparse.Namespace) -> int:
    """Walk address histories to sharpen the ownership graph."""
    from .backfill import Backfiller
    from .cluster import OwnershipGraph

    settings = Settings.from_env()
    if args.addresses is not None:
        settings = Settings(
            **{
                **settings.__dict__,
                "backfill_max_addresses_per_run": args.addresses,
            }
        )
    if args.pages is not None:
        settings = Settings(
            **{**settings.__dict__, "backfill_max_pages_per_run": args.pages}
        )

    graph = OwnershipGraph(settings.cluster_path)
    try:
        before = graph.stats()
        report = Backfiller(settings).run(graph, dry_run=args.dry_run)
        after = graph.stats()
        print(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "addresses_considered": report.addresses_considered,
                    "addresses_processed": report.addresses_processed,
                    "addresses_completed": report.addresses_completed,
                    "pages_fetched": report.pages_fetched,
                    "transactions_seen": report.transactions_seen,
                    "errors": report.errors[:5],
                    "graph_before": before,
                    "graph_after": after,
                },
                indent=2,
            )
        )
    finally:
        graph.close()
    return 0


def _cmd_cloud_run(args: argparse.Namespace) -> int:
    """One stateless run: scan recent blocks, alert on matches, persist state.

    Designed for a scheduled cloud runner where the filesystem is not durable.
    Dedup state is loaded from and saved to the portable backend (the
    automation KV store when configured), because an empty SQLite file every
    run would re-alert the same events indefinitely.
    """
    from .notifier import Notifier
    from .notify import build_channels
    from .redact import install_redaction, register_many
    from .scanner import scan_once
    from .state import open_state
    from .store import Store

    install_redaction()
    settings = Settings.from_env()
    register_many(
        [
            settings.telegram_bot_token,
            settings.smtp_password,
            *settings.telegram_chat_ids,
        ]
    )

    # Local databases are scratch space on a cloud pod; the durable state
    # travels through the backend instead.
    state = open_state(local_path=args.state_path)
    store = Store(settings.db_path)
    try:
        loaded = state.load()
        if loaded.last_scanned_height is not None:
            # Seed the cursor so this run resumes rather than rescanning.
            store.set_state("last_scanned_height", loaded.last_scanned_height)

        scan_result = scan_once(settings, store=store)

        channels = build_channels(settings)
        notifier = settings.build_notifier(store, channels)
        notifier.install_state(state)
        run = notifier.run()

        # Persist the cursor and the dedup set, so the next run neither
        # rescans nor re-alerts.
        notifier.snapshot_state(scanned_to=scan_result.scanned_to)

        print(
            json.dumps(
                {
                    "blocks": [scan_result.scanned_from, scan_result.scanned_to],
                    "transactions_examined": scan_result.transactions_examined,
                    "wakeups_found": len(scan_result.wakeups),
                    "new_wakeups": scan_result.new_wakeups,
                    "alerts_matched": run.matched,
                    "alerts_delivered": run.delivered,
                    "duplicates_suppressed": run.suppressed_duplicates,
                    "digest_mode": settings.alert_digest_mode,
                    "state_backend": state.name,
                    "tracked_outpoints": len(notifier.alerted_outpoints),
                    "errors": scan_result.errors[:3],
                },
                indent=2,
            )
        )
    finally:
        store.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "dormant_radar.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    from .worker import run_worker

    run_worker()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="dormant-radar",
        description="Detect and score dormant Bitcoin wallet wake-ups.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan recent blocks once")
    p_scan.add_argument("--window", type=int, default=None, help="blocks to scan")
    p_scan.set_defaults(func=_cmd_scan)

    p_events = sub.add_parser("events", help="list stored wake-ups")
    p_events.add_argument("--limit", type=int, default=20)
    p_events.add_argument("--min-value", type=int, default=0, help="minimum sats")
    p_events.add_argument(
        "--hypothesis", choices=["lost", "holding", "structural"], default=None
    )
    p_events.set_defaults(func=_cmd_events)

    p_stats = sub.add_parser("stats", help="show aggregate stats")
    p_stats.set_defaults(func=_cmd_stats)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_worker = sub.add_parser("worker", help="run the background poller")
    p_worker.set_defaults(func=_cmd_worker)

    p_train = sub.add_parser("train", help="train the anomaly model on ordinary spends")
    p_train.add_argument("--blocks", type=int, default=20, help="recent blocks to sample")
    p_train.add_argument("--epochs", type=int, default=400)
    p_train.set_defaults(func=_cmd_train)

    p_anom = sub.add_parser("anomaly", help="score illustrative spend shapes")
    p_anom.set_defaults(func=_cmd_anomaly)

    p_clusters = sub.add_parser("clusters", help="inspect the ownership graph")
    p_clusters.add_argument("--address", default=None, help="look up one address")
    p_clusters.set_defaults(func=_cmd_clusters)

    p_hunt = sub.add_parser("hunt", help="scan once and alert on matches")
    p_hunt.add_argument(
        "--dry-run",
        action="store_true",
        help="evaluate the policy and show matches without sending anything",
    )
    p_hunt.set_defaults(func=_cmd_hunt)

    p_alerts = sub.add_parser("alerts", help="show alert policy and channel status")
    p_alerts.set_defaults(func=_cmd_alerts)

    p_backfill = sub.add_parser(
        "backfill", help="walk address histories to sharpen ownership"
    )
    p_backfill.add_argument("--addresses", type=int, default=None)
    p_backfill.add_argument("--pages", type=int, default=None)
    p_backfill.add_argument(
        "--dry-run", action="store_true", help="fetch but do not record anything"
    )
    p_backfill.set_defaults(func=_cmd_backfill)

    p_cloud = sub.add_parser(
        "cloud-run",
        help="one stateless run for a scheduled cloud runner",
    )
    p_cloud.add_argument(
        "--state-path",
        default=None,
        help="local state file when no KV store is configured",
    )
    p_cloud.set_defaults(func=_cmd_cloud_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
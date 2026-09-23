"""Runtime configuration, read from the environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

MEMPOOL_BASE_URL = "https://mempool.space/api"


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _optional_int(name: str, default: int | None = None) -> int | None:
    """Read an optional int from the environment.

    An absent or blank variable yields `default`, NOT None. Conflating those was
    a real bug: `from_env()` passed None for every unset alert threshold, which
    silently disabled alerting entirely — a wake-up would be detected and then
    match nothing. "Unset" must mean "use the default"; disabling a trigger is
    expressed explicitly (see `_disabled_int` below).
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in ("none", "off", "disabled", ""):
        return None
    return int(raw)


def _optional_float(name: str, default: float | None = None) -> float | None:
    """Read an optional float from the environment; absent yields `default`."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in ("none", "off", "disabled"):
        return None
    return float(raw)


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in ("0", "false", "False", "no")


def _list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Tunables for the radar.

    `dormant_after_blocks` is the age threshold that separates an ordinary
    spend from a "dormant wake-up". Blocks are approximate time: a year is
    ~52_560 blocks at the 10-minute target.
    """

    base_url: str = MEMPOOL_BASE_URL
    db_path: str = "data/dormant_radar.db"
    dormant_after_blocks: int = 210_240  # roughly four years
    scan_window_blocks: int = 6
    poll_interval_seconds: float = 120.0
    request_timeout_seconds: float = 20.0
    max_retries: int = 3
    min_spent_sats: int = 100_000_000  # 1 BTC; filters dust-scale noise
    max_txs_per_block: int = 400
    model_path: str = "data/anomaly_model.npz"
    cluster_path: str = "data/clusters.db"
    use_clustering: bool = True
    use_neural: bool = True

    # -- backfill ------------------------------------------------------
    # Walking address history sharpens ownership immediately instead of waiting
    # for days of scanning. Budgeted so it can never stall a scan cycle.
    backfill_enabled: bool = False
    backfill_max_addresses_per_run: int = 10
    backfill_max_pages_per_run: int = 20

    # -- alerting ------------------------------------------------------
    alerts_enabled: bool = False
    alert_min_value_sats: int | None = 5_000_000_000
    alert_min_dormant_years: float | None = 8.0
    alert_min_anomaly_score: float | None = None
    alert_min_cluster_size: int | None = None
    alert_on_self_transfer: bool = False
    alert_max_per_hour: int = 20
    # Digest mode: accumulate matching wake-ups and send one summary per window
    # instead of one message each. This is what keeps a channel readable.
    alert_digest_mode: bool = False
    alert_digest_interval_seconds: float = 3600.0
    alert_digest_max_items: int = 25
    # When false, alerts carry only the facts, not the reasoning chain. Useful
    # when the channel is a third party you would rather not share analysis with.
    alert_include_rationale: bool = True

    # -- channels ------------------------------------------------------
    telegram_bot_token: str | None = None
    telegram_chat_ids: tuple[str, ...] = ()
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_starttls: bool = True
    alert_email_from: str | None = None
    alert_email_to: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        chat_ids = tuple(_list("TELEGRAM_CHAT_ID") or _list("TELEGRAM_CHAT_IDS") or [])
        return cls(
            base_url=os.environ.get("DORMANT_RADAR_BASE_URL", MEMPOOL_BASE_URL),
            db_path=os.environ.get("DORMANT_RADAR_DB", "data/dormant_radar.db"),
            dormant_after_blocks=_int("DORMANT_AFTER_BLOCKS", 210_240),
            scan_window_blocks=_int("SCAN_WINDOW_BLOCKS", 6),
            poll_interval_seconds=_float("POLL_INTERVAL_SECONDS", 120.0),
            request_timeout_seconds=_float("REQUEST_TIMEOUT_SECONDS", 20.0),
            max_retries=_int("MAX_RETRIES", 3),
            min_spent_sats=_int("MIN_SPENT_SATS", 100_000_000),
            max_txs_per_block=_int("MAX_TXS_PER_BLOCK", 400),
            model_path=os.environ.get("DORMANT_RADAR_MODEL", "data/anomaly_model.npz"),
            cluster_path=os.environ.get("DORMANT_RADAR_CLUSTERS", "data/clusters.db"),
            use_clustering=_bool("USE_CLUSTERING", True),
            use_neural=_bool("USE_NEURAL", True),
            backfill_enabled=_bool("BACKFILL_ENABLED", False),
            backfill_max_addresses_per_run=_int("BACKFILL_MAX_ADDRESSES_PER_RUN", 10),
            backfill_max_pages_per_run=_int("BACKFILL_MAX_PAGES_PER_RUN", 20),
            alerts_enabled=_bool("ALERTS_ENABLED", False),
            # Defaults are passed explicitly so an unset variable means "use the
            # default" rather than "disable this trigger". Passing nothing here
            # was a bug that made alerting silently match nothing.
            alert_min_value_sats=_optional_int("ALERT_MIN_VALUE_SATS", 5_000_000_000),
            alert_min_dormant_years=_optional_float("ALERT_MIN_DORMANT_YEARS", 8.0),
            alert_min_anomaly_score=_optional_float("ALERT_MIN_ANOMALY_SCORE", None),
            alert_min_cluster_size=_optional_int("ALERT_MIN_CLUSTER_SIZE", None),
            alert_on_self_transfer=_bool("ALERT_ON_SELF_TRANSFER", False),
            alert_max_per_hour=_int("ALERT_MAX_PER_HOUR", 20),
            alert_digest_mode=_bool("ALERT_DIGEST_MODE", False),
            alert_digest_interval_seconds=_float("ALERT_DIGEST_INTERVAL_SECONDS", 3600.0),
            alert_digest_max_items=_int("ALERT_DIGEST_MAX_ITEMS", 25),
            alert_include_rationale=_bool("ALERT_INCLUDE_RATIONALE", True),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_ids=chat_ids,
            smtp_host=os.environ.get("SMTP_HOST") or None,
            smtp_port=_int("SMTP_PORT", 587),
            smtp_username=os.environ.get("SMTP_USERNAME") or None,
            smtp_password=os.environ.get("SMTP_PASSWORD") or None,
            smtp_starttls=_bool("SMTP_STARTTLS", True),
            alert_email_from=os.environ.get("ALERT_EMAIL_FROM") or None,
            alert_email_to=tuple(_list("ALERT_EMAIL_TO")),
        )

    def alert_policy(self):
        """Build the alert policy from these settings."""
        from .alerting import AlertPolicy

        return AlertPolicy(
            min_value_sats=self.alert_min_value_sats,
            min_dormant_years=self.alert_min_dormant_years,
            min_anomaly_score=self.alert_min_anomaly_score,
            min_cluster_size=self.alert_min_cluster_size,
            alert_on_self_transfer=self.alert_on_self_transfer,
            max_alerts_per_hour=self.alert_max_per_hour,
        )

    def build_notifier(self, store, channels, force_disabled: bool = False):
        """Construct a Notifier wired to these settings.

        A single construction point so the worker and the CLI cannot drift
        apart on digest mode or policy.
        """
        from .notifier import Notifier

        return Notifier(
            store=store,
            policy=self.alert_policy(),
            channels=channels,
            enabled=self.alerts_enabled and not force_disabled,
            digest_mode=self.alert_digest_mode,
            digest_interval_seconds=self.alert_digest_interval_seconds,
            digest_max_items=self.alert_digest_max_items,
        )
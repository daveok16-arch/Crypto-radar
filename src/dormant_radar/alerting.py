"""Alert policy: deciding which wake-ups are worth a human's attention.

A hunter that reports everything is a hunter nobody reads. The rule here is
deliberately simple and inspectable — a set of independent triggers, each with
a stated reason — rather than a single opaque score.

Every trigger is optional and configurable. The defaults are intentionally
strict: silence is the normal state, and an alert should mean something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import WakeUp


@dataclass(frozen=True)
class AlertPolicy:
    """Thresholds for escalating a wake-up to a notification.

    All thresholds are floors: a wake-up alerts if it clears *any* one of
    them (they are OR-ed), because each captures a different kind of
    significance. Setting a threshold to None disables that trigger.
    """

    min_value_sats: Optional[int] = 5_000_000_000        # 50 BTC
    min_dormant_years: Optional[float] = 8.0
    min_anomaly_score: Optional[float] = None            # off by default
    min_cluster_size: Optional[int] = None               # off by default
    alert_on_self_transfer: bool = False                 # internal moves, usually noise
    max_alerts_per_hour: int = 20

    @property
    def single_trigger(self) -> bool:
        """True when exactly one kind of event can fire, useful for sanity checks."""
        return sum(
            x is not None
            for x in (
                self.min_value_sats,
                self.min_dormant_years,
                self.min_anomaly_score,
                self.min_cluster_size,
            )
        ) + int(self.alert_on_self_transfer) == 1


@dataclass
class Alert:
    """A wake-up plus the human-readable reasons it was escalated."""

    wakeup: dict
    reasons: list[str] = field(default_factory=list)
    severity: str = "notice"

    @property
    def txid(self) -> str:
        return self.wakeup.get("txid", "")

    @property
    def dedupe_key(self) -> str:
        """Stable identity so a restart cannot re-send the same alert.

        Keyed on the spent outpoint rather than the txid: a transaction can
        contain several dormant inputs of different interest, and each is a
        distinct event.
        """
        return self.wakeup.get("spent_outpoint") or self.txid


def _anomaly_score(wakeup: dict) -> Optional[float]:
    neural = wakeup.get("neural") or {}
    anomaly = neural.get("anomaly") or {}
    score = anomaly.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _cluster_size(wakeup: dict) -> Optional[int]:
    ownership = wakeup.get("ownership") or {}
    if not ownership.get("known"):
        return None
    size = ownership.get("cluster_size")
    return int(size) if isinstance(size, int) else None


def _is_self_transfer(wakeup: dict) -> bool:
    ownership = wakeup.get("ownership") or {}
    if ownership.get("known"):
        return bool(ownership.get("is_self_transfer"))
    # Fall back to the rule-level rationale when clustering is unavailable.
    rationale = " ".join((wakeup.get("cause") or {}).get("rationale", []))
    return "returns to the originating address" in rationale


def evaluate(wakeup: dict, policy: AlertPolicy) -> Optional[Alert]:
    """Decide whether a stored wake-up warrants an alert.

    Returns None when it does not, so callers can simply skip.
    """
    reasons: list[str] = []
    severity = "notice"

    value = wakeup.get("value_sats", 0)
    years = wakeup.get("dormant_years", 0.0)

    if policy.min_value_sats is not None and value >= policy.min_value_sats:
        btc = value / 100_000_000
        reasons.append(f"value {btc:.2f} BTC clears the threshold")
        severity = "high"

    if (
        policy.min_dormant_years is not None
        and years >= policy.min_dormant_years
    ):
        reasons.append(f"dormant {years:.1f} years, at or beyond the threshold")
        severity = "high"

    anomaly = _anomaly_score(wakeup)
    if (
        policy.min_anomaly_score is not None
        and anomaly is not None
        and anomaly >= policy.min_anomaly_score
    ):
        reasons.append(f"anomaly score {anomaly:.1f} clears the threshold")

    cluster = _cluster_size(wakeup)
    if (
        policy.min_cluster_size is not None
        and cluster is not None
        and cluster >= policy.min_cluster_size
    ):
        reasons.append(f"ownership cluster of {cluster} addresses clears the threshold")

    if policy.alert_on_self_transfer and _is_self_transfer(wakeup):
        reasons.append("value returns to the same ownership cluster (internal move)")
        severity = "low"

    if not reasons:
        return None

    # An internal move is the least interesting thing this can report; if it
    # only fired for that reason, say so plainly rather than as a headline.
    if len(reasons) == 1 and reasons[0].startswith("value returns"):
        severity = "low"

    return Alert(wakeup=wakeup, reasons=reasons, severity=severity)
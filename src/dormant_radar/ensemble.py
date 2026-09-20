"""Hybrid decision layer: symbolic evidence plus a neural anomaly opinion.

The two halves answer different questions and are kept in separate fields:

- The symbolic scorer says *what kind of cause* is most likely and shows its
  reasoning. It is the decision.
- The anomaly model says *how unlike ordinary spending* the event is. It has
  no opinion about cause, because no labelled cause data exists.

The anomaly score is allowed to nudge the symbolic distribution, but only
within a bounded number of log-odds, and only using a signal that is defensible:
an unusually *shape-different* spend should make a simple "holder selling into
strength" reading less certain, redistributing some weight toward structural
causes. The cap means the neural opinion can never overturn the evidence base,
and both raw outputs stay visible so the adjustment is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .anomaly import AnomalyScore, AnomalyDetector
from .cluster import OwnershipSummary
from .features import FeatureContext
from .models import CauseScore, Tx, WakeUp
from .scorer import score_cause

# The single tunable that keeps the hybrid honest: how much of a nudge the
# anomaly signal is permitted. Two log-odds is a strong but not decisive push.
MAX_NEURAL_ADJUSTMENT = 2.0

# Anomaly scores above this are treated as genuinely unusual; below it the
# signal is noise-level and is ignored entirely.
ANOMALY_FLOOR = 1.0

# A spend that is anomalous in *shape* (many inputs, large value, wide fan-out)
# is more structural than a simple holder sale. Age is not in the feature set
# at all, so every name here is genuinely about transaction shape.
SHAPE_FEATURES = {
    "log_value_sats",
    "n_inputs_log",
    "n_outputs_log",
    "input_share",
    "output_fanout_ratio",
    "fee_rate_ratio",
    "value_per_input_log",
}


@dataclass
class HybridScore:
    """Combined output, with every input to the decision preserved."""

    cause: CauseScore
    anomaly: AnomalyScore | None
    rule_distribution: dict[str, float]
    final_distribution: dict[str, float]
    adjustment_log_odds: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Hybrid-specific fields only.

        The chosen cause lives in `WakeUp.cause`; duplicating its hypothesis
        and rationale here would invite the two copies to drift apart. What
        this block adds is the comparison: the rule's distribution, the final
        distribution after the neural nudge, and the anomaly evidence itself.
        """
        return {
            "rule_distribution": {k: round(v, 3) for k, v in self.rule_distribution.items()},
            "final_distribution": {k: round(v, 3) for k, v in self.final_distribution.items()},
            "anomaly": self.anomaly.as_dict() if self.anomaly else None,
            "neural_adjustment_log_odds": round(self.adjustment_log_odds, 3),
            "notes": self.notes,
        }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / total for k, v in scores.items()}


def combine(
    wakeup: WakeUp,
    tx: Tx | None = None,
    price_multiple: float | None = None,
    network_upgrade_window: bool = False,
    detector: AnomalyDetector | None = None,
    ownership: OwnershipSummary | None = None,
) -> HybridScore:
    """Produce the final hybrid score for one wake-up."""
    cause = score_cause(
        wakeup,
        tx=tx,
        price_multiple=price_multiple,
        network_upgrade_window=network_upgrade_window,
        ownership=ownership,
    )
    rule_distribution = dict(cause.distribution)

    if detector is None:
        return HybridScore(
            cause=cause,
            anomaly=None,
            rule_distribution=rule_distribution,
            final_distribution=rule_distribution,
            adjustment_log_odds=0.0,
            notes=["No trained anomaly model available; rule-based score only."],
        )

    anomaly = detector.score(wakeup, tx=tx, context=FeatureContext(price_multiple=price_multiple))

    # Convert the rule probabilities to log-odds so evidence can be added.
    log_odds = {k: _logit(v) for k, v in rule_distribution.items()}
    notes: list[str] = []
    adjustment = 0.0

    if anomaly.score > ANOMALY_FLOOR:
        # How much of the shape is unusual, not just the aggregate error.
        shape_weight = sum(
            contribution
            for name, contribution in anomaly.top_features
            if name in SHAPE_FEATURES
        )
        # Magnitude grows with the anomaly score past the floor, saturating
        # at the cap so no single signal can dominate.
        magnitude = min(anomaly.score - ANOMALY_FLOOR, MAX_NEURAL_ADJUSTMENT)
        if shape_weight > 0:
            adjustment = magnitude
            note = (
                f"Spend shape is unusual (anomaly score {anomaly.score:.2f}); "
                f"weight shifts toward structural causes by {adjustment:.2f} log-odds."
            )
        else:
            adjustment = magnitude * 0.4
            note = (
                f"Anomaly score {anomaly.score:.2f} with no dominant shape signal; "
                "a small shift is applied."
            )
        # The cap is enforced after scaling, not just on the magnitude, so no
        # combination of signals can exceed the documented bound.
        adjustment = max(-MAX_NEURAL_ADJUSTMENT, min(MAX_NEURAL_ADJUSTMENT, adjustment))
        log_odds["structural"] += adjustment
        log_odds["holding"] -= adjustment * 0.5
        log_odds["lost"] -= adjustment * 0.5
        notes.append(note)
    else:
        notes.append(
            f"Anomaly score {anomaly.score:.2f} is within the ordinary range; "
            "the rule-based score is left unchanged."
        )

    final = _normalise({k: _sigmoid(v) for k, v in log_odds.items()})
    best = max(final, key=final.get)
    cause.hypothesis = best
    ranked = sorted(final.values(), reverse=True)
    cause.confidence = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]

    return HybridScore(
        cause=cause,
        anomaly=anomaly,
        rule_distribution=rule_distribution,
        final_distribution=final,
        adjustment_log_odds=adjustment,
        notes=notes,
    )
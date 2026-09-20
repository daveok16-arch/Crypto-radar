"""Feature extraction for the anomaly model.

Every wake-up becomes a fixed-length numeric vector describing the *shape* of
its spend: size, how many inputs and outputs were involved, where value went,
how it relates to fees, and what the market was doing.

Note what is deliberately absent: **the age of the output**.

Age is the very definition of a wake-up, so a model trained on ordinary
short-lived spends would find every input maximally anomalous on age alone.
That would make the neural score a noisy restatement of the rule already in
`scorer.py`, which handles age explicitly and interpretably. Excluding age
keeps the two layers complementary: the rule reasons about *how long*, the
model reasons about *what shape*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Tx, WakeUp

FEATURE_NAMES = [
    "log_value_sats",       # 0
    "n_inputs_log",         # 1
    "n_outputs_log",        # 2
    "input_share",          # 3
    "fee_rate_ratio",       # 4
    "self_transfer_share",  # 5
    "is_segwit",            # 6
    "is_taproot",           # 7
    "is_legacy",            # 8
    "price_multiple_c",     # 9
    "round_value_flag",     # 10
    "output_fanout_ratio",  # 11
    "value_per_input_log",  # 12
]

FEATURE_COUNT = len(FEATURE_NAMES)


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(value, 0.0)))


@dataclass
class FeatureContext:
    """Optional market context supplied when extracting features."""

    price_multiple: float | None = None


def extract_features(
    wakeup: WakeUp,
    tx: Tx | None = None,
    source_tx: Tx | None = None,
    context: FeatureContext | None = None,
) -> np.ndarray:
    """Build the feature vector for one wake-up.

    `source_tx` is the transaction that *created* the spent output, used to
    detect a return-to-sender (a self transfer). It may be omitted, in which
    case the related features fall back to neutral values.
    """
    context = context or FeatureContext()
    values = np.zeros(FEATURE_COUNT, dtype=np.float64)

    values[0] = _safe_log1p(wakeup.value_sats)

    n_inputs = len(tx.inputs) if tx else 0
    n_outputs = len(tx.output_values_sats) if tx else 0
    values[1] = _safe_log1p(n_inputs)
    values[2] = _safe_log1p(n_outputs)

    # How much of the transaction's input value this one output represents.
    total_in = 0.0
    if tx:
        total_in = sum(i.prevout_value_sats or 0 for i in tx.inputs)
    values[3] = (wakeup.value_sats / total_in) if total_in > 0 else 0.0

    # Fee relative to the value being moved: large value, tiny fee.
    if tx and tx.fee_sats is not None and wakeup.value_sats > 0:
        values[4] = (tx.fee_sats / wakeup.value_sats) * 10_000

    if tx and wakeup.address and tx.output_addresses:
        recipients = [a for a in tx.output_addresses if a]
        if recipients:
            returning = sum(1 for a in recipients if a == wakeup.address)
            values[5] = returning / len(recipients)
            values[11] = len(set(recipients)) / len(recipients)

    script = (wakeup.script_type or "").lower()
    values[6] = 1.0 if ("wpkh" in script or "wsh" in script) else 0.0
    values[7] = 1.0 if "taproot" in script or script.startswith("p2tr") else 0.0
    values[8] = 1.0 if "p2pkh" in script or script == "pubkeyhash" else 0.0

    if context.price_multiple is not None:
        values[9] = context.price_multiple

    # Whole-coin amounts are a human tell; machine outputs rarely land there.
    btc = wakeup.value_sats / 100_000_000
    values[10] = 1.0 if btc >= 1.0 and abs(btc - round(btc)) < 1e-9 else 0.0

    values[12] = _safe_log1p(wakeup.value_sats / n_inputs) if n_inputs else 0.0

    return values


@dataclass
class Normalizer:
    """Mean/std standardisation, saved with the model so inference matches training."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Normalizer":
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # A feature that never varies would divide by zero; leave it centred only.
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        return matrix * self.std + self.mean
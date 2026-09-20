"""Unsupervised anomaly detection over spend features.

The model is an autoencoder trained only on *ordinary* spends (short-lived
outputs, everyday amounts and transaction shapes). It never sees a labelled
cause. What it learns is the manifold of normal spending; a wake-up is then
scored by how badly that manifold fails to reconstruct it.

This is the honest use of a neural network on this data. There are no ground
truth labels for *why* a wallet was dormant, so a supervised classifier would
be learning the labeller's assumptions. Anomaly detection needs no labels: it
asks only "how unlike ordinary spending is this?", which the chain data can
actually answer.

Per-feature reconstruction error is retained so a score can be explained by
naming the features that drove it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_NAMES, FeatureContext, Normalizer, extract_features
from .models import Tx, WakeUp


@dataclass
class AnomalyScore:
    """How unusual a spend is, with the features responsible."""

    score: float                       # robust z-score against the training distribution
    raw_error: float                   # mean squared reconstruction error
    top_features: list[tuple[str, float]] = field(default_factory=list)
    trained: bool = True

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "raw_error": round(self.raw_error, 6),
            "top_features": [
                {"feature": name, "contribution": round(value, 4)}
                for name, value in self.top_features
            ],
            "trained": self.trained,
        }


class AnomalyDetector:
    """Wraps a trained autoencoder plus the normalisation it was fit with."""

    def __init__(self, net, normalizer: Normalizer, error_mean: float, error_std: float):
        self.net = net
        self.normalizer = normalizer
        self.error_mean = float(error_mean)
        self.error_std = float(error_std) if error_std > 1e-9 else 1.0

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector | None":
        """Load a saved model.

        Returns None when no model exists. Raises ValueError when the model was
        trained against a different feature layout: silently applying it would
        produce confident nonsense, which is worse than refusing to run.
        """
        import os

        if not os.path.exists(path):
            return None
        from .features import FEATURE_COUNT
        from .nn import load_network

        net, extra = load_network(path)
        saved_count = int(extra["feature_count"]) if "feature_count" in extra else None
        if saved_count != FEATURE_COUNT:
            raise ValueError(
                f"model at {path} was trained on {saved_count} features but the "
                f"code expects {FEATURE_COUNT}; retrain with `dormant-radar train`"
            )
        normalizer = Normalizer(mean=extra["mean"], std=extra["std"])
        return cls(
            net=net,
            normalizer=normalizer,
            error_mean=float(extra["error_mean"]),
            error_std=float(extra["error_std"]),
        )

    def score(
        self,
        wakeup: WakeUp,
        tx: Tx | None = None,
        context: FeatureContext | None = None,
    ) -> AnomalyScore:
        vector = extract_features(wakeup, tx=tx, context=context)
        normalised = self.normalizer.transform(vector.reshape(1, -1))
        reconstructed = self.net.predict(normalised)
        per_feature = (reconstructed - normalised) ** 2
        raw_error = float(per_feature.mean())

        # Robust standardisation against the training error distribution so a
        # score of 1.0 means "one training-spread above ordinary".
        robust = (raw_error - self.error_mean) / self.error_std

        contributions = per_feature[0]
        order = np.argsort(contributions)[::-1]
        top = [(FEATURE_NAMES[i], float(contributions[i])) for i in order[:4]]

        return AnomalyScore(score=float(robust), raw_error=raw_error, top_features=top)
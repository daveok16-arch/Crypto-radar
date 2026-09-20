"""Collection and training for the anomaly model.

Training data is the opposite of the detection target: many ordinary spends
(outputs that moved quickly) plus the model's own reconstruction objective.
Nothing here uses a cause label, because none exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .chain import MempoolClient
from .config import Settings

from .features import FeatureContext, Normalizer, extract_features
from .models import Tx, WakeUp
from .nn import Adam, NeuralNet, save_network

logger = logging.getLogger(__name__)


@dataclass
class TrainingSet:
    matrix: np.ndarray
    contexts: list[FeatureContext] = field(default_factory=list)

    def __len__(self) -> int:
        return self.matrix.shape[0]


def collect_ordinary_spends(
    settings: Settings,
    client: Optional[MempoolClient] = None,
    blocks: int = 20,
    max_inputs_per_tx: int = 1,
    min_age_blocks: int = 1,
    max_age_blocks: int = 5_000,
) -> TrainingSet:
    """Walk recent blocks and keep features of short-lived, simple spends.

    Ordinary spending means: an output that moved within a few thousand
    blocks, from a normal transaction. Those are the samples the autoencoder
    treats as "normal". Mixed-in features keep the feature space honest.
    """
    client = client or MempoolClient(settings)
    tip = client.tip_height()
    rows: list[np.ndarray] = []
    contexts: list[FeatureContext] = []

    for height in range(tip - blocks + 1, tip + 1):
        block_hash = client.block_hash(height)
        if not block_hash:
            continue
        txids = client.block_txids(block_hash)[: settings.max_txs_per_block]
        for txid in txids:
            try:
                tx = client.transaction(txid)
            except Exception as exc:
                logger.warning("skipping %s during collection: %s", txid, exc)
                continue
            if tx is None:
                continue
            # Ordinary spending is not a large sweep of many inputs.
            if len(tx.inputs) > max_inputs_per_tx:
                continue
            for tx_input in tx.inputs:
                source_height = tx_input.prevout_block_height
                value = tx_input.prevout_value_sats
                if source_height is None or value is None:
                    continue
                age = height - source_height
                if age < min_age_blocks or age > max_age_blocks:
                    continue
                wakeup = WakeUp(
                    txid=tx.txid,
                    spend_block_height=height,
                    spent=tx_input.outpoint,
                    value_sats=value,
                    address=tx_input.prevout_address,
                    dormant_blocks=age,
                    dormant_years=age / 52_560,
                    script_type=tx_input.prevout_script_type,
                    observed_at=0.0,
                )
                rows.append(extract_features(wakeup, tx=tx))
                contexts.append(FeatureContext())

    if not rows:
        raise RuntimeError(
            "collected no ordinary spends; the block window may be too small"
        )
    return TrainingSet(matrix=np.vstack(rows), contexts=contexts)


@dataclass
class TrainReport:
    samples: int
    train_samples: int
    validation_samples: int
    epochs: int
    initial_loss: float
    final_loss: float
    validation_loss: float
    error_mean: float
    error_std: float
    model_path: str


def train_autoencoder(
    training: TrainingSet,
    model_path: str,
    hidden_sizes: tuple[int, ...] = (24, 12, 24),
    epochs: int = 400,
    learning_rate: float = 5e-3,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 7,
    verbose: bool = False,
) -> TrainReport:
    """Fit the autoencoder to ordinary spends and persist it.

    The anomaly scale is calibrated on a held-out split, not on the data the
    model was fit to. An autoencoder can drive its training error arbitrarily
    low by memorising, which would leave an artificially tiny error spread and
    make every real-world input look like a many-sigma outlier. Measuring the
    spread on unseen data keeps the reported score meaningful.
    """
    from .features import FEATURE_COUNT

    normalizer = Normalizer.fit(training.matrix)
    x = normalizer.transform(training.matrix)

    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    n_val = max(1, int(len(order) * validation_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    x_train, x_val = x[train_idx], x[val_idx]

    sizes = [FEATURE_COUNT, *hidden_sizes, FEATURE_COUNT]
    net = NeuralNet(sizes, seed=seed)
    opt = Adam(net.params(), lr=learning_rate)

    initial_loss = None
    final_loss = float("nan")
    n = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        batch_order = rng.permutation(n)
        epoch_losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = batch_order[start : start + batch_size]
            batch = x_train[idx]
            loss = net.loss_and_grad(batch, batch)
            opt.step(net.grads())
            epoch_losses.append(loss)
        final_loss = float(np.mean(epoch_losses))
        if initial_loss is None:
            initial_loss = final_loss
        if verbose and epoch % 100 == 0:
            logger.info("epoch %d train loss %.6f", epoch, final_loss)

    # Calibrate the anomaly scale on data the model has not seen.
    val_reconstructed = net.predict(x_val)
    val_errors = ((val_reconstructed - x_val) ** 2).mean(axis=1)
    validation_loss = float(val_errors.mean())

    save_network(
        net,
        model_path,
        extra={
            "mean": normalizer.mean,
            "std": normalizer.std,
            "error_mean": np.array(validation_loss),
            "error_std": np.array(val_errors.std()),
            "feature_count": np.array(FEATURE_COUNT),
        },
    )

    return TrainReport(
        samples=x.shape[0],
        train_samples=x_train.shape[0],
        validation_samples=x_val.shape[0],
        epochs=epochs,
        initial_loss=float(initial_loss),
        final_loss=final_loss,
        validation_loss=validation_loss,
        error_mean=validation_loss,
        error_std=float(val_errors.std()),
        model_path=model_path,
    )
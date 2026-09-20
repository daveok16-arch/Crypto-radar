"""Anomaly detector and hybrid ensemble tests.

The key behavioural claim to pin down: a model trained only on ordinary
spends assigns clearly higher anomaly scores to genuinely unusual shapes,
and the ensemble's neural adjustment stays inside its documented cap.
"""

import numpy as np
import pytest

from dormant_radar.anomaly import AnomalyDetector
from dormant_radar.ensemble import MAX_NEURAL_ADJUSTMENT, combine
from dormant_radar.models import Outpoint, Tx, TxInput, WakeUp
from dormant_radar.trainer import TrainingSet, train_autoencoder


def make_wakeup(years=6.0, value=500_000_000, address="bc1qowner", n_inputs=1):
    return WakeUp(
        txid="aa" * 32,
        spend_block_height=900_000,
        spent=Outpoint("bb" * 32, 0),
        value_sats=value,
        address=address,
        dormant_blocks=int(years * 52_560),
        dormant_years=years,
        script_type="v0_p2wpkh",
        observed_at=1_700_000_000.0,
    )


def make_tx(n_inputs=1, value=500_000_000, addresses=("bc1qdst",), fee=1000):
    return Tx(
        txid="cc" * 32,
        block_height=900_000,
        inputs=[
            TxInput(Outpoint(f"{i:02x}" * 32, 0), value, "bc1qsender", 800_000, "v0_p2wpkh")
            for i in range(n_inputs)
        ],
        output_values_sats=[value],
        output_addresses=list(addresses),
        fee_sats=fee,
    )


def train_small_model(tmp_path, seed=0):
    """Fit a real model on synthetic ordinary data so tests exercise real inference."""
    from dormant_radar.features import extract_features

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(400):
        wakeup = make_wakeup(
            years=float(rng.uniform(0.1, 2.0)),
            value=int(rng.uniform(1e6, 5e8)),
        )
        rows.append(extract_features(wakeup, tx=make_tx(n_inputs=1)))
    training = TrainingSet(matrix=np.vstack(rows))
    path = str(tmp_path / "model.npz")
    train_autoencoder(training, model_path=path, epochs=150, verbose=False)
    detector = AnomalyDetector.load(path)
    assert detector is not None
    return detector


def test_load_returns_none_when_no_model(tmp_path):
    assert AnomalyDetector.load(str(tmp_path / "absent.npz")) is None


def test_load_rejects_model_with_wrong_feature_count(tmp_path):
    """A model trained on a different feature layout must refuse to load.

    Silently applying it would broadcast-fail at best and produce confident
    nonsense at worst, so the mismatch is an explicit error.
    """
    from dormant_radar.nn import NeuralNet, save_network

    net = NeuralNet([5, 4, 5], seed=1)
    path = str(tmp_path / "stale.npz")
    save_network(
        net,
        path,
        extra={
            "mean": np.zeros(5),
            "std": np.ones(5),
            "error_mean": np.array(0.1),
            "error_std": np.array(0.01),
            "feature_count": np.array(5),
        },
    )
    with pytest.raises(ValueError, match="feature"):
        AnomalyDetector.load(path)


def test_anomaly_score_dict_shape(tmp_path):
    detector = train_small_model(tmp_path)
    payload = detector.score(make_wakeup(), tx=make_tx()).as_dict()
    assert set(payload) == {"score", "raw_error", "top_features", "trained"}
    assert isinstance(payload["score"], float)
    assert payload["top_features"]
    assert all(set(item) == {"feature", "contribution"} for item in payload["top_features"])


def test_unusual_shape_scores_more_anomalous_than_ordinary(tmp_path):
    """The central behavioural claim, tested on a model trained only on normal data.

    Age is deliberately not a feature, so the unusual case must differ in
    *shape*: far larger value arriving from a sweep of many inputs.
    """
    detector = train_small_model(tmp_path)

    ordinary = detector.score(
        make_wakeup(years=1.0, value=100_000_000), tx=make_tx(n_inputs=1)
    )
    unusual = detector.score(
        make_wakeup(years=14.0, value=900_000_000), tx=make_tx(n_inputs=60)
    )
    assert unusual.raw_error > ordinary.raw_error


def test_age_alone_does_not_change_score(tmp_path):
    """A same-shape spend must score identically regardless of how long it slept."""
    detector = train_small_model(tmp_path)
    young = detector.score(make_wakeup(years=1.0, value=100_000_000), tx=make_tx())
    old = detector.score(make_wakeup(years=15.0, value=100_000_000), tx=make_tx())
    assert young.raw_error == old.raw_error


def test_ensemble_without_detector_matches_rules(tmp_path):
    result = combine(make_wakeup(), tx=make_tx(), detector=None)
    assert result.anomaly is None
    assert result.final_distribution == result.rule_distribution
    assert result.adjustment_log_odds == 0.0
    assert result.notes


def test_ensemble_distribution_stays_a_probability_simplex(tmp_path):
    detector = train_small_model(tmp_path)
    result = combine(make_wakeup(), tx=make_tx(), detector=detector)
    assert abs(sum(result.final_distribution.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in result.final_distribution.values())


def test_neural_adjustment_is_capped(tmp_path):
    """No anomaly signal, however extreme, may exceed the documented cap."""
    detector = train_small_model(tmp_path)
    extreme = combine(
        make_wakeup(years=16.0, value=999_000_000),
        tx=make_tx(n_inputs=200),
        detector=detector,
    )
    assert abs(extreme.adjustment_log_odds) <= MAX_NEURAL_ADJUSTMENT + 1e-9


def test_hybrid_payload_exposes_both_scores(tmp_path):
    detector = train_small_model(tmp_path)
    payload = combine(make_wakeup(), tx=make_tx(), detector=detector).as_dict()
    assert "rule_distribution" in payload
    assert "final_distribution" in payload
    assert payload["anomaly"] is not None
    assert "neural_adjustment_log_odds" in payload


def test_ensemble_notes_explain_the_decision(tmp_path):
    detector = train_small_model(tmp_path)
    result = combine(make_wakeup(), tx=make_tx(), detector=detector)
    assert result.notes
    assert any("anomaly" in note.lower() for note in result.notes)


def test_scoring_is_deterministic(tmp_path):
    detector = train_small_model(tmp_path)
    first = detector.score(make_wakeup(), tx=make_tx()).raw_error
    second = detector.score(make_wakeup(), tx=make_tx()).raw_error
    assert first == second
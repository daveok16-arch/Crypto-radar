"""Feature extraction and normalisation tests."""

import numpy as np

from dormant_radar.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureContext,
    Normalizer,
    extract_features,
)
from dormant_radar.models import Outpoint, Tx, TxInput, WakeUp


def make_wakeup(years=6.0, value=500_000_000, address="bc1qowner", script="v0_p2wpkh"):
    return WakeUp(
        txid="aa" * 32,
        spend_block_height=900_000,
        spent=Outpoint("bb" * 32, 0),
        value_sats=value,
        address=address,
        dormant_blocks=int(years * 52_560),
        dormant_years=years,
        script_type=script,
        observed_at=1_700_000_000.0,
    )


def test_vector_has_declared_length():
    vector = extract_features(make_wakeup())
    assert vector.shape == (FEATURE_COUNT,)
    assert FEATURE_COUNT == len(FEATURE_NAMES)


def test_features_are_finite():
    vector = extract_features(make_wakeup(), tx=Tx(txid="cc" * 32, block_height=1))
    assert np.all(np.isfinite(vector))


def test_no_tx_does_not_crash():
    """Missing transaction context must degrade to neutral values, not raise."""
    vector = extract_features(make_wakeup(), tx=None)
    assert np.all(np.isfinite(vector))
    # input/output derived features are zeroed when there is no transaction.
    assert vector[FEATURE_NAMES.index("n_inputs_log")] == 0.0


def test_age_is_excluded_from_features():
    """Age must not be a feature: it would make the neural score restate the rule.

    Age is the definition of a wake-up, so including it would make every input
    look anomalous to a model trained on short-lived spends. This test fails
    loudly if anyone reintroduces it.
    """
    assert "log_dormant_blocks" not in FEATURE_NAMES
    young = extract_features(make_wakeup(years=1.0))
    old = extract_features(make_wakeup(years=14.0))
    # Two spends of the same shape, differing only in age, must be identical.
    assert np.array_equal(young, old)


def test_value_is_monotonic():
    small = extract_features(make_wakeup(value=1_000_000))
    large = extract_features(make_wakeup(value=1_000_000_000))
    assert large[FEATURE_NAMES.index("log_value_sats")] > small[FEATURE_NAMES.index("log_value_sats")]


def test_self_transfer_share_detected():
    wakeup = make_wakeup(address="bc1qowner")
    tx = Tx(
        txid="dd" * 32,
        block_height=900_000,
        output_values_sats=[500_000_000],
        output_addresses=["bc1qowner"],
    )
    vector = extract_features(wakeup, tx=tx)
    assert vector[FEATURE_NAMES.index("self_transfer_share")] == 1.0

    other = Tx(
        txid="ee" * 32,
        block_height=900_000,
        output_values_sats=[500_000_000],
        output_addresses=["bc1qsomeoneelse"],
    )
    assert extract_features(wakeup, tx=other)[FEATURE_NAMES.index("self_transfer_share")] == 0.0


def test_script_type_flags():
    segwit = extract_features(make_wakeup(script="v0_p2wpkh"))
    assert segwit[FEATURE_NAMES.index("is_segwit")] == 1.0

    taproot = extract_features(make_wakeup(script="p2tr"))
    assert taproot[FEATURE_NAMES.index("is_taproot")] == 1.0

    legacy = extract_features(make_wakeup(script="p2pkh"))
    assert legacy[FEATURE_NAMES.index("is_legacy")] == 1.0


def test_round_value_flag():
    whole = extract_features(make_wakeup(value=500_000_000))  # exactly 5 BTC
    odd = extract_features(make_wakeup(value=512_345_678))
    index = FEATURE_NAMES.index("round_value_flag")
    assert whole[index] == 1.0
    assert odd[index] == 0.0


def test_price_multiple_passed_through():
    vector = extract_features(make_wakeup(), context=FeatureContext(price_multiple=2.5))
    assert vector[FEATURE_NAMES.index("price_multiple_c")] == 2.5


def test_many_inputs_change_shape_features():
    wakeup = make_wakeup()
    one = Tx(txid="a1" * 32, block_height=1,
             inputs=[TxInput(Outpoint("b1" * 32, 0), 500_000_000)])
    many = Tx(
        txid="a2" * 32,
        block_height=1,
        inputs=[TxInput(Outpoint(f"{i:02x}" * 32, 0), 500_000_000) for i in range(1, 30)],
    )
    assert (
        extract_features(wakeup, tx=many)[FEATURE_NAMES.index("n_inputs_log")]
        > extract_features(wakeup, tx=one)[FEATURE_NAMES.index("n_inputs_log")]
    )


def test_normalizer_roundtrip():
    matrix = np.random.default_rng(2).normal(size=(50, 4)) * 5 + 10
    normalizer = Normalizer.fit(matrix)
    transformed = normalizer.transform(matrix)
    assert np.allclose(normalizer.inverse(transformed), matrix)


def test_normalizer_handles_constant_feature():
    """A feature with zero variance must not produce division by zero."""
    matrix = np.ones((10, 3))
    normalizer = Normalizer.fit(matrix)
    transformed = normalizer.transform(matrix)
    assert np.all(np.isfinite(transformed))
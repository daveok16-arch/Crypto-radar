"""Alert policy tests: what gets escalated, and what stays silent."""

from dormant_radar.alerting import AlertPolicy, evaluate


def make_wakeup(
    value_sats=1_000_000_000,
    years=5.0,
    anomaly=None,
    cluster_size=None,
    self_transfer=False,
    known=True,
):
    wakeup = {
        "txid": "aa" * 32,
        "spent_outpoint": "bb" * 32 + ":0",
        "value_sats": value_sats,
        "value_btc": value_sats / 100_000_000,
        "dormant_years": years,
        "dormant_blocks": int(years * 52_560),
        "spend_block_height": 900_000,
        "address": "bc1qexample",
        "cause": {"hypothesis": "holding", "confidence": 0.2, "distribution": {},
                  "rationale": []},
    }
    if anomaly is not None:
        wakeup["neural"] = {"anomaly": {"score": anomaly}}
    if cluster_size is not None or self_transfer or not known:
        wakeup["ownership"] = {
            "known": known,
            "cluster_size": cluster_size or 0,
            "is_self_transfer": self_transfer,
            "owned_output_share": 1.0 if self_transfer else 0.0,
            "cluster_id": "root",
        }
    return wakeup


def test_quiet_when_nothing_clears_the_policy():
    """Silence is the normal state; a small recent spend must not alert."""
    policy = AlertPolicy(min_value_sats=5_000_000_000, min_dormant_years=8.0)
    assert evaluate(make_wakeup(value_sats=10_000_000, years=1.0), policy) is None


def test_value_trigger_fires_and_explains_itself():
    policy = AlertPolicy(min_value_sats=5_000_000_000, min_dormant_years=None)
    alert = evaluate(make_wakeup(value_sats=10_000_000_000, years=1.0), policy)
    assert alert is not None
    assert alert.severity == "high"
    assert any("value" in reason for reason in alert.reasons)


def test_dormancy_trigger_fires():
    policy = AlertPolicy(min_value_sats=None, min_dormant_years=8.0)
    alert = evaluate(make_wakeup(value_sats=1_000, years=12.0), policy)
    assert alert is not None
    assert any("dormant" in reason for reason in alert.reasons)


def test_triggers_are_or_ed_not_anded():
    """Clearing either threshold is enough; both are not required."""
    policy = AlertPolicy(min_value_sats=5_000_000_000, min_dormant_years=8.0)
    big_recent = evaluate(make_wakeup(value_sats=9_000_000_000, years=0.5), policy)
    small_ancient = evaluate(make_wakeup(value_sats=1_000, years=15.0), policy)
    assert big_recent is not None
    assert small_ancient is not None


def test_anomaly_trigger_only_when_configured():
    off = AlertPolicy(min_value_sats=None, min_dormant_years=None)
    assert evaluate(make_wakeup(anomaly=99.0), off) is None

    on = AlertPolicy(
        min_value_sats=None, min_dormant_years=None, min_anomaly_score=10.0
    )
    alert = evaluate(make_wakeup(anomaly=99.0), on)
    assert alert is not None
    assert any("anomaly" in reason for reason in alert.reasons)


def test_anomaly_trigger_ignores_missing_score():
    """A model that was never trained must not crash or fire on nothing."""
    policy = AlertPolicy(
        min_value_sats=None, min_dormant_years=None, min_anomaly_score=1.0
    )
    assert evaluate(make_wakeup(anomaly=None), policy) is None


def test_cluster_trigger_requires_known_ownership():
    policy = AlertPolicy(
        min_value_sats=None, min_dormant_years=None, min_cluster_size=100
    )
    assert evaluate(make_wakeup(known=False, cluster_size=500), policy) is None
    alert = evaluate(make_wakeup(known=True, cluster_size=500), policy)
    assert alert is not None
    assert any("cluster" in reason for reason in alert.reasons)


def test_self_transfer_is_silent_by_default():
    policy = AlertPolicy(min_value_sats=None, min_dormant_years=None)
    assert evaluate(make_wakeup(self_transfer=True, cluster_size=5), policy) is None


def test_self_transfer_alerting_is_opt_in_and_marked_low():
    policy = AlertPolicy(
        min_value_sats=None,
        min_dormant_years=None,
        alert_on_self_transfer=True,
    )
    alert = evaluate(make_wakeup(self_transfer=True, cluster_size=5), policy)
    assert alert is not None
    assert alert.severity == "low"


def test_self_transfer_detected_via_rationale_without_clustering():
    """Without ownership data, the rule's rationale is the fallback signal."""
    policy = AlertPolicy(
        min_value_sats=None, min_dormant_years=None, alert_on_self_transfer=True
    )
    wakeup = make_wakeup(known=False)
    wakeup["cause"]["rationale"] = [
        "Every output returns to the originating address, the signature "
        "of a self-transfer that preserves custody (same-address match "
        "only; ownership clustering unavailable)."
    ]
    alert = evaluate(wakeup, policy)
    assert alert is not None


def test_dedupe_key_uses_outpoint_not_txid():
    """One transaction can carry several dormant inputs; each is distinct."""
    policy = AlertPolicy(min_value_sats=1, min_dormant_years=None)
    first = evaluate(make_wakeup(), policy)
    assert first.dedupe_key == "bb" * 32 + ":0"
    assert first.dedupe_key != first.txid


def test_single_trigger_property():
    assert AlertPolicy(min_value_sats=1, min_dormant_years=None).single_trigger
    assert not AlertPolicy(
        min_value_sats=1, min_dormant_years=1.0
    ).single_trigger
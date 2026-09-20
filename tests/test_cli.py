"""CLI tests.

These exist because a missing import in `alerts` shipped unnoticed: no test
exercised the CLI entry points, so a NameError only appeared when a human ran
the command. Every subcommand is invoked here so that class of bug is caught.
"""

import json

import pytest

from dormant_radar import cli


def run(argv, monkeypatch, tmp_path, env=None):
    """Invoke the CLI with an isolated database."""
    monkeypatch.setenv("DORMANT_RADAR_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("DORMANT_RADAR_CLUSTERS", str(tmp_path / "clusters.db"))
    monkeypatch.setenv("DORMANT_RADAR_MODEL", str(tmp_path / "no-model.npz"))
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return cli.main(argv)


def test_alerts_command_reports_policy(capsys, monkeypatch, tmp_path):
    code = run(["alerts"], monkeypatch, tmp_path, env={"ALERTS_ENABLED": "0"})
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["enabled"] is False
    assert "policy" in payload
    assert payload["already_alerted"] == 0


def test_alerts_command_accepts_threshold_overrides(capsys, monkeypatch, tmp_path):
    code = run(
        ["alerts"],
        monkeypatch,
        tmp_path,
        env={"ALERT_MIN_VALUE_SATS": "1000000", "ALERT_MIN_DORMANT_YEARS": "3.5"},
    )
    assert code == 0
    policy = json.loads(capsys.readouterr().out)["policy"]
    assert policy["min_value_sats"] == 1_000_000
    assert policy["min_dormant_years"] == 3.5


def test_alerts_command_lists_configured_channels(capsys, monkeypatch, tmp_path):
    """A token plus a chat id must surface as a telegram channel."""
    code = run(
        ["alerts"],
        monkeypatch,
        tmp_path,
        env={
            "TELEGRAM_BOT_TOKEN": "123456789:AAFakeTokenValueForTesting_abcdefghijklmnop",
            "TELEGRAM_CHAT_ID": "12345",
        },
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["channels"] == ["telegram"]


def test_alerts_command_unconfigured_has_no_channels(capsys, monkeypatch, tmp_path):
    """Channels must be empty when nothing is configured.

    The environment is cleared explicitly: a developer or CI machine may well
    have a real token set, and the test must not depend on that being absent.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_IDS", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    code = run(["alerts"], monkeypatch, tmp_path)
    assert code == 0
    assert json.loads(capsys.readouterr().out)["channels"] == []


def test_stats_command_on_empty_store(capsys, monkeypatch, tmp_path):
    assert run(["stats"], monkeypatch, tmp_path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_wakeups"] == 0


def test_events_command_on_empty_store(capsys, monkeypatch, tmp_path):
    assert run(["events"], monkeypatch, tmp_path) == 0
    assert json.loads(capsys.readouterr().out) == {"count": 0, "events": []}


def test_clusters_command_on_empty_graph(capsys, monkeypatch, tmp_path):
    assert run(["clusters"], monkeypatch, tmp_path) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stats"]["addresses"] == 0


def test_clusters_command_address_lookup(capsys, monkeypatch, tmp_path):
    code = run(["clusters", "--address", "bc1qunknown"], monkeypatch, tmp_path)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # Unknown input must be reported as unknown, not as a cluster of one.
    assert payload["cluster_id"] is None
    assert payload["cluster_size"] == 0


def test_anomaly_command_without_model_exits_nonzero(capsys, monkeypatch, tmp_path):
    code = run(["anomaly"], monkeypatch, tmp_path)
    assert code == 1
    assert "train" in capsys.readouterr().out


def test_hunt_dry_run_sends_nothing(capsys, monkeypatch, tmp_path, monkeypatch_scan):
    code = run(
        ["hunt", "--dry-run"],
        monkeypatch,
        tmp_path,
        env={"ALERTS_ENABLED": "1"},
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["alerts"]["delivered"] == 0
    assert "nothing was sent" in payload["alerts"]["note"]


@pytest.fixture
def monkeypatch_scan(monkeypatch):
    """Stub the network scan so the CLI test stays offline."""
    from dormant_radar.scanner import ScanResult

    def fake_scan(settings, store=None, **kwargs):
        return ScanResult(tip_height=1, scanned_from=1, scanned_to=1)

    monkeypatch.setattr("dormant_radar.scanner.scan_once", fake_scan)
    return monkeypatch
"""Channel tests.

The Telegram tests matter most: the bot token rides in the request URL, so any
code path that logs a failed request leaks it. These tests assert on captured
log output, not just on return values.
"""

import logging

import pytest

from dormant_radar.alerting import Alert
from dormant_radar.notify import EmailChannel, TelegramChannel, build_channels
from dormant_radar import redact as redact_module

FAKE_TOKEN = "123456789:AAFakeTokenValueForTesting_abcdefghijklmnop"


@pytest.fixture(autouse=True)
def clean_registry():
    redact_module.clear_registered()
    yield
    redact_module.clear_registered()


def make_alert():
    return Alert(
        wakeup={
            "txid": "tt" * 32,
            "spent_outpoint": "oo" * 32 + ":0",
            "spend_block_height": 900_000,
            "value_sats": 20_000_000_000,
            "value_btc": 200.0,
            "address": "bc1qexample",
            "dormant_blocks": 600_000,
            "dormant_years": 11.4,
            "cause": {"hypothesis": "holding", "confidence": 0.2,
                      "distribution": {}, "rationale": ["because"]},
        },
        reasons=["value clears the threshold"],
        severity="high",
    )


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


def test_telegram_sends_to_every_chat(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json["chat_id"]))
        return FakeResponse(200)

    monkeypatch.setattr("dormant_radar.notify.requests.post", fake_post)
    channel = TelegramChannel(bot_token=FAKE_TOKEN, chat_ids=["1", "2"])
    assert channel.send(make_alert()) is True
    assert [c[1] for c in calls] == ["1", "2"]


def test_telegram_reports_failure_without_raising(monkeypatch):
    monkeypatch.setattr(
        "dormant_radar.notify.requests.post",
        lambda *a, **k: FakeResponse(500),
    )
    channel = TelegramChannel(bot_token=FAKE_TOKEN, chat_ids=["1"])
    assert channel.send(make_alert()) is False


def test_telegram_does_not_log_token_on_http_error(monkeypatch, caplog):
    monkeypatch.setattr(
        "dormant_radar.notify.requests.post",
        lambda *a, **k: FakeResponse(429),
    )
    channel = TelegramChannel(bot_token=FAKE_TOKEN, chat_ids=["1"])
    with caplog.at_level(logging.DEBUG):
        channel.send(make_alert())
    assert FAKE_TOKEN not in caplog.text


def test_telegram_does_not_log_token_on_exception(monkeypatch, caplog):
    """The exact leak this design prevents: a request exception embeds the URL."""
    import requests

    def boom(url, json=None, timeout=None):
        # Simulate what requests actually does: the URL, including the token,
        # appears in the exception message.
        raise requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
            f"exceeded with url: /bot{FAKE_TOKEN}/sendMessage"
        )

    monkeypatch.setattr("dormant_radar.notify.requests.post", boom)
    channel = TelegramChannel(bot_token=FAKE_TOKEN, chat_ids=["1"])
    with caplog.at_level(logging.DEBUG):
        assert channel.send(make_alert()) is False

    assert FAKE_TOKEN not in caplog.text
    assert "api.telegram.org" in caplog.text  # the useful part survives


def test_telegram_without_chats_is_a_no_op():
    channel = TelegramChannel(bot_token=FAKE_TOKEN, chat_ids=[])
    assert channel.send(make_alert()) is False


def test_email_channel_requires_configuration():
    assert EmailChannel(host="", recipients=()).send(make_alert()) is False


def test_email_sends_and_does_not_log_password(monkeypatch, caplog):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self, context=None):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = user

        def send_message(self, message):
            sent["subject"] = message["Subject"]
            sent["body"] = message.get_content()

    monkeypatch.setattr("dormant_radar.notify.smtplib.SMTP", FakeSMTP)
    channel = EmailChannel(
        host="smtp.example.com",
        username="u",
        password="hunter2secret",
        sender="from@example.com",
        recipients=("to@example.com",),
    )
    with caplog.at_level(logging.DEBUG):
        assert channel.send(make_alert()) is True

    assert sent["tls"] is True
    assert sent["login"] == "u"
    assert "hunter2secret" not in caplog.text
    assert "Dormant wake-up" in sent["subject"]


def test_email_failure_is_caught_and_logged(monkeypatch, caplog):
    def boom(host, port, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("dormant_radar.notify.smtplib.SMTP", boom)
    channel = EmailChannel(host="smtp.example.com", recipients=("to@example.com",))
    with caplog.at_level(logging.DEBUG):
        assert channel.send(make_alert()) is False
    assert "connection refused" in caplog.text


# --- channel construction from settings --------------------------------


class FakeSettings:
    telegram_bot_token = None
    telegram_chat_ids = ()
    smtp_host = None
    smtp_port = 587
    smtp_username = None
    smtp_password = None
    smtp_starttls = True
    alert_email_from = None
    alert_email_to = ()
    alert_include_rationale = True


def test_build_channels_empty_when_nothing_configured():
    assert build_channels(FakeSettings()) == []


def test_build_channels_telegram_only():
    settings = FakeSettings()
    settings.telegram_bot_token = FAKE_TOKEN
    settings.telegram_chat_ids = ("123",)
    channels = build_channels(settings)
    assert [c.name for c in channels] == ["telegram"]


def test_build_channels_requires_recipients_for_email():
    """A host alone is not a usable channel."""
    settings = FakeSettings()
    settings.smtp_host = "smtp.example.com"
    assert build_channels(settings) == []

    settings.alert_email_to = ("to@example.com",)
    assert [c.name for c in build_channels(settings)] == ["email"]


def test_build_channels_both():
    settings = FakeSettings()
    settings.telegram_bot_token = FAKE_TOKEN
    settings.telegram_chat_ids = ("123",)
    settings.smtp_host = "smtp.example.com"
    settings.alert_email_to = ("to@example.com",)
    assert [c.name for c in build_channels(settings)] == ["telegram", "email"]
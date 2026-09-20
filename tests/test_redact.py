"""Redaction tests.

These guard the single most likely way this project leaks a credential: the
Telegram bot token travels in the request URL path, and `requests` embeds the
URL in its exception messages. A logged retry warning is enough to write a
live token into a log file.
"""

import logging

import pytest

from dormant_radar import redact as redact_module
from dormant_radar.redact import (
    RedactingFilter,
    install_redaction,
    redact,
    register_many,
    register_secret,
)

# Shaped like a real token but not one.
FAKE_TOKEN = "123456789:AAFakeTokenValueForTesting_abcdefghijklmnop"
FAKE_TELEGRAM_URL = f"https://api.telegram.org/bot{FAKE_TOKEN}/sendMessage"


@pytest.fixture(autouse=True)
def clean_registry():
    redact_module.clear_registered()
    yield
    redact_module.clear_registered()


def test_registered_secret_is_removed_from_text():
    register_secret(FAKE_TOKEN)
    assert FAKE_TOKEN not in redact(f"calling {FAKE_TELEGRAM_URL}")


def test_requests_style_exception_message_is_scrubbed():
    """The concrete failure this module exists for."""
    register_secret(FAKE_TOKEN)
    message = (
        "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
        f"exceeded with url: /bot{FAKE_TOKEN}/sendMessage"
    )
    scrubbed = redact(message)
    assert FAKE_TOKEN not in scrubbed
    assert "api.telegram.org" in scrubbed  # host is fine to keep


def test_telegram_token_scrubbed_without_registration():
    """Pattern matching must catch a token even if nobody registered it."""
    text = f"URL is /bot{FAKE_TOKEN}/sendMessage"
    assert FAKE_TOKEN not in redact(text)


def test_bearer_and_password_assignments_scrubbed():
    assert "s3cr3t" not in redact("Authorization: Bearer s3cr3t")
    assert "hunter2" not in redact("password=hunter2")
    assert "abc123" not in redact("api_key: abc123")


def test_long_hex_blobs_scrubbed():
    secret = "a" * 64
    assert secret not in redact(f"key material {secret} here")


def test_short_values_are_not_registered():
    """Avoid registering strings so short they would blank out ordinary text."""
    register_secret("abc")
    assert redact("abcdef") == "abcdef"


def test_empty_text_is_returned_unchanged():
    assert redact("") == ""


def test_unrelated_text_is_untouched():
    message = "scan complete: blocks 100-105, 42 txs, 1 wake-up"
    assert redact(message) == message


def test_overlapping_secrets_redact_cleanly():
    """A shorter secret contained in a longer one must not cause a partial leak."""
    register_many([FAKE_TOKEN, FAKE_TOKEN[:20]])
    scrubbed = redact(f"token={FAKE_TOKEN}")
    assert FAKE_TOKEN[:20] not in scrubbed
    assert FAKE_TOKEN not in scrubbed


def test_logging_filter_scrubs_message_arguments():
    """The filter must scrub before formatting, including %-style args."""
    register_secret(FAKE_TOKEN)
    logger = logging.getLogger("test.redaction.args")
    logger.propagate = False
    logger.handlers.clear()
    logger.addFilter(RedactingFilter())

    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(Capture())
    logger.setLevel(logging.DEBUG)
    logger.warning("request failed for %s", FAKE_TELEGRAM_URL)

    assert records, "expected a captured record"
    assert FAKE_TOKEN not in records[0]


def test_logging_filter_scrubs_preformatted_text():
    register_secret(FAKE_TOKEN)
    logger = logging.getLogger("test.redaction.direct")
    logger.propagate = False
    logger.handlers.clear()
    logger.addFilter(RedactingFilter())

    records: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger.addHandler(Capture())
    logger.setLevel(logging.DEBUG)
    logger.error("failed: %s", f"GET {FAKE_TELEGRAM_URL}")

    assert FAKE_TOKEN not in records[0]


def test_install_redaction_filters_handlers_too():
    """Handlers attached to the root logger must also be covered."""
    register_secret(FAKE_TOKEN)
    root = logging.getLogger()
    added = install_redaction(root)
    assert added in root.filters
    for handler in root.handlers:
        assert added in handler.filters
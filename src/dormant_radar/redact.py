"""Secret redaction for anything that leaves the process.

Why this module exists
----------------------
The Telegram Bot API embeds the bot token in the URL *path*:

    https://api.telegram.org/bot<TOKEN>/sendMessage

That means the token travels inside any request URL, and `requests`
exceptions include the URL in their message. A single logged retry warning is
enough to write a live credential into a log file, a terminal scrollback, or a
log aggregator — silently, and only once, which is the worst way to find out.

SMTP credentials leak the same way through tracebacks that stringify the
provider object or the exception.

So every message that could reach a log or a channel passes through `redact`
first. This is defence in depth: the caller should not have to remember.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

# Literal token patterns, matched even when the secret's value is unknown.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Telegram bot token, wherever it appears (path, query, prose).
    (re.compile(r"bot\d+:[A-Za-z0-9_-]{30,}"), "bot<redacted>"),
    # Telegram token without the "bot" prefix, as used bare in config.
    (re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"), "<redacted-telegram-token>"),
    # Long hex/base64 blobs that look like keys.
    (re.compile(r"\b[A-Fa-f0-9]{40,}\b"), "<redacted-hex-secret>"),
    # "Authorization: Bearer <token>" and similar. Handled before the generic
    # key=value rule because here the keyword and the secret are separated by
    # another word ("Bearer"), so adjacency-based matching would miss it.
    # The length floor is only 4: the "Bearer" keyword already disambiguates,
    # and a floor high enough to miss a short secret is a silent leak.
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{4,}"), r"\1 <redacted>"),
    # Generic "key: value" / "key=value" credential assignments.
    (re.compile(r"(?i)\b(token|password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+"),
     r"\1=<redacted>"),
]

# Values registered at runtime (from environment) are matched literally too,
# which catches secrets whose shape we cannot predict.
_registered: set[str] = set()

MIN_REGISTERED_LENGTH = 8


def register_secret(value: str | None) -> None:
    """Register a literal secret so it is redacted wherever it appears."""
    if value and len(value) >= MIN_REGISTERED_LENGTH:
        _registered.add(value)


def register_many(values: Iterable[str | None]) -> None:
    for value in values:
        register_secret(value)


def clear_registered() -> None:
    """Test helper: forget registered secrets."""
    _registered.clear()


def redact(text: str) -> str:
    """Return `text` with known and recognisable secrets replaced.

    Registered values are replaced first (longest first, so overlapping
    secrets redact cleanly), then pattern-based scrubbing catches anything
    that looks like a credential even if never registered.
    """
    if not text:
        return text
    result = text
    for secret in sorted(_registered, key=len, reverse=True):
        result = result.replace(secret, "<redacted>")
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs secrets from every record.

    Attach to the root logger so no handler — ours or a library's — can emit
    an unredacted credential. The traceback text is scrubbed too, because
    exceptions raised by HTTP libraries routinely embed the request URL.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            # Replace args in place: this is the only place the original
            # values are still available before formatting.
            if isinstance(record.args, dict):
                record.args = {
                    key: redact(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    redact(value) if isinstance(value, str) else value
                    for value in record.args
                )
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def install_redaction(logger: logging.Logger | None = None) -> RedactingFilter:
    """Install the redacting filter on a logger and its handlers.

    Applied to the root logger by default so third-party libraries cannot
    leak either.
    """
    target = logger or logging.getLogger()
    redaction = RedactingFilter()
    target.addFilter(redaction)
    for handler in target.handlers:
        handler.addFilter(redaction)
    return redaction
"""Notification channels: Telegram and email.

Privacy posture, stated plainly
-------------------------------
These are **not** confidential channels. Telegram is a third party and the
bot's message history sits on its servers; email traverses relays and lands in
a mailbox that is usually unencrypted at rest. An alert is therefore a durable
record, held by someone else, of which wallets you consider interesting.

That is a real exposure and this module does not pretend otherwise. What it
does do:

- never logs a token, because the Telegram token rides in the request URL and
  `requests` puts the URL in its exception messages (see `redact.py`);
- keeps alert bodies free of anything the operator did not choose to monitor
  (no API keys, no local paths);
- lets the operator reduce what leaves the machine, via `include_rationale`
  and a value floor in the policy.

If the alert content is itself sensitive, the honest answer is a local
channel (a file, a webhook on a private network), not a cloud chat app.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional, Protocol

import requests

from .alerting import Alert
from .redact import redact, register_secret

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"

# Telegram rejects messages longer than 4096 characters. Exceeding it yields a
# 400 and the whole message is lost, so every body is truncated deliberately.
TELEGRAM_MAX_CHARS = 4096


class Channel(Protocol):
    """A delivery target for alerts.

    `send_text` is the primitive; `send` is a convenience wrapper. Digest mode
    needs the primitive, and routing both through one transport means there is
    only one place where credentials and truncation are handled.
    """

    name: str

    def send_text(self, text: str, subject: str) -> bool:
        """Deliver a pre-rendered message. Returns True on success; never raises."""
        ...

    def send(self, alert: Alert) -> bool:
        """Deliver one alert. Returns True on success; never raises."""
        ...


def truncate_for_channel(text: str, limit: int = TELEGRAM_MAX_CHARS) -> str:
    """Trim a message to a channel's size limit without cutting mid-line.

    Truncation always announces itself: silently dropping the tail would leave
    the operator believing they had seen everything.
    """
    if len(text) <= limit:
        return text
    marker = "\n\n... message truncated (too long for this channel)"
    keep = limit - len(marker)
    if keep <= 0:
        return text[:limit]
    cut = text.rfind("\n", 0, keep)
    if cut < keep // 2:
        cut = keep
    return text[:cut] + marker


def format_alert(alert: Alert, include_rationale: bool = True) -> str:
    """Render an alert as plain text suitable for a chat message or email body."""
    w = alert.wakeup
    lines = [
        f"[{alert.severity.upper()}] Dormant wallet wake-up",
        "",
        f"Value       : {w.get('value_btc', 0)} BTC",
        f"Dormant     : {w.get('dormant_years', 0)} years ({w.get('dormant_blocks', 0)} blocks)",
        f"Block height: {w.get('spend_block_height')}",
        f"Address     : {w.get('address')}",
        f"Outpoint    : {w.get('spent_outpoint')}",
        f"Txid        : {w.get('txid')}",
    ]

    cause = w.get("cause") or {}
    if cause:
        lines.append("")
        lines.append(f"Probable cause: {cause.get('hypothesis')} (confidence {cause.get('confidence')})")
        distribution = cause.get("distribution") or {}
        if distribution:
            rendered = ", ".join(f"{k} {v}" for k, v in sorted(distribution.items()))
            lines.append(f"Distribution  : {rendered}")

    ownership = w.get("ownership") or {}
    if ownership.get("known"):
        lines.append(
            f"Ownership     : cluster of {ownership.get('cluster_size')} addresses, "
            f"self-transfer={ownership.get('is_self_transfer')}"
        )
    else:
        lines.append("Ownership     : unknown (address not previously observed)")

    lines.append("")
    lines.append("Why this alerted:")
    lines.extend(f"  - {reason}" for reason in alert.reasons)

    if include_rationale and cause.get("rationale"):
        lines.append("")
        lines.append("Reasoning:")
        lines.extend(f"  - {item}" for item in cause["rationale"])

    return redact("\n".join(lines))


@dataclass
class TelegramChannel:
    """Sends alerts via a Telegram bot."""

    bot_token: str
    chat_ids: list[str]
    timeout: float = 15.0
    include_rationale: bool = True
    name: str = "telegram"

    def __post_init__(self) -> None:
        register_secret(self.bot_token)

    def send(self, alert: Alert) -> bool:
        """Convenience wrapper: render the alert and deliver it as text."""
        text = format_alert(alert, include_rationale=self.include_rationale)
        return self.send_text(text, subject="Dormant wallet wake-up")

    def send_text(self, text: str, subject: str = "") -> bool:
        if not self.bot_token or not self.chat_ids:
            logger.warning("telegram channel not configured; skipping message")
            return False

        body = truncate_for_channel(text)
        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        delivered = False

        for chat_id in self.chat_ids:
            try:
                response = requests.post(
                    url,
                    json={"chat_id": chat_id, "text": body, "disable_web_page_preview": True},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    delivered = True
                else:
                    # Note: the URL is never logged directly; it carries the token.
                    logger.warning(
                        "telegram send failed with HTTP %s for chat %s",
                        response.status_code,
                        chat_id,
                    )
            except requests.RequestException as exc:
                # Stringify through redact(): the exception embeds the URL,
                # and therefore the token.
                logger.warning("telegram send error: %s", redact(str(exc)))
        return delivered


@dataclass
class EmailChannel:
    """Sends alerts over SMTP with STARTTLS."""

    host: str
    port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    sender: str = ""
    recipients: tuple[str, ...] = ()
    use_starttls: bool = True
    timeout: float = 20.0
    include_rationale: bool = True
    name: str = "email"

    def __post_init__(self) -> None:
        register_secret(self.password)

    def send(self, alert: Alert) -> bool:
        """Convenience wrapper: render the alert and deliver it as text."""
        subject = (
            f"[{alert.severity.upper()}] Dormant wake-up: "
            f"{alert.wakeup.get('value_btc', 0)} BTC"
        )
        text = format_alert(alert, include_rationale=self.include_rationale)
        return self.send_text(text, subject=subject)

    def send_text(self, text: str, subject: str = "") -> bool:
        if not self.host or not self.recipients:
            logger.warning("email channel not configured; skipping message")
            return False

        message = EmailMessage()
        message["Subject"] = subject or "Dormant Radar"
        message["From"] = self.sender or (self.username or "dormant-radar@localhost")
        message["To"] = ", ".join(self.recipients)
        message.set_content(text)

        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_starttls:
                    server.starttls(context=ssl.create_default_context())
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(message)
            return True
        except Exception as exc:  # smtplib raises several unrelated types
            # Redact before logging: SMTP errors can echo the auth exchange.
            logger.warning("email send error: %s", redact(str(exc)))
            return False


def format_digest(
    alerts: list[Alert],
    window_seconds: float,
    max_items: int = 25,
    include_rationale: bool = False,
) -> str:
    """Render one summary message covering every alert in a window.

    Ordering is by value, descending, so that if the body has to be truncated
    the largest movements survive. The header always states the true total,
    including items beyond `max_items`, so the count is never misleading about
    how much happened.

    `include_rationale` defaults to False: a digest is a summary, and full
    reasoning per item would bury the signal it exists to surface.
    """
    if not alerts:
        # Callers must not send an empty digest; see Notifier.run_digest.
        raise ValueError("refusing to format an empty digest")

    ordered = sorted(
        alerts, key=lambda a: a.wakeup.get("value_sats", 0), reverse=True
    )
    total_value = sum(a.wakeup.get("value_sats", 0) for a in ordered)
    window_hours = max(window_seconds / 3600.0, 0.0)

    header = [
        f"Dormant Radar digest: {len(ordered)} alert(s) in {window_hours:.1f}h",
        f"Total value: {total_value / 100_000_000:.2f} BTC",
        "",
    ]

    lines: list[str] = []
    for index, alert in enumerate(ordered[:max_items], start=1):
        w = alert.wakeup
        cause = w.get("cause") or {}
        hypothesis = cause.get("hypothesis") or "unscored"
        ownership = w.get("ownership") or {}
        if ownership.get("known"):
            owner = f"cluster {ownership.get('cluster_size')}"
            if ownership.get("is_self_transfer"):
                owner += ", self-transfer"
        else:
            owner = "ownership unknown"

        lines.append(
            f"{index}. {w.get('value_btc', 0)} BTC | "
            f"{w.get('dormant_years', 0)}y dormant | {hypothesis} | {owner}"
        )
        lines.append(f"   {w.get('address') or 'unknown address'}")
        lines.append(f"   why: {'; '.join(alert.reasons)}")
        if include_rationale and cause.get("rationale"):
            lines.append(f"   note: {cause['rationale'][0]}")
        lines.append("")

    hidden = len(ordered) - max_items
    if hidden > 0:
        lines.append(
            f"... and {hidden} more not shown "
            f"(raise ALERT_DIGEST_MAX_ITEMS to see them)."
        )

    return redact("\n".join(header + lines).rstrip())


def build_channels(settings) -> list[Channel]:
    """Construct the channels configured in settings.

    Channels are independent: if only Telegram is configured, only Telegram is
    used. A channel that is not configured is simply absent, never a no-op
    that pretends to send.
    """
    channels: list[Channel] = []

    if settings.telegram_bot_token and settings.telegram_chat_ids:
        channels.append(
            TelegramChannel(
                bot_token=settings.telegram_bot_token,
                chat_ids=list(settings.telegram_chat_ids),
                include_rationale=settings.alert_include_rationale,
            )
        )

    if settings.smtp_host and settings.alert_email_to:
        channels.append(
            EmailChannel(
                host=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_username,
                password=settings.smtp_password,
                sender=settings.alert_email_from or settings.smtp_username or "",
                recipients=tuple(settings.alert_email_to),
                use_starttls=settings.smtp_starttls,
                include_rationale=settings.alert_include_rationale,
            )
        )

    return channels
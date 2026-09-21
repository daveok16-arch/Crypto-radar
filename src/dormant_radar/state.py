"""Portable state for stateless deployments.

Why this exists
---------------
The alert path needs two pieces of small, durable state:

1. **Which outpoints have already been alerted.** Without this, every run
   re-alerts the same wake-ups. On a cloud runner each run may land on a fresh
   pod with an empty filesystem, so a local SQLite file cannot be relied on —
   the observably correct-but-terrible outcome is the same alert every hour.

2. **The scan cursor.** Without it, each run rescans the same block window.

Both are tiny (a cursor plus a set of short strings), so they travel well as a
single JSON document. The ownership graph is a different matter: it is large
and its absence degrades accuracy rather than correctness, so it is not carried
here. A run with no graph still scores, and reports `ownership.known = false`
honestly.

Backends
--------
`KVState` speaks the automation service's key-value API, which is scoped
per-automation and exists precisely for this. `LocalState` is a JSON file for
development and single-machine runs. `open_state()` picks whichever is
available, preferring the KV store so a cloud run behaves identically to a
local one.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

STATE_KEY = "dormant_radar_state"
STATE_VERSION = 1


@dataclass
class RadarState:
    """The durable state a stateless run needs in order to behave correctly."""

    version: int = STATE_VERSION
    last_scanned_height: Optional[int] = None
    alerted_outpoints: list[str] = field(default_factory=list)
    last_alert_at: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "last_scanned_height": self.last_scanned_height,
            "alerted_outpoints": self.alerted_outpoints,
            "last_alert_at": self.last_alert_at,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RadarState":
        if not isinstance(data, dict):
            return cls()
        # An unknown version means a schema we do not understand; starting fresh
        # risks re-alerting, so it is logged rather than silently ignored.
        version = data.get("version")
        if version != STATE_VERSION:
            logger.warning(
                "state version %r does not match %r; treating as empty",
                version,
                STATE_VERSION,
            )
            return cls()
        outpoints = data.get("alerted_outpoints") or []
        return cls(
            version=STATE_VERSION,
            last_scanned_height=data.get("last_scanned_height"),
            alerted_outpoints=[str(o) for o in outpoints],
            last_alert_at=data.get("last_alert_at"),
        )


class StateBackend(Protocol):
    name: str

    def load(self) -> RadarState: ...

    def save(self, state: RadarState) -> None: ...


@dataclass
class LocalState:
    """JSON file backend for development and single-machine runs."""

    path: str
    name: str = "local"

    def load(self) -> RadarState:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return RadarState.from_dict(json.load(handle))
        except FileNotFoundError:
            return RadarState()
        except (OSError, ValueError) as exc:
            # Corrupt state must not crash a run; it means re-scanning, not
            # losing alerts permanently.
            logger.warning("could not read state from %s: %s", self.path, exc)
            return RadarState()

    def save(self, state: RadarState) -> None:
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(state.as_dict(), handle)
        except OSError as exc:
            logger.warning("could not write state to %s: %s", self.path, exc)


@dataclass
class KVState:
    """Key-value backend for the automation service.

    Uses only the standard library so the automation script needs no extra
    dependencies beyond what the project already ships.
    """

    base_url: str
    token: str
    key: str = STATE_KEY
    name: str = "kv"

    def _request(self, method: str, path: str, body: Optional[dict] = None):
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        return urllib.request.urlopen(request, timeout=20)

    def load(self) -> RadarState:
        try:
            with self._request("GET", f"/v1/kv/{self.key}") as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return RadarState()  # first run
            logger.warning("kv read failed (%s); starting from empty state", exc.code)
            return RadarState()
        except Exception as exc:  # network faults must not kill a run
            logger.warning("kv read error: %s", exc)
            return RadarState()
        value = payload.get("value") if isinstance(payload, dict) else payload
        return RadarState.from_dict(value if isinstance(value, dict) else None)

    def save(self, state: RadarState) -> None:
        try:
            with self._request("PUT", f"/v1/kv/{self.key}", state.as_dict()):
                pass
            logger.info("state saved to kv store")
        except Exception as exc:
            # A failed save means the next run may repeat an alert. That is
            # loud on purpose, because it is the failure that annoys a human.
            logger.error("FAILED to persist state: %s", exc)


def open_state(local_path: Optional[str] = None) -> StateBackend:
    """Choose a backend: the KV store when configured, else a local file."""
    token = os.environ.get("AUTOMATION_KV_TOKEN", "")
    base = os.environ.get("AUTOMATION_API_URL", "").rstrip("/")
    if token and base:
        return KVState(base_url=base, token=token)
    return LocalState(
        path=local_path or os.environ.get("DORMANT_RADAR_STATE", "data/state.json")
    )
"""Read-only client for the mempool.space public REST API.

No API key is required. The client is deliberately defensive: public
endpoints rate-limit and occasionally return transient errors, so every
call is retried with exponential backoff and callers are never handed a
silently-partial result.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from .config import Settings
from .models import Outpoint, Tx, TxInput

logger = logging.getLogger(__name__)


class ChainError(RuntimeError):
    """Raised when chain data cannot be retrieved after retries."""


class MempoolClient:
    """Thin, cached wrapper around the mempool.space API."""

    def __init__(self, settings: Settings, session: Optional[requests.Session] = None):
        self.settings = settings
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "dormant-radar/0.1"})
        self._height_cache: dict[str, Optional[int]] = {}

    # -- low level -------------------------------------------------------

    def _request(self, path: str) -> Optional[requests.Response]:
        """GET with retries. Returns None on 404, raises ChainError otherwise."""
        url = f"{self.settings.base_url.rstrip('/')}{path}"
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self._session.get(
                    url, timeout=self.settings.request_timeout_seconds
                )
                if response.status_code == 404:
                    return None
                if response.status_code == 429 or response.status_code >= 500:
                    raise ChainError(f"HTTP {response.status_code} from {url}")
                response.raise_for_status()
                return response
            except (requests.RequestException, ChainError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    logger.warning(
                        "request failed (attempt %d/%d) for %s: %s",
                        attempt,
                        self.settings.max_retries,
                        url,
                        exc,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise ChainError(f"giving up on {url}: {last_error}")

    def _get(self, path: str) -> Any:
        """GET and parse a JSON body."""
        response = self._request(path)
        if response is None:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ChainError(f"invalid JSON from {path}: {exc}") from exc

    def _get_text(self, path: str) -> Optional[str]:
        """GET a plain-text body (some mempool.space endpoints are not JSON)."""
        response = self._request(path)
        if response is None:
            return None
        return response.text.strip()

    # -- endpoints -------------------------------------------------------

    def tip_height(self) -> int:
        height = self._get("/blocks/tip/height")
        if not isinstance(height, int):
            raise ChainError(f"unexpected tip height payload: {height!r}")
        return height

    def block_hash(self, height: int) -> Optional[str]:
        text = self._get_text(f"/block-height/{height}")
        if not text:
            return None
        return text

    def block_txids(self, block_hash: str) -> list[str]:
        payload = self._get(f"/block/{block_hash}/txids")
        return list(payload or [])

    def raw_tx(self, txid: str) -> Optional[dict]:
        return self._get(f"/tx/{txid}")

    def outspends(self, txid: str) -> list[dict]:
        payload = self._get(f"/tx/{txid}/outspends")
        return list(payload or [])

    def block_height_for_tx(self, txid: str) -> Optional[int]:
        """Mining height of a transaction, cached for the process lifetime."""
        if txid in self._height_cache:
            return self._height_cache[txid]
        payload = self._get(f"/tx/{txid}")
        height = None
        if isinstance(payload, dict):
            status = payload.get("status") or {}
            if status.get("confirmed"):
                height = status.get("block_height")
        self._height_cache[txid] = height
        return height

    def transaction(self, txid: str) -> Optional[Tx]:
        """Fetch a transaction and resolve its inputs' values and ages."""
        payload = self.raw_tx(txid)
        if not isinstance(payload, dict):
            return None
        return self.parse_tx(payload, resolve_prevout_height=True)

    def address_txs(
        self,
        address: str,
        after_txid: Optional[str] = None,
        limit: int = 50,
    ) -> tuple[list[Tx], Optional[str]]:
        """Fetch one page of an address's transactions, newest first.

        Returns (transactions, next_cursor). The cursor is None when the page
        is short, i.e. there is nothing more to fetch.

        Note that these payloads already embed the spent output's address under
        `vin[].prevout`, so a single call yields co-spend partners without a
        second request per input. That is what makes backfill affordable.
        """
        path = f"/address/{address}/txs"
        if after_txid:
            path += f"?after_txid={after_txid}"
        payload = self._get(path)
        if not isinstance(payload, list):
            return [], None

        # Prevout heights are not resolved here: the caller is walking history,
        # and each resolution costs a request. Clustering does not need age.
        transactions = [self.parse_tx(item, resolve_prevout_height=False) for item in payload]
        transactions = [tx for tx in transactions if tx is not None]
        next_cursor = transactions[-1].txid if len(transactions) >= limit else None
        return transactions, next_cursor

    def parse_tx(self, payload: dict, resolve_prevout_height: bool = True) -> Optional[Tx]:
        """Build a Tx from a mempool.space transaction payload.

        Single parser for every endpoint. The input/output layout is subtle
        (the spent outpoint lives on `vin`, not inside `prevout`), and having
        two copies of that logic is how the layout bug recurs.
        """
        txid = payload.get("txid")
        if not txid:
            return None

        status = payload.get("status") or {}
        block_height = status.get("block_height") if status.get("confirmed") else None

        inputs: list[TxInput] = []
        for vin in payload.get("vin", []):
            if vin.get("is_coinbase"):
                continue
            # mempool.space puts the spent outpoint on the input itself and
            # the spent output's details under `prevout`.
            spent_txid = vin.get("txid")
            if not spent_txid:
                continue
            prevout = vin.get("prevout")
            if not isinstance(prevout, dict):
                prevout = {}
            spent_height = None
            if resolve_prevout_height:
                spent_height = self.block_height_for_tx(spent_txid)
            inputs.append(
                TxInput(
                    outpoint=Outpoint(spent_txid, int(vin.get("vout", 0))),
                    prevout_value_sats=prevout.get("value"),
                    prevout_address=prevout.get("scriptpubkey_address"),
                    prevout_block_height=spent_height,
                    prevout_script_type=prevout.get("scriptpubkey_type"),
                )
            )

        fee = payload.get("fee")
        vouts = payload.get("vout", [])
        outputs = [int(v.get("value", 0)) for v in vouts]
        addresses = [v.get("scriptpubkey_address") for v in vouts]
        return Tx(
            txid=txid,
            block_height=block_height,
            inputs=inputs,
            output_values_sats=outputs,
            output_addresses=addresses,
            fee_sats=int(fee) if fee is not None else None,
        )
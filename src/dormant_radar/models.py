"""Plain data structures shared by the chain client, detector, and scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Outpoint:
    """A reference to one output of one transaction."""

    txid: str
    vout: int

    def __str__(self) -> str:
        return f"{self.txid}:{self.vout}"


@dataclass
class TxInput:
    """One input of a transaction, with the prevout it spends when known."""

    outpoint: Outpoint
    prevout_value_sats: Optional[int] = None
    prevout_address: Optional[str] = None
    prevout_block_height: Optional[int] = None
    prevout_script_type: Optional[str] = None


@dataclass
class Tx:
    """A transaction relevant to wake-up detection.

    `output_values_sats` and `output_addresses` are parallel arrays indexed by
    output position; callers must set them together. Both are always populated
    from the same `vout` list in `chain.py`, so they stay aligned there.
    """

    txid: str
    block_height: Optional[int]
    inputs: list[TxInput] = field(default_factory=list)
    output_values_sats: list[int] = field(default_factory=list)
    output_addresses: list[Optional[str]] = field(default_factory=list)
    fee_sats: Optional[int] = None

    @property
    def total_output_sats(self) -> int:
        return sum(self.output_values_sats)


@dataclass
class WakeUp:
    """A detected spend of an output that had been dormant for a long time."""

    txid: str
    spend_block_height: int
    spent: Outpoint
    value_sats: int
    address: Optional[str]
    dormant_blocks: int
    dormant_years: float
    script_type: Optional[str]
    observed_at: float
    cause: Optional["CauseScore"] = None
    neural: Optional[dict] = None
    ownership: Optional[dict] = None

    def as_dict(self) -> dict:
        payload = {
            "txid": self.txid,
            "spend_block_height": self.spend_block_height,
            "spent_outpoint": str(self.spent),
            "value_sats": self.value_sats,
            "value_btc": round(self.value_sats / 100_000_000, 8),
            "address": self.address,
            "dormant_blocks": self.dormant_blocks,
            "dormant_years": round(self.dormant_years, 2),
            "script_type": self.script_type,
            "observed_at": self.observed_at,
        }
        if self.cause is not None:
            payload["cause"] = self.cause.as_dict()
        if self.neural is not None:
            payload["neural"] = self.neural
        if self.ownership is not None:
            payload["ownership"] = self.ownership
        return payload


@dataclass
class CauseScore:
    """Probabilistic attribution of *why* a dormant output moved.

    The hypotheses are mutually exclusive and sum to ~1.0. They encode the
    reasoning that on-chain data alone cannot settle: a wake-up is evidence
    *against* permanent loss, but holding and structural causes remain possible.
    """

    hypothesis: str
    confidence: float
    distribution: dict[str, float]
    rationale: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "confidence": round(self.confidence, 3),
            "distribution": {k: round(v, 3) for k, v in self.distribution.items()},
            "rationale": self.rationale,
        }
"""Cause attribution for dormant wallet wake-ups.

On-chain data cannot tell us *why* a wallet sat idle. What it can do is
give evidence that shifts probabilities between a small set of mutually
exclusive hypotheses. This module encodes that reasoning as transparent
log-odds updates, so the score is inspectable rather than a black box.

Hypotheses
----------
lost
    Keys were believed unrecoverable (misplaced, discarded media, death
    without an inheritance plan). A spend is strong evidence *against*
    this, so it starts improbable and each protective signal pushes it down.
holding
    The owner deliberately did not move the coins (long-term conviction,
    cold storage, a treasury). Positive evidence: activity coincides with a
    high price or a network upgrade, or the owner keeps custody by sending
    value back to an address they control.
structural
    The coins were not really someone's to spend in the ordinary sense:
    exchange or custodian wallets consolidating, seized/frozen balances,
    multi-sig where co-signers were previously unreachable, burn-adjacent
    or unspendable scripts finally swept.

The age of the output also carries information. Very long dormancy (the
tens-of-thousands-of-blocks, pre-2013 vintages) points at loss or
structural holdings; recent dormancy is mostly ordinary holding behavior.
"""

from __future__ import annotations

import math
from typing import Optional

from .cluster import OwnershipSummary
from .detector import blocks_to_years
from .models import CauseScore, Tx, WakeUp

HYPOTHESES = ("lost", "holding", "structural")

# Prior in log-odds space. A wallet that moved after years of silence is
# more likely to have been held on purpose than to have been dead and
# revived, so loss starts behind the others.
PRIOR = {"lost": -1.0, "holding": 0.4, "structural": -0.2}

# Age bands judged against this many blocks. These mirror common on-chain
# analysis conveniences rather than any authoritative source.
ONE_YEAR = 52_560
FOUR_YEARS = 4 * ONE_YEAR
TEN_YEARS = 10 * ONE_YEAR

# A price this many times the trailing average suggests a holder selling
# into strength rather than a mere internal move.
HIGH_PRICE_MULTIPLE = 2.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / total for k, v in scores.items()}


def score_cause(
    wakeup: WakeUp,
    tx: Optional[Tx] = None,
    price_multiple: Optional[float] = None,
    network_upgrade_window: bool = False,
    ownership: Optional[OwnershipSummary] = None,
) -> CauseScore:
    """Attribute a probable cause to a single wake-up.

    Parameters
    ----------
    wakeup:
        The detected dormant-output spend.
    tx:
        The spending transaction, used to look for self-transfers and
        consolidation patterns. Optional; omitting it only removes evidence.
    price_multiple:
        Current price divided by a trailing average (for example the 200-day
        mean). A value well above 1 means the market is hot.
    network_upgrade_window:
        True when the spend falls in a window around a protocol upgrade,
        which historically coincides with deliberate movement.
    ownership:
        Cluster-derived ownership facts. When `known` is False the weaker
        same-address check is used instead, and the rationale says so.
    """
    log_odds = dict(PRIOR)
    rationale: list[str] = []

    years = wakeup.dormant_years

    # -- evidence 1: the output actually moved -------------------------
    log_odds["lost"] -= 2.0
    log_odds["holding"] += 0.3
    log_odds["structural"] += 0.2
    rationale.append(
        "The output was spent, which is direct evidence the key was accessible "
        "at spend time and weakens the recovery-loss hypothesis."
    )

    # -- evidence 2: dormancy length -----------------------------------
    if years >= 10:
        # Pre-2014 vintages: loss was very likely, and holders who move this
        # late are often estates or custodians rather than conviction buyers.
        log_odds["structural"] += 0.8
        log_odds["holding"] += 0.3
        rationale.append(
            f"Dormant about {years:.1f} years; vintages this old skew toward "
            "long-lost or custodial balances rather than active conviction."
        )
    elif years >= 4:
        log_odds["holding"] += 0.4
        rationale.append(
            f"Dormant about {years:.1f} years, the range typical of deliberate "
            "long-term holding."
        )
    else:
        log_odds["holding"] += 0.2
        rationale.append(
            f"Dormant about {years:.1f} years; age alone is weak evidence here."
        )

    # -- evidence 3: market context ------------------------------------
    if price_multiple is not None:
        if price_multiple >= HIGH_PRICE_MULTIPLE:
            log_odds["holding"] += 0.7
            log_odds["lost"] -= 0.2
            rationale.append(
                f"Price is {price_multiple:.1f}x its trailing average, a "
                "classic window for long-term holders to sell into strength."
            )
        elif price_multiple <= 0.7:
            log_odds["holding"] -= 0.2
            log_odds["structural"] += 0.2
            rationale.append(
                f"Price is {price_multiple:.1f}x its trailing average, which "
                "argues against a profit-taking move."
            )

    # -- evidence 4: upgrade window ------------------------------------
    if network_upgrade_window:
        log_odds["holding"] += 0.3
        rationale.append(
            "The spend lands near a protocol upgrade, when owners often move "
            "coins deliberately."
        )

    # -- evidence 5: ownership, from clustering when available ---------
    if ownership is not None and ownership.known:
        if ownership.is_self_transfer:
            share = ownership.owned_output_share
            # Stronger than the same-address check because the value may
            # return to a different address owned by the same cluster.
            boost = 1.4 if share >= 0.99 else 0.7
            log_odds["holding"] += boost
            log_odds["structural"] -= 0.3
            rationale.append(
                f"{share:.0%} of output value returns to addresses in the same "
                f"ownership cluster ({ownership.cluster_size} addresses), so "
                "custody is preserved rather than sold."
            )
        else:
            rationale.append(
                "No output returns to the originating ownership cluster, which "
                "is consistent with a genuine disposal."
            )
        if ownership.cluster_size >= 50:
            log_odds["structural"] += 0.9
            rationale.append(
                f"The cluster spans {ownership.cluster_size} addresses, large "
                "enough to suggest a custodial or institutional wallet."
            )
        elif ownership.cluster_size >= 15:
            log_odds["structural"] += 0.4
            rationale.append(
                f"The cluster spans {ownership.cluster_size} addresses, larger "
                "than typical individual use."
            )
    elif tx is not None and wakeup.address:
        # Fallback: only same-address matches are visible without clustering.
        recipients = [a for a in tx.output_addresses if a]
        if recipients and all(a == wakeup.address for a in recipients):
            log_odds["holding"] += 1.2
            log_odds["structural"] -= 0.3
            rationale.append(
                "Every output returns to the originating address, the signature "
                "of a self-transfer that preserves custody (same-address match "
                "only; ownership clustering unavailable)."
            )
        elif wakeup.address in recipients:
            log_odds["holding"] += 0.5
            rationale.append(
                "Part of the value returns to the originating address, "
                "consistent with change from a deliberate move."
            )

    # -- evidence 6: consolidation shape --------------------------------
    if tx is not None and len(tx.inputs) >= 10:
        log_odds["structural"] += 0.8
        rationale.append(
            f"The transaction sweeps {len(tx.inputs)} inputs, which is "
            "typical of custodial consolidation."
        )

    distribution = _normalise({k: _sigmoid(v) for k, v in log_odds.items()})
    best = max(distribution, key=distribution.get)
    spread = sorted(distribution.values(), reverse=True)
    confidence = spread[0] - spread[1] if len(spread) > 1 else spread[0]

    return CauseScore(
        hypothesis=best,
        confidence=confidence,
        distribution=distribution,
        rationale=rationale,
    )
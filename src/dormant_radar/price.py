"""Market context used by the scorer.

Only the pieces the cause model needs: a current price and a trailing
average to judge whether the market is hot. CoinGecko's public endpoint
needs no key. Failures degrade to None rather than breaking a scan.
"""

from __future__ import annotations

import logging
import statistics
from typing import Optional

import requests

logger = logging.getLogger(__name__)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class PriceContext:
    def __init__(self, base_url: str = COINGECKO_BASE, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def market_chart(self, days: int = 200, vs_currency: str = "usd") -> list[float]:
        try:
            response = requests.get(
                f"{self.base_url}/coins/bitcoin/market_chart",
                params={"vs_currency": vs_currency, "days": days},
                timeout=self.timeout,
                headers={"User-Agent": "dormant-radar/0.1"},
            )
            response.raise_for_status()
            prices = response.json().get("prices", [])
            return [float(p[1]) for p in prices]
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("price lookup failed: %s", exc)
            return []

    def price_multiple(self, trailing_days: int = 200) -> Optional[float]:
        """Current price divided by its trailing mean, or None if unavailable."""
        series = self.market_chart(days=trailing_days)
        if len(series) < 2:
            return None
        trailing_mean = statistics.fmean(series)
        if trailing_mean <= 0:
            return None
        return series[-1] / trailing_mean
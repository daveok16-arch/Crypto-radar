"""Dormant wallet wake-up radar for Bitcoin.

Detects spends of long-aged transaction outputs from public chain data and
scores the most likely reason (lost, deliberately held, or structural).
"""

__version__ = "0.1.0"
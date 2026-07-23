"""Shadow monitor and dynamic challenger for JuniAuto.

Re-exports the public API used by main.py and the decision cycle.
See docs/knowledge-base/part5-shadow-and-replay.md §2.41-§2.42.
"""
from __future__ import annotations

from juniauto.shadow.challenger import DynamicChallenger
from juniauto.shadow.monitor import ShadowMonitor, ShadowStats

__all__ = [
    "ShadowMonitor",
    "ShadowStats",
    "DynamicChallenger",
]

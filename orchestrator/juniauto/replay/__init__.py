"""Self-test replay proof harness (§3.5).

Determinism contract: given the same stored inputs (features + posterior + market
state + config), the pipeline must emit the same set of actions. Any drift is a
regression the harness catches before the trading day starts.
"""
from juniauto.replay.harness import ReplayHarness, ReplayResult

__all__ = ["ReplayHarness", "ReplayResult"]

"""Trailing stop management — Chandelier + posterior-conditional hybrid.

Design decisions (2026-08-04):

    * Broker-side DAY stop-market orders on all held positions (Alpaca
      supports stop-market on fractional shares with time_in_force=DAY).
    * Level = max(chandelier_floor, posterior_conditional_price, prior_stop) —
      ratchets up only, never loosens (§2.31 trailing drift).
    * Chandelier: HWM - k_chandelier * ATR_20  (industry-standard k=3).
    * Posterior-conditional: close_t * (1 - k_vol * daily_vol_pct), where
      k_vol switches between k_loose_vol (conservative_edge >= 0) and
      k_tight_vol (conservative_edge < 0). Ties the stop to the Bayesian's
      current view — when the model turns adverse, the stop tightens.
    * Refresh cadence: 09:45 phantom cycle (hysteresis-triggered REPLACE)
      and 15:55 live cycle (fresh DAY-stop submit for next session).
    * Re-entry after a stop-out: EV-hurdle bump added to
      minimum_hurdle_bps in _evaluate_gateway. Decays exponentially with
      halflife = bump_halflife_sessions.
    * Rollout gate: `canary_symbols` in StopsConfig. Empty list = shadow
      mode (compute + persist active_stops rows, but do NOT submit to
      Alpaca). Populate with an allowlist to enable submits gradually.

Spec anchoring:
    * §2.6 conservative_edge = mu - zq * sigma_total drives the tight/loose
      switch in the posterior-conditional component.
    * §2.26 minimum_required_edge_bps is where the hurdle bump plugs in.
    * §2.31 target-drift trailing property preserved via the max() ratchet.
    * §2.44 no non-economic gates: the hurdle bump is an EV buffer, not a
      block. A strong enough model signal overwhelms the bump and re-enters.
    * §3.1 PDT: never submit a stop on entry day (would fire as a day trade).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, TYPE_CHECKING

from juniauto.config import StopsConfig
from juniauto.monitoring import metrics as m
from juniauto.utils import get_logger
from juniauto.utils.time_utils import ET, to_et, trading_days_between

if TYPE_CHECKING:
    from juniauto.data.alpaca_feed import AlpacaFeed, Bar
    from juniauto.db import QuestDBClient
    from juniauto.execution.pdt import PDTTracker

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure-function level computation (tested in isolation from IO)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StopLevel:
    """One recomputed stop level. Feeds active_stops persistence + Alpaca submit."""
    symbol: str
    stop_price: float
    high_water_mark: float
    chandelier_component: float
    posterior_component: float
    method: str                       # 'hybrid' | 'chandelier' | 'posterior' | 'flat_prior'
    atr_20: float
    daily_vol_pct: float
    posterior_edge_bps: float
    posterior_sigma_bps: float
    conservative_edge_bps: float
    qty: float


def compute_atr(bars: Iterable["Bar"], lookback: int = 20) -> float:
    """Wilder's ATR over the last `lookback` daily bars.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    Returns the simple mean of the last `lookback` TR values. First bar
    contributes only high-low (no prev_close).
    """
    seq = list(bars)
    if len(seq) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(seq)):
        prev_close = float(seq[i - 1].close)
        h = float(seq[i].high)
        low_ = float(seq[i].low)
        tr = max(h - low_, abs(h - prev_close), abs(low_ - prev_close))
        trs.append(tr)
    tail = trs[-lookback:]
    if not tail:
        return 0.0
    return sum(tail) / len(tail)


def compute_level(
    *,
    cfg: StopsConfig,
    symbol: str,
    qty: float,
    entry_price: float,
    current_close: float,
    prior_stop_price: float | None,
    prior_high_water_mark: float | None,
    atr_20: float,
    daily_vol_pct: float,
    posterior_edge_bps: float,
    posterior_sigma_bps: float,
    zq: float = 1.0,
) -> StopLevel:
    """Pure-function level computation. Callable from tests without IO.

    Ratchet property: the returned stop_price is guaranteed >= prior_stop_price
    when both components stay above the prior. If both new components fall
    below the prior stop, we keep the prior (trailing never loosens).
    """
    # High-water mark ratchets up on every recompute — never resets.
    seed_hwm = max(entry_price, current_close)
    high_water_mark = max(seed_hwm, prior_high_water_mark or 0.0)

    # Component 1 — Chandelier floor
    chandelier_price = high_water_mark - cfg.k_chandelier * atr_20

    # Component 2 — Posterior-conditional (spec-native EV accelerator)
    conservative_edge_bps = posterior_edge_bps - zq * posterior_sigma_bps
    if conservative_edge_bps >= 0.0:
        k_vol = cfg.k_loose_vol
    else:
        k_vol = cfg.k_tight_vol
    posterior_price = current_close * max(0.0, 1.0 - k_vol * max(0.0, daily_vol_pct))

    # Hybrid — tightest (highest) of the two, but never below prior stop.
    candidates: list[float] = [chandelier_price, posterior_price]
    if prior_stop_price is not None and prior_stop_price > 0.0:
        candidates.append(prior_stop_price)
    stop_price = max(candidates)

    # Method attribution (which component actually set the level)
    if prior_stop_price is not None and stop_price == prior_stop_price and stop_price > max(chandelier_price, posterior_price):
        method = "flat_prior"
    elif chandelier_price >= posterior_price:
        method = "chandelier"
    else:
        method = "posterior"
    # If both new components clear prior and both are in play, call it hybrid.
    if (prior_stop_price is None or stop_price > (prior_stop_price or 0.0)) and abs(chandelier_price - posterior_price) < 1e-6:
        method = "hybrid"

    return StopLevel(
        symbol=symbol,
        stop_price=stop_price,
        high_water_mark=high_water_mark,
        chandelier_component=chandelier_price,
        posterior_component=posterior_price,
        method=method,
        atr_20=atr_20,
        daily_vol_pct=daily_vol_pct,
        posterior_edge_bps=posterior_edge_bps,
        posterior_sigma_bps=posterior_sigma_bps,
        conservative_edge_bps=conservative_edge_bps,
        qty=qty,
    )


# Alpaca broker constraints on stop-order prices:
#   * Stocks priced >= $1.00: stop must be in whole-penny increments ($0.01).
#   * Stocks priced <  $1.00: stop must be in sub-penny increments ($0.0001).
#   * For SELL stops: stop_price must be STRICTLY LESS THAN the current price.
# Violations return error code 42210000. See:
#   https://docs.alpaca.markets/docs/orders-at-alpaca#order-types
_TICK_STOCK = 0.01
_TICK_SUB_DOLLAR = 0.0001
# Minimum gap between the submitted stop and current market. Absorbs
# intraday jitter so a random bid tick doesn't fire the stop immediately
# after submit. 25 bps = 0.25% below market — tight enough that a name
# already breaking down still gets a live stop, wide enough that ordinary
# spread noise doesn't trigger it. Not currently a config knob (revisit
# after 30 days of live data).
_MIN_STOP_OFFSET_BPS = 25.0


def snap_to_broker_tick(
    computed_stop: float,
    current_price: float,
    side: str = "sell",
) -> tuple[float, str]:
    """Round the computed stop to Alpaca's minimum tick and enforce the
    below-market constraint for SELL stops.

    Returns ``(adjusted_stop, note)`` where ``note`` is:
        ""                       — no adjustment needed
        "penny_floor"            — floored to nearest tick
        "capped_below_market"    — pulled down to min offset below current
        "too_close_to_market"    — market so close no valid stop exists;
                                   caller should skip this submission

    For side="sell" (the only case we currently use), the returned stop is
    guaranteed to be strictly less than current_price.
    """
    if computed_stop <= 0 or current_price <= 0:
        return 0.0, "too_close_to_market"
    tick = _TICK_STOCK if current_price >= 1.0 else _TICK_SUB_DOLLAR

    if side != "sell":
        # Buy stops (not used by TrailingStopManager today) — ceiling to tick.
        adjusted = math.ceil(computed_stop / tick) * tick
        return round(adjusted, 4), ""

    # Enforce below-market cap: stop must be <= current * (1 - min_offset).
    max_allowed = current_price * (1.0 - _MIN_STOP_OFFSET_BPS / 10_000.0)
    was_capped = computed_stop > max_allowed
    capped = min(computed_stop, max_allowed)

    # Floor to broker tick (rounds DOWN, so stop is at or below `capped`).
    adjusted = math.floor(capped / tick) * tick
    # Guard against float representation drift (e.g. 386.05000000000001).
    decimals = 4 if tick == _TICK_SUB_DOLLAR else 2
    adjusted = round(adjusted, decimals)

    # Final sanity: if floor pushed us to zero (or below) the position is
    # priced too low for any valid offset. Skip the submission.
    if adjusted <= 0 or adjusted >= current_price:
        return 0.0, "too_close_to_market"

    if was_capped:
        return adjusted, "capped_below_market"
    if abs(adjusted - computed_stop) >= tick / 2.0:
        return adjusted, "penny_floor"
    return adjusted, ""


def should_replace(
    *,
    cfg: StopsConfig,
    new_stop: float,
    current_broker_stop: float,
    atr_20: float,
) -> bool:
    """Hysteresis check at 09:45. REPLACE only if:

    1. new_stop is HIGHER than current (trailing — never loosen), AND
    2. |delta| exceeds max(pct-of-price, fraction-of-ATR) threshold.
    """
    if current_broker_stop <= 0.0:
        return False
    if new_stop <= current_broker_stop:
        return False   # never loosen
    delta = new_stop - current_broker_stop
    pct_threshold = cfg.min_replace_delta_pct * current_broker_stop
    atr_threshold = cfg.min_replace_delta_atr_frac * max(0.0, atr_20)
    return delta > max(pct_threshold, atr_threshold)


# ---------------------------------------------------------------------------
# Manager — wires computation to IO (QuestDB + Alpaca + PDT + Prometheus)
# ---------------------------------------------------------------------------

class TrailingStopManager:
    """Owns per-position stop lifecycle across scan cycles.

    See module docstring for the design overview. Public methods called by
    main.py:

        manage_cycle(cycle_type, ..., execute=True)
            — recompute levels for all held positions; submit or REPLACE at
              Alpaca where the canary filter allows.
        reconcile_triggered_stops(since, now)
            — called by _resolution_loop to detect stop fills, write
              stop_triggers rows, register EV-hurdle penalties.
        get_hurdle_bump_bps(symbol, now)
            — called by _evaluate_gateway to bump minimum_hurdle_bps.
    """

    def __init__(
        self,
        cfg: StopsConfig,
        db: "QuestDBClient",
        alpaca: "AlpacaFeed",
        pdt: "PDTTracker",
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._alpaca = alpaca
        self._pdt = pdt
        # emit shadow-mode gauge once on init (updated at each cycle too)
        m.stops_shadow_mode_gauge.set(1.0 if self._is_shadow_mode() else 0.0)

    # ---- Rollout gating ----
    def _is_shadow_mode(self) -> bool:
        return not self._cfg.canary_symbols

    def is_canary(self, symbol: str) -> bool:
        """Return True if `symbol` is allowed for real broker-side submits."""
        if not self._cfg.enabled:
            return False
        if not self._cfg.canary_symbols:
            return False
        if "*" in self._cfg.canary_symbols:
            return True
        return symbol in self._cfg.canary_symbols

    # ---- EV-hurdle bump ----
    def get_hurdle_bump_bps(self, symbol: str, now: datetime) -> float:
        """Decaying bump added to minimum_hurdle_bps for a symbol that
        recently stopped out. Called by _evaluate_gateway.

        Returns 0.0 when no active penalty exists or when the penalty has
        decayed below cfg.bump_deactivate_threshold_bps.
        """
        try:
            row = self._db.query_one(
                """
                SELECT ts, initial_bump_bps, halflife_sessions
                  FROM stop_penalties
                 WHERE symbol = %s AND active = true
                 ORDER BY ts DESC
                 LIMIT 1
                """,
                (symbol,),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("hurdle_bump_query_failed", symbol=symbol, error=str(e))
            return 0.0
        if not row:
            return 0.0
        ts, initial_bump, halflife = row
        if ts is None or initial_bump is None or halflife is None:
            return 0.0
        try:
            pen_dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        except (TypeError, ValueError):
            return 0.0
        sessions_since = trading_days_between(to_et(pen_dt).date(), to_et(now).date())
        if sessions_since < 0:
            return 0.0
        bump = float(initial_bump) * (0.5 ** (sessions_since / float(halflife)))
        return max(0.0, bump)

    def _register_penalty(
        self,
        *,
        symbol: str,
        now: datetime,
        realized_return_bps: float,
    ) -> None:
        """Insert a stop_penalties row so future _evaluate_gateway calls see
        the bump. Called from reconcile_triggered_stops on each detected fill.
        """
        with self._db.sender() as s:
            s.row(
                "stop_penalties",
                symbols={"symbol": symbol},
                columns={
                    "initial_bump_bps": float(self._cfg.bump_initial_bps),
                    "halflife_sessions": float(self._cfg.bump_halflife_sessions),
                    "triggered_return_bps": float(realized_return_bps),
                    "active": True,
                },
                at=now,
            )

    # ---- Cycle management (main entry point from _daily_decision_cycle) ----
    def manage_cycle(
        self,
        *,
        cycle_type: str,             # '0945' | '1230' | '1555'
        positions: list[dict],       # from alpaca.get_positions()
        bars_by_symbol: dict[str, list],
        features_by_symbol: dict[str, dict],
        predictions_by_symbol: dict[str, dict],
        now: datetime,
        zq: float = 1.0,
    ) -> dict[str, int]:
        """Recompute stop levels for every held position and reconcile with
        Alpaca (submit / REPLACE / no-op) per the canary filter.

        Returns a summary dict: {'computed', 'submitted', 'replaced',
        'skipped_pdt', 'skipped_shadow', 'canceled', 'errors'}.
        """
        summary = {
            "computed": 0, "submitted": 0, "replaced": 0,
            "skipped_pdt": 0, "skipped_shadow": 0, "canceled": 0, "errors": 0,
        }
        if not self._cfg.enabled:
            log.info("stops_disabled_skip_cycle", cycle_type=cycle_type)
            return summary

        # Load prior state — HWM + last stop_price per symbol.
        prior_state = self._load_prior_state()
        # Load standing broker-side stops.
        broker_stops = self._load_broker_stops()
        # Position entry timestamps from executions.
        entry_ts_by_symbol = self._load_entry_timestamps([p["symbol"] for p in positions])

        active_count = 0
        shadow_count = 0
        distances_bps: list[float] = []
        submits_since_pause = 0

        for pos in positions:
            symbol = pos["symbol"]
            qty = float(pos.get("qty_available", pos["qty"]))
            if qty <= 0.0:
                continue

            # PDT / min-hold gate — skip on entry day when configured
            entry_ts = entry_ts_by_symbol.get(symbol)
            if entry_ts is not None and self._cfg.entry_day_exempt:
                if self._pdt.will_stop_be_day_trade(symbol, entry_ts, now):
                    summary["skipped_pdt"] += 1
                    m.stops_skipped_pdt_total.labels(reason="entry_day").inc()
                    log.info("stop_skip_entry_day", symbol=symbol, entry_ts=entry_ts.isoformat())
                    continue

            # Fetch inputs for the level computation
            bars = bars_by_symbol.get(symbol) or []
            if len(bars) < 2:
                summary["errors"] += 1
                log.info("stop_skip_no_bars", symbol=symbol)
                continue
            current_close = float(bars[-1].close)
            if current_close <= 0.0:
                summary["errors"] += 1
                continue
            atr_20 = compute_atr(bars, lookback=self._cfg.atr_lookback_days)
            feat = features_by_symbol.get(symbol) or {}
            realized_vol_bps = float(feat.get("realized_vol_bps") or 0.0)
            # Annualized bps -> daily fraction: bps/10000 / sqrt(252).
            daily_vol_pct = 0.0
            if realized_vol_bps > 0.0:
                daily_vol_pct = (realized_vol_bps / 10_000.0) / math.sqrt(252.0)
            pred = predictions_by_symbol.get(symbol) or {}
            posterior_edge_bps = float(pred.get("mu_edge_bps") or 0.0)
            posterior_sigma_bps = float(pred.get("sigma_total_bps") or 0.0)

            entry_price = float(pos.get("avg_entry_price") or current_close)
            prior = prior_state.get(symbol) or {}
            prior_stop = prior.get("stop_price")
            prior_hwm = prior.get("high_water_mark")

            level = compute_level(
                cfg=self._cfg,
                symbol=symbol,
                qty=qty,
                entry_price=entry_price,
                current_close=current_close,
                prior_stop_price=prior_stop,
                prior_high_water_mark=prior_hwm,
                atr_20=atr_20,
                daily_vol_pct=daily_vol_pct,
                posterior_edge_bps=posterior_edge_bps,
                posterior_sigma_bps=posterior_sigma_bps,
                zq=zq,
            )
            summary["computed"] += 1
            distances_bps.append(10_000.0 * (current_close - level.stop_price) / current_close)

            in_canary = self.is_canary(symbol)
            existing_broker = broker_stops.get(symbol)
            alpaca_order_id: str | None = None
            submitted_to_broker = False
            # Broker-adjusted stop (penny-floored + capped below market). This is
            # what actually goes to Alpaca AND what we persist to active_stops,
            # so audit rows reflect reality rather than the theoretical level.
            broker_stop = level.stop_price
            broker_note = ""

            if not in_canary:
                # Shadow mode for this symbol — persist only, do NOT submit.
                shadow_count += 1
                summary["skipped_shadow"] += 1
            else:
                # Snap computed level to a valid Alpaca tick + below-market cap.
                # Handles both sub-penny rejection (42210000 "sub-penny increment")
                # and above-market rejection (42210000 "stop price must be less
                # than current price").
                broker_stop, broker_note = snap_to_broker_tick(
                    computed_stop=level.stop_price,
                    current_price=current_close,
                    side="sell",
                )
                if broker_note == "too_close_to_market":
                    summary["skipped_market_too_close"] = summary.get("skipped_market_too_close", 0) + 1
                    m.stops_skipped_market_too_close_total.inc()
                    log.info(
                        "stop_skip_market_too_close", symbol=symbol,
                        computed_stop=round(level.stop_price, 4),
                        current_close=round(current_close, 4),
                    )
                    # Persist the un-submitted row so we have audit visibility.
                    self._persist_active_stop(
                        symbol=symbol, level=level, entry_ts=entry_ts,
                        entry_price=entry_price, alpaca_order_id=None,
                        cycle_type=cycle_type, in_canary=True,
                        submitted_to_broker=False, now=now,
                        adjusted_stop_price=level.stop_price,
                    )
                    continue

                if broker_note:
                    log.info(
                        "stop_price_adjusted", symbol=symbol,
                        computed=round(level.stop_price, 4),
                        adjusted=round(broker_stop, 4),
                        reason=broker_note, current_close=round(current_close, 4),
                    )

                # 15:55 = fresh DAY submit (prior DAY expired at 16:00 unless still standing)
                # 09:45/12:30 = hysteresis REPLACE only where needed
                try:
                    if cycle_type == "1555":
                        # cancel any leftover from today's session (shouldn't exist post-expiry
                        # but paranoia — Alpaca sometimes leaves rejected/expired orders visible)
                        if existing_broker is not None:
                            try:
                                self._alpaca.cancel_order(existing_broker["id"])
                                m.stops_cancel_events_total.labels(reason="pre_1555_refresh").inc()
                            except Exception as ce:  # noqa: BLE001
                                log.warning("stop_precycle_cancel_failed", symbol=symbol, error=str(ce))
                        alpaca_order_id = self._alpaca.submit_stop_market(
                            symbol=symbol, qty=qty, stop_price=broker_stop, side="sell",
                        )
                        submitted_to_broker = True
                        summary["submitted"] += 1
                        m.stops_submit_events_total.labels(cycle_type=cycle_type).inc()
                    else:
                        # 09:45 or 12:30: hysteresis REPLACE
                        if existing_broker is None:
                            # No standing stop (shouldn't normally happen post-1555 submit,
                            # but a fresh entry today with entry_day_exempt=false could land here)
                            alpaca_order_id = self._alpaca.submit_stop_market(
                                symbol=symbol, qty=qty, stop_price=broker_stop, side="sell",
                            )
                            submitted_to_broker = True
                            summary["submitted"] += 1
                            m.stops_submit_events_total.labels(cycle_type=cycle_type).inc()
                        else:
                            cur = float(existing_broker.get("stop_price") or 0.0)
                            if should_replace(cfg=self._cfg, new_stop=broker_stop,
                                              current_broker_stop=cur, atr_20=atr_20):
                                try:
                                    self._alpaca.cancel_order(existing_broker["id"])
                                except Exception as ce:  # noqa: BLE001
                                    log.warning("stop_replace_cancel_failed", symbol=symbol, error=str(ce))
                                alpaca_order_id = self._alpaca.submit_stop_market(
                                    symbol=symbol, qty=qty, stop_price=broker_stop, side="sell",
                                )
                                submitted_to_broker = True
                                summary["replaced"] += 1
                                m.stops_replace_events_total.labels(reason="hysteresis").inc()
                            else:
                                alpaca_order_id = existing_broker["id"]
                                submitted_to_broker = True   # broker still holds a stop
                except Exception as e:  # noqa: BLE001
                    log.error("stop_submit_error", symbol=symbol, error=str(e),
                              error_type=type(e).__name__)
                    summary["errors"] += 1
                    submitted_to_broker = False

                # Rate-limit pacing — Alpaca ~200 req/min
                submits_since_pause += 1
                if submits_since_pause >= self._cfg.submit_batch_size:
                    time.sleep(self._cfg.submit_pause_ms / 1000.0)
                    submits_since_pause = 0

            if submitted_to_broker:
                active_count += 1

            self._persist_active_stop(
                symbol=symbol, level=level, entry_ts=entry_ts,
                entry_price=entry_price, alpaca_order_id=alpaca_order_id,
                cycle_type=cycle_type, in_canary=in_canary,
                submitted_to_broker=submitted_to_broker, now=now,
                adjusted_stop_price=broker_stop,
            )

        # Emit metrics for this cycle
        m.stops_active_count_gauge.labels(cycle_type=cycle_type).set(active_count)
        m.stops_shadow_count_gauge.labels(cycle_type=cycle_type).set(shadow_count)
        if distances_bps:
            m.stops_avg_distance_bps_gauge.labels(cycle_type=cycle_type).set(
                sum(distances_bps) / len(distances_bps)
            )
        m.stops_shadow_mode_gauge.set(1.0 if self._is_shadow_mode() else 0.0)
        # Active hurdle bumps count
        try:
            row = self._db.query_one(
                "SELECT count(*) FROM stop_penalties WHERE active = true"
            )
            if row and row[0] is not None:
                m.stops_hurdle_bumps_active_gauge.set(int(row[0]))
        except Exception:  # noqa: BLE001 — best-effort
            pass

        log.info(
            "stops_manage_cycle_summary",
            cycle_type=cycle_type,
            **summary,
            active=active_count,
            shadow=shadow_count,
            in_canary=(not self._is_shadow_mode()),
        )
        return summary

    # ---- Reconciliation (called from _resolution_loop) ----
    def reconcile_triggered_stops(self, since: datetime, now: datetime) -> int:
        """Query Alpaca for filled stop orders since `since`. For each:
            - Insert a stop_triggers row (audit).
            - Register a stop_penalties row (EV-hurdle bump).
            - Mark the active_stops row triggered=true.
            - Emit stops_triggered_total metric.

        Does NOT insert into `executions` — the standard SELL flow through
        OrderManager.route already handles that. Stop fills come in via
        Alpaca's fill webhook / order status; they're the same event.

        Returns the count of newly-reconciled triggers.
        """
        try:
            fills = self._alpaca.list_recent_stop_fills(since=since)
        except Exception as e:  # noqa: BLE001
            log.warning("stops_reconcile_fetch_failed", error=str(e))
            return 0

        n_reconciled = 0
        for fill in fills:
            symbol = fill["symbol"]
            oid = fill["id"]
            # De-dup: don't register the same fill twice
            try:
                seen = self._db.query_one(
                    "SELECT 1 FROM stop_triggers WHERE alpaca_order_id = %s LIMIT 1",
                    (oid,),
                )
            except Exception:  # noqa: BLE001
                seen = None
            if seen:
                continue

            # Look up entry price from position history for realized_return
            entry_row = None
            try:
                entry_row = self._db.query_one(
                    """
                    SELECT MIN(ts), avg(fill_price)
                      FROM executions
                     WHERE symbol = %s AND side = 'buy'
                    """,
                    (symbol,),
                )
            except Exception:  # noqa: BLE001
                pass
            entry_ts = entry_row[0] if entry_row and entry_row[0] else None
            entry_price = float(entry_row[1]) if entry_row and entry_row[1] else 0.0

            fill_price = float(fill.get("fill_price") or 0.0)
            stop_price = float(fill.get("stop_price") or 0.0)
            realized_return_bps = 0.0
            if entry_price > 0.0 and fill_price > 0.0:
                realized_return_bps = 10_000.0 * (fill_price - entry_price) / entry_price
            slippage_vs_stop_bps = 0.0
            if stop_price > 0.0 and fill_price > 0.0:
                slippage_vs_stop_bps = 10_000.0 * (fill_price - stop_price) / stop_price
            holding_days = 0
            if entry_ts is not None:
                try:
                    e_dt = entry_ts if isinstance(entry_ts, datetime) else datetime.fromisoformat(str(entry_ts))
                    holding_days = trading_days_between(to_et(e_dt).date(), to_et(now).date())
                except (TypeError, ValueError):
                    pass

            with self._db.sender() as s:
                s.row(
                    "stop_triggers",
                    symbols={"symbol": symbol},
                    columns={
                        "entry_ts": entry_ts if entry_ts is not None else now,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "fill_price": fill_price,
                        "slippage_vs_stop_bps": slippage_vs_stop_bps,
                        "realized_return_bps": realized_return_bps,
                        "qty": float(fill.get("qty") or 0.0),
                        "holding_days": int(holding_days),
                        "alpaca_order_id": oid,
                    },
                    at=fill.get("filled_at") or now,
                )
            self._register_penalty(
                symbol=symbol, now=now, realized_return_bps=realized_return_bps,
            )
            m.stops_triggered_total.inc()
            n_reconciled += 1
            log.info(
                "stop_triggered",
                symbol=symbol, stop_price=stop_price, fill_price=fill_price,
                realized_return_bps=round(realized_return_bps, 1),
                slippage_vs_stop_bps=round(slippage_vs_stop_bps, 1),
                holding_days=holding_days,
            )

        # Decay penalty rows that have fallen below the deactivation threshold.
        self._deactivate_expired_penalties(now)
        return n_reconciled

    def _deactivate_expired_penalties(self, now: datetime) -> None:
        """Mark stop_penalties rows inactive when the decayed bump falls
        below cfg.bump_deactivate_threshold_bps. Prevents unbounded active
        row growth over months."""
        try:
            rows = self._db.query(
                """
                SELECT symbol, ts, initial_bump_bps, halflife_sessions
                  FROM stop_penalties
                 WHERE active = true
                """,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("penalty_decay_query_failed", error=str(e))
            return
        expired: list[str] = []
        for symbol, ts, initial_bump, halflife in rows:
            try:
                pen_dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
            except (TypeError, ValueError):
                continue
            sessions = trading_days_between(to_et(pen_dt).date(), to_et(now).date())
            if sessions < 0:
                continue
            decayed = float(initial_bump) * (0.5 ** (sessions / float(halflife)))
            if decayed < self._cfg.bump_deactivate_threshold_bps:
                expired.append(symbol)
        for symbol in expired:
            # QuestDB update via INSERT with active=false — active_stops-style
            # append. The gauge query already filters by active=true from
            # newest so the newer row wins.
            with self._db.sender() as s:
                s.row(
                    "stop_penalties",
                    symbols={"symbol": symbol},
                    columns={
                        "initial_bump_bps": 0.0,
                        "halflife_sessions": 1.0,
                        "triggered_return_bps": 0.0,
                        "active": False,
                    },
                    at=now,
                )

    # ---- IO helpers ----
    def _load_prior_state(self) -> dict[str, dict[str, float]]:
        """Latest row per symbol from active_stops -> {stop_price, high_water_mark}.

        Uses a windowed LATEST BY-style query.
        """
        out: dict[str, dict[str, float]] = {}
        try:
            rows = self._db.query(
                """
                SELECT symbol, current_stop_price, high_water_mark
                  FROM active_stops
                 LATEST ON ts PARTITION BY symbol
                """,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("stops_prior_state_query_failed", error=str(e))
            return out
        for symbol, stop_price, hwm in rows:
            out[symbol] = {
                "stop_price": float(stop_price) if stop_price is not None else 0.0,
                "high_water_mark": float(hwm) if hwm is not None else 0.0,
            }
        return out

    def _load_broker_stops(self) -> dict[str, dict]:
        """symbol -> {id, stop_price, qty} for standing broker-side stops."""
        try:
            stops = self._alpaca.list_open_stops()
        except Exception as e:  # noqa: BLE001
            log.warning("stops_broker_query_failed", error=str(e))
            return {}
        # Alpaca could theoretically hold multiple stops per symbol; take newest.
        out: dict[str, dict] = {}
        for s in stops:
            sym = s["symbol"]
            prev = out.get(sym)
            if prev is None:
                out[sym] = s
            else:
                # Keep the most recently submitted one
                p_ts = prev.get("submitted_at") or datetime.min.replace(tzinfo=timezone.utc)
                c_ts = s.get("submitted_at") or datetime.min.replace(tzinfo=timezone.utc)
                if c_ts > p_ts:
                    out[sym] = s
        return out

    def _load_entry_timestamps(self, symbols: list[str]) -> dict[str, datetime]:
        """Latest BUY execution ts per symbol. Conservative for the entry-day
        exemption: a partial-add today makes the whole position 'today's entry'
        for stop-management purposes, deferring the stop to tomorrow. That's
        the safe direction — an over-eager stop on same-day-entry shares
        would fire as a day trade."""
        if not symbols:
            return {}
        out: dict[str, datetime] = {}
        try:
            rows = self._db.query(
                "SELECT symbol, MAX(ts) FROM executions WHERE side = 'buy' GROUP BY symbol",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("stops_entry_ts_query_failed", error=str(e))
            return {}
        wanted = set(symbols)
        for sym, ts in rows:
            if sym not in wanted or ts is None:
                continue
            try:
                out[sym] = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
            except (TypeError, ValueError):
                continue
        return out

    def _persist_active_stop(
        self,
        *,
        symbol: str,
        level: StopLevel,
        entry_ts: datetime | None,
        entry_price: float,
        alpaca_order_id: str | None,
        cycle_type: str,
        in_canary: bool,
        submitted_to_broker: bool,
        now: datetime,
        adjusted_stop_price: float | None = None,
    ) -> None:
        # Persist the ADJUSTED (broker-tick, below-market-capped) stop price
        # so audit rows reflect what Alpaca actually holds. Fall back to the
        # theoretical level.stop_price when no adjustment happened (shadow mode).
        persisted_stop = float(adjusted_stop_price) if adjusted_stop_price is not None else float(level.stop_price)
        with self._db.sender() as s:
            columns: dict[str, object] = {
                "entry_ts": entry_ts if entry_ts is not None else now,
                "entry_price": float(entry_price),
                "high_water_mark": float(level.high_water_mark),
                "current_stop_price": persisted_stop,
                "chandelier_component": float(level.chandelier_component),
                "posterior_component": float(level.posterior_component),
                "posterior_edge_bps": float(level.posterior_edge_bps),
                "posterior_sigma_bps": float(level.posterior_sigma_bps),
                "conservative_edge_bps": float(level.conservative_edge_bps),
                "atr_20": float(level.atr_20),
                "daily_vol_pct": float(level.daily_vol_pct),
                "qty": float(level.qty),
                "active": bool(submitted_to_broker),
                "triggered": False,
                "in_canary": bool(in_canary),
                "submitted_to_broker": bool(submitted_to_broker),
            }
            if alpaca_order_id is not None:
                columns["alpaca_order_id"] = str(alpaca_order_id)
            s.row(
                "active_stops",
                symbols={
                    "symbol": symbol,
                    "cycle_type": cycle_type,
                    "method": level.method,
                },
                columns=columns,
                at=now,
            )

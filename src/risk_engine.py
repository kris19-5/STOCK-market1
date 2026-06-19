"""
risk_engine.py
--------------
Layer 3 of the DecodeLabs systematic engine: RISK MANAGEMENT
(mechanical circuit breakers for behavioral discipline)

Since this backtester operates on DAILY bars (not intraday ticks), the
three circuit breakers from the methodology are implemented at the
trade/session level, which is the closest faithful mapping of an
intraday discipline system onto a daily-bar backtest:

  1. 15-Minute Hard Stop (intraday concept)
     -> Mapped here as: once daily realized loss reaches 50% of the
        daily loss limit, NO NEW ENTRIES are allowed for the rest of
        that session (the "step away from the charts" cool-off).

  2. Half-Size Rule
     -> After 2 consecutive losing trades, position size for the next
        trade is halved. Size is restored to full only after a winning
        trade resets the consecutive-loss counter.

  3. Session End Trigger (Hard Lock)
     -> If daily realized loss reaches 70% of the daily loss limit,
        the engine force-closes any open position and blocks all
        further trading for that day.

The RiskEngine is stateful across the bar-by-bar simulation loop in
backtest_engine.py: it tracks daily P&L, consecutive losses, and
whether trading is currently locked.
"""

from dataclasses import dataclass, field


@dataclass
class RiskEngine:
    capital: float
    daily_loss_limit_pct: float = 0.02      # e.g. 2% of capital per day
    half_size_after_losses: int = 2
    hard_stop_trigger_pct: float = 0.5       # 50% of daily limit -> stop new entries
    session_lock_trigger_pct: float = 0.7    # 70% of daily limit -> hard lock + flatten

    # --- internal state (reset daily) ---
    current_day: object = None
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trading_paused_today: bool = False   # hit the 15-min hard stop equivalent
    session_locked_today: bool = False   # hit the 70% hard lock

    # --- stats for reporting ---
    hard_stop_events: int = field(default_factory=lambda: 0)
    session_lock_events: int = field(default_factory=lambda: 0)
    half_size_trades: int = field(default_factory=lambda: 0)

    def _daily_loss_limit_value(self) -> float:
        return self.capital * self.daily_loss_limit_pct

    def new_day(self, day) -> None:
        """Call once per simulated day before evaluating any signals."""
        if self.current_day != day:
            self.current_day = day
            self.daily_pnl = 0.0
            self.trading_paused_today = False
            self.session_locked_today = False

    def can_enter_new_trade(self) -> bool:
        """
        Gate for new entries. Blocks entries if either:
          - the 15-min-hard-stop-equivalent has triggered today, or
          - the session is hard-locked today.
        """
        return not (self.trading_paused_today or self.session_locked_today)

    def position_size_multiplier(self) -> float:
        """
        Half-Size Rule: after `half_size_after_losses` consecutive losses,
        new trades are sized at 50%. Restored to 1.0 after the next win.
        """
        if self.consecutive_losses >= self.half_size_after_losses:
            self.half_size_trades += 1
            return 0.5
        return 1.0

    def register_trade_close(self, pnl: float) -> None:
        """
        Call when a trade closes (realized P&L for that trade).
        Updates daily P&L, consecutive loss streak, and evaluates
        whether circuit breakers should now trigger.
        """
        self.daily_pnl += pnl

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        limit_value = self._daily_loss_limit_value()
        if limit_value <= 0:
            return

        loss_fraction_of_limit = (-self.daily_pnl) / limit_value if self.daily_pnl < 0 else 0.0

        # 15-Minute Hard Stop equivalent: pause new entries for the rest of the day
        if loss_fraction_of_limit >= self.hard_stop_trigger_pct and not self.trading_paused_today:
            self.trading_paused_today = True
            self.hard_stop_events += 1

        # Session End Trigger: hard lock + force flat for the rest of the day
        if loss_fraction_of_limit >= self.session_lock_trigger_pct and not self.session_locked_today:
            self.session_locked_today = True
            self.session_lock_events += 1

    def must_force_close(self) -> bool:
        """If session is locked, any open position must be force-closed."""
        return self.session_locked_today

"""
backtest_engine.py
-------------------
Bar-by-bar simulation that wires together:
  Layer 1 (Inputs)        -> raw OHLCV
  Layer 2 (Process)       -> indicators.py + signals.py
  Layer 3 (Risk Mgmt)     -> risk_engine.RiskEngine

For each symbol, the engine walks the daily bars in order and:
  - lets the RiskEngine know about the new day
  - if currently flat and Long_Entry fires AND RiskEngine permits new
    entries -> open a long position, sized by position_size_multiplier()
  - if currently in a position and (Long_Exit fires OR stop-loss/take-
    profit hit OR RiskEngine forces a flatten) -> close the position
    and report the realized P&L back to the RiskEngine

A simple fixed-fractional position sizing model is used: each full-size
trade risks `risk_per_trade_pct` of capital, sized by distance to the
stop-loss (ATR-free version: uses a fixed stop_loss_pct of entry price
for simplicity and transparency).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from indicators import build_feature_frame
from signals import generate_signals
from risk_engine import RiskEngine


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop_loss: float
    take_profit: float
    size_multiplier: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: pd.Series
    trades: list
    risk_engine: RiskEngine
    feature_frame: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    initial_capital: float = 100_000.0,
    risk_per_trade_pct: float = 0.01,
    stop_loss_pct: float = 0.03,
    take_profit_pct: float = 0.06,
    daily_loss_limit_pct: float = 0.02,
    strict_confluence: bool = True,
    ema_fast: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    sr_window: int = 20,
    wick_threshold: float = 1.2,
) -> BacktestResult:
    """
    df must have columns Open, High, Low, Close, Volume and a DatetimeIndex,
    sorted ascending by date, for a single symbol.
    """
    feat = build_feature_frame(
        df, ema_fast=ema_fast, ema_slow=ema_slow, rsi_period=rsi_period, sr_window=sr_window
    )
    feat = generate_signals(
        feat, wick_threshold=wick_threshold, strict_confluence=strict_confluence
    )

    risk = RiskEngine(capital=initial_capital, daily_loss_limit_pct=daily_loss_limit_pct)

    cash = initial_capital
    position: Optional[Trade] = None
    trades: list[Trade] = []
    equity_curve = []

    for date, row in feat.iterrows():
        risk.new_day(date.date())

        # Mark-to-market equity for the curve (before any action this bar)
        if position is not None:
            unrealized = (row["Close"] - position.entry_price) * position.shares
            mtm_equity = cash + unrealized
        else:
            mtm_equity = cash
        equity_curve.append((date, mtm_equity))

        # --- Manage open position: check exits first ---
        if position is not None:
            exit_price = None
            exit_reason = None

            if risk.must_force_close():
                exit_price = row["Close"]
                exit_reason = "RISK_HARD_LOCK"
            elif row["Low"] <= position.stop_loss:
                exit_price = position.stop_loss
                exit_reason = "STOP_LOSS"
            elif row["High"] >= position.take_profit:
                exit_price = position.take_profit
                exit_reason = "TAKE_PROFIT"
            elif bool(row["Long_Exit"]):
                exit_price = row["Close"]
                exit_reason = "SIGNAL_EXIT"

            if exit_price is not None:
                pnl = (exit_price - position.entry_price) * position.shares
                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                position.pnl = pnl
                position.pnl_pct = (exit_price / position.entry_price - 1.0) * 100
                cash += pnl
                trades.append(position)
                risk.register_trade_close(pnl)
                position = None

        # --- Consider new entry (only if flat) ---
        if position is None and bool(row["Long_Entry"]) and risk.can_enter_new_trade():
            entry_price = row["Close"]
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)

            size_mult = risk.position_size_multiplier()
            risk_amount = cash * risk_per_trade_pct * size_mult
            per_share_risk = entry_price - stop_loss
            shares = risk_amount / per_share_risk if per_share_risk > 0 else 0

            if shares > 0:
                position = Trade(
                    symbol=symbol,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    size_multiplier=size_mult,
                )

    # Force-close any still-open position at the last bar for clean reporting
    if position is not None:
        last_row = feat.iloc[-1]
        exit_price = last_row["Close"]
        pnl = (exit_price - position.entry_price) * position.shares
        position.exit_date = feat.index[-1]
        position.exit_price = exit_price
        position.exit_reason = "END_OF_BACKTEST"
        position.pnl = pnl
        position.pnl_pct = (exit_price / position.entry_price - 1.0) * 100
        cash += pnl
        trades.append(position)
        risk.register_trade_close(pnl)

    equity_series = pd.Series(
        data=[v for _, v in equity_curve],
        index=[d for d, _ in equity_curve],
        name=f"{symbol}_equity",
    )

    return BacktestResult(
        symbol=symbol,
        equity_curve=equity_series,
        trades=trades,
        risk_engine=risk,
        feature_frame=feat,
    )


def summarize_trades(trades: list) -> dict:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_pnl_pct": np.nan,
            "total_pnl": 0.0,
            "max_win_pct": np.nan,
            "max_loss_pct": np.nan,
        }

    pnl_pcts = [t.pnl_pct for t in trades]
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnl_pcts if p > 0]

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_pnl_pct": float(np.mean(pnl_pcts)),
        "total_pnl": float(np.sum(pnls)),
        "max_win_pct": float(np.max(pnl_pcts)),
        "max_loss_pct": float(np.min(pnl_pcts)),
    }

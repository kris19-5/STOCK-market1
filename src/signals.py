"""
signals.py
----------
Converts the filter stack (indicators.py) into discrete entry/exit signals
using the confluence rules from the DecodeLabs methodology:

  LONG ENTRY requires ALL of:
    1. Trend filter   : EMA-50 > EMA-200 (Golden Cross regime, i.e. BULL)
    2. Momentum filter: RSI > 50 (bullish confirmation)
    3. Structural filter: price near rolling Support (reversal zone)
    4. Microstructure filter: Wick-to-Body Ratio exceeds threshold on a
       bullish rejection candle (lower wick dominant), i.e. a hammer-like
       rejection at support

  EXIT (for an open long) on ANY of:
    - Death Cross (EMA_fast crosses below EMA_slow)
    - RSI falls back below 45 (momentum failure, small buffer below 50
      to avoid noise-driven whipsaw exits)
    - Price tags rolling Resistance (structural target reached)
    - Stop-loss / take-profit hit (handled in the backtest engine itself,
      since it is position-aware)
"""

import pandas as pd


def generate_signals(
    feat: pd.DataFrame,
    rsi_bull_threshold: float = 50.0,
    rsi_exit_threshold: float = 45.0,
    wick_threshold: float = 1.2,
    strict_confluence: bool = True,
) -> pd.DataFrame:
    """
    feat: output of indicators.build_feature_frame
    strict_confluence:
        True  -> ALL four filters must align (Trend + Momentum + Structure + Wick)
        False -> simpler EMA regime + RSI only (looser, more trades)

    Returns feat with two added boolean columns: Long_Entry, Long_Exit
    """
    out = feat.copy()

    trend_ok = out["Trend_Regime"] == "BULL"
    momentum_ok = out["RSI"] > rsi_bull_threshold

    if strict_confluence:
        structure_ok = out["Near_Support"]
        # Bullish rejection candle: lower wick is the dominant wick AND
        # R_wb clears the threshold (long lower wick relative to body)
        wick_ok = (out["LowerWick"] > out["UpperWick"]) & (out["R_wb"] >= wick_threshold)
        out["Long_Entry"] = trend_ok & momentum_ok & structure_ok & wick_ok
    else:
        out["Long_Entry"] = trend_ok & momentum_ok

    exit_trend = out["Death_Cross"]
    exit_momentum = out["RSI"] < rsi_exit_threshold
    exit_structure = out["Near_Resistance"]

    out["Long_Exit"] = exit_trend | exit_momentum | exit_structure

    return out

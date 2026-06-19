"""
indicators.py
--------------
Layer 2 of the DecodeLabs systematic engine: PROCESS (FILTERS)

Computes the structural / trend / momentum filters described in the
Technical Analysis & Price Action methodology:

  - EMA-50 / EMA-200 (trend filter, Golden Cross / Death Cross)
  - RSI-14 (momentum filter, bullish confirmation requires RSI > 50)
  - Rolling Support / Resistance (structural boundaries)
  - Wick-to-Body Ratio, R_wb (candlestick microstructure / reversal signal)

All functions take and return pandas Series/DataFrames so they compose
cleanly inside the backtest engine.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Classic Wilder RSI.
    RSI > 50 is used downstream as the bullish momentum confirmation filter.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (no down days), RSI -> 100
    rsi_val = rsi_val.where(avg_loss != 0, 100.0)
    return rsi_val


def rolling_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Structural boundaries: simple rolling-window swing low / swing high,
    used as a proxy for Support and Resistance zones.

    Support  = rolling min of Low over `window` bars (excluding current bar)
    Resistance = rolling max of High over `window` bars (excluding current bar)
    """
    support = df["Low"].rolling(window=window, min_periods=window).min().shift(1)
    resistance = df["High"].rolling(window=window, min_periods=window).max().shift(1)
    return pd.DataFrame({"Support": support, "Resistance": resistance})


def wick_to_body_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candlestick microstructure: Wick-to-Body Ratio (R_wb).

    Body       = |Close - Open|
    UpperWick  = High - max(Open, Close)
    LowerWick  = min(Open, Close) - Low
    R_wb       = max(UpperWick, LowerWick) / Body   (NaN-safe, large when body ~ 0)

    A large R_wb on a candle that touches a Support/Resistance zone is read
    as a potential rejection / reversal signal (e.g. hammer, shooting star).
    """
    body = (df["Close"] - df["Open"]).abs()
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]

    # Avoid divide-by-zero: floor body at a tiny epsilon relative to price
    safe_body = body.replace(0, np.nan)
    r_wb = pd.concat([upper_wick, lower_wick], axis=1).max(axis=1) / safe_body
    r_wb = r_wb.fillna(r_wb.max(skipna=True) if r_wb.notna().any() else 0)

    return pd.DataFrame(
        {
            "Body": body,
            "UpperWick": upper_wick,
            "LowerWick": lower_wick,
            "R_wb": r_wb,
        }
    )


def build_feature_frame(
    df: pd.DataFrame,
    ema_fast: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    sr_window: int = 20,
) -> pd.DataFrame:
    """
    Assembles the full filter stack on top of raw OHLCV.
    Expects df with columns: Open, High, Low, Close, Volume and a DatetimeIndex.
    """
    out = df.copy()

    out["EMA_fast"] = ema(out["Close"], ema_fast)
    out["EMA_slow"] = ema(out["Close"], ema_slow)
    out["RSI"] = rsi(out["Close"], rsi_period)

    sr = rolling_support_resistance(out, window=sr_window)
    out = out.join(sr)

    wick = wick_to_body_ratio(out)
    out = out.join(wick)

    # Trend regime: Golden Cross (fast > slow) vs Death Cross (fast < slow)
    out["Trend_Regime"] = np.where(out["EMA_fast"] > out["EMA_slow"], "BULL", "BEAR")

    # Cross events (regime transition points)
    fast_above = out["EMA_fast"] > out["EMA_slow"]
    out["Golden_Cross"] = fast_above & (~fast_above.shift(1).fillna(False))
    out["Death_Cross"] = (~fast_above) & (fast_above.shift(1).fillna(False))

    # Proximity to structural support/resistance (within 3% of the rolling level)
    out["Near_Support"] = (out["Low"] <= out["Support"] * 1.03) & out["Support"].notna()
    out["Near_Resistance"] = (out["High"] >= out["Resistance"] * 0.97) & out["Resistance"].notna()

    return out

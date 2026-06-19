"""
data_loader.py
---------------
Layer 1 of the DecodeLabs systematic engine: INPUTS

Fetches daily OHLCV data for NSE-listed symbols via yfinance
(ticker format: "BEL.NS", "CANBK.NS", etc.).

If yfinance cannot reach the network (e.g. sandboxed environments
without internet access), this module transparently falls back to a
realistic SIMULATED OHLCV series so the rest of the engine can still be
tested end-to-end. The simulated path is clearly flagged in the
returned DataFrame's `.attrs["source"]` so you always know which data
you're looking at.

On your own machine (with normal internet access) this will pull real
historical NSE data with no changes needed.
"""

from typing import Optional

import numpy as np
import pandas as pd


def _simulate_ohlcv(
    symbol: str,
    periods: int = 750,
    start_price: float = 500.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.28,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a realistic-looking daily OHLCV series using GBM for the
    close price, then derives open/high/low from intraday noise, plus a
    volume series with mild autocorrelation. NOT real market data --
    used only as a network-outage fallback so the engine remains testable.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))

    dt = 1 / 252
    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    vol = annual_vol * np.sqrt(dt)

    log_returns = rng.normal(drift, vol, periods)
    close = start_price * np.exp(np.cumsum(log_returns))

    # Derive Open as previous close with small gap, High/Low from intraday range
    open_ = np.empty(periods)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.003, periods - 1))

    intraday_range = np.abs(rng.normal(0, vol * 0.6, periods)) * close
    high = np.maximum(open_, close) + intraday_range * rng.uniform(0.2, 1.0, periods)
    low = np.minimum(open_, close) - intraday_range * rng.uniform(0.2, 1.0, periods)
    low = np.maximum(low, 1.0)  # guard against non-positive prices

    base_volume = rng.integers(500_000, 3_000_000)
    volume = np.abs(rng.normal(base_volume, base_volume * 0.3, periods)).astype(int)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.attrs["source"] = "SIMULATED"
    return df


def load_ohlcv(
    symbol: str,
    period: str = "3y",
    interval: str = "1d",
    simulate_on_failure: bool = True,
    sim_start_price: float = 500.0,
) -> pd.DataFrame:
    """
    symbol: NSE ticker WITHOUT suffix, e.g. "BEL" (the ".NS" suffix is
            added automatically for yfinance).
    period: yfinance period string, e.g. "1y", "3y", "5y", "max".

    Returns a DataFrame indexed by date with columns
    Open, High, Low, Close, Volume, and df.attrs["source"] set to
    either "yfinance" or "SIMULATED".
    """
    yf_symbol = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        import yfinance as yf

        raw = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no data (network blocked or invalid ticker)")

        # yfinance sometimes returns MultiIndex columns for a single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.attrs["source"] = "yfinance"
        return df

    except Exception as e:
        if not simulate_on_failure:
            raise
        print(
            f"[data_loader] WARNING: live fetch failed for {yf_symbol} "
            f"({e}). Falling back to SIMULATED data for engine testing."
        )
        periods = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(period, 756)
        return _simulate_ohlcv(symbol, periods=periods, start_price=sim_start_price)


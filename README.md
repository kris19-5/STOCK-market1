# stock-market1

A systematic, rules-based backtesting engine for NSE (Indian stock market) equities, built around a three-layer trading methodology: **Inputs → Process (Filters) → Risk Management**.

This repo implements and backtests the "Technical Analysis & Price Action" framework — a disciplined alternative to reactive, discretionary trading, combining trend/momentum/structural filters with mechanical behavioral circuit breakers.

> **Status:** personal project / private use. Not financial advice. See [Disclaimer](#disclaimer).

---

## What this does

For any NSE-listed stock, the engine:

1. Pulls daily OHLCV price data (`src/data_loader.py`)
2. Computes a stack of technical filters — EMA-50/200 trend regime, RSI-14 momentum, rolling Support/Resistance, and Wick-to-Body Ratio candlestick microstructure (`src/indicators.py`)
3. Generates long entry/exit signals only when filters align in confluence (`src/signals.py`)
4. Runs every trade through a stateful risk engine that mirrors real circuit breakers — a hard-stop after 50% of the daily loss limit, a half-size rule after 2 consecutive losses, and a full session lock at 70% of the daily loss limit (`src/risk_engine.py`)
5. Simulates the full equity curve bar-by-bar and reports trade-level results (`src/backtest_engine.py`, orchestrated by `src/run_backtest.py`)

## Methodology summary

| Layer | Component | Rule |
|---|---|---|
| Inputs | OHLCV | Daily candlestick data + volume only — no discretionary override |
| Process | Trend | EMA-50 > EMA-200 required for long entries (Golden Cross regime) |
| Process | Momentum | RSI-14 > 50 required to confirm bullish signals |
| Process | Structure | Price must be near rolling Support for a long setup |
| Process | Microstructure | Wick-to-Body Ratio (R_wb) confirms a rejection candle at the structural zone |
| Risk | 15-Min Hard Stop | New entries blocked for the rest of the session at 50% of daily loss limit |
| Risk | Half-Size Rule | Position size halved after 2 consecutive losing trades |
| Risk | Session End Trigger | Hard lock + force-close at 70% of daily loss limit |

Full methodology writeup (with worked examples and the formula for R_wb) is maintained separately as a Word document, outside this repo.

## Repo structure

```
stock-market1/
├── src/
│   ├── data_loader.py      # Layer 1 — fetch OHLCV (yfinance, NSE tickers)
│   ├── indicators.py        # Layer 2 — EMA, RSI, Support/Resistance, R_wb
│   ├── signals.py           # Layer 2 — confluence logic → entry/exit signals
│   ├── risk_engine.py       # Layer 3 — stateful circuit breakers
│   ├── backtest_engine.py   # Simulation loop wiring all layers together
│   └── run_backtest.py      # Main entry point — edit config here, run this
├── output/                  # Generated on each run (gitignored — see below)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/stock-market1.git
cd stock-market1
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python run_backtest.py
```

This runs the engine across the configured watchlist over the configured lookback period and writes to `../output/`:

- `backtest_summary.csv` — per-symbol trade count, win rate, P&L, and how often each circuit breaker fired
- `equity_curves.png` — normalized equity curves for all symbols
- `trade_log.txt` — every individual trade with entry/exit/reason

## Configuration

All tunable parameters live at the top of `src/run_backtest.py`:

```python
WATCHLIST = ["BEL", "CANBK", "IREDA", "ITC", "MAZDOCK", "SBIN", "RELIANCE", "TCS"]
PERIOD = "3y"
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE_PCT = 0.01      # 1% of capital risked per full-size trade
STOP_LOSS_PCT = 0.03           # 3% stop-loss from entry
TAKE_PROFIT_PCT = 0.06         # 6% take-profit from entry (2:1 reward:risk)
DAILY_LOSS_LIMIT_PCT = 0.02    # 2% of capital = one day's loss limit
STRICT_CONFLUENCE = True       # All 4 filters required; False = looser EMA+RSI only
```

`STRICT_CONFLUENCE = True` is highly selective by design — it will generate few trades per symbol, since it requires trend + momentum + structure + a confirming wick all to align. Set it to `False` to test signal frequency and to stress-test the risk engine's circuit breakers (which only engage meaningfully with enough trade volume to produce loss clusters within a session).

## Important notes

- **Data source:** real NSE OHLCV data is pulled via `yfinance` (ticker format `SYMBOL.NS`). If the network is unavailable (e.g. in a sandboxed environment), `data_loader.py` transparently falls back to a clearly-flagged simulated price series so the engine logic remains testable end-to-end — check `df.attrs["source"]` to confirm which you're looking at.
- **Circuit breaker timing:** the 15-minute hard stop is conceptually an intraday rule. Since this backtester runs on daily bars, it's mapped to "block new entries for the rest of the trading day" once triggered — the closest faithful translation onto end-of-day data. A true intraday version would need an intraday NSE data feed (not reliably available for free; paid options like Kite Connect or TrueData would be required).
- **Position sizing:** fixed-fractional — each full-size trade risks `RISK_PER_TRADE_PCT` of capital, sized by distance to the stop-loss. The half-size rule cuts this to half after two consecutive losses.

## Disclaimer

This project is for personal research and educational purposes only. It is **not** financial advice, and past backtest performance — especially on simulated data — is not indicative of future results. Live trading carries risk of loss. Verify everything independently before risking real capital.

## License

All rights reserved — private use only. Not licensed for redistribution.

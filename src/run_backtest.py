"""
run_backtest.py
-----------------
Main entry point. Runs the full systematic engine
(Inputs -> Process/Filters -> Risk Management) across a watchlist of
NSE symbols, and prints / saves a summary report with equity curves.

USAGE (from the repo root, with normal internet access):
    pip install -r requirements.txt
    cd src
    python run_backtest.py

Output is written to ../output/ (relative to this file).

On a sandboxed/offline machine, this will automatically fall back to
simulated OHLCV data per symbol (clearly labeled in the output) so you
can still verify the engine logic end-to-end.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_ohlcv
from backtest_engine import run_backtest, summarize_trades

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# CONFIG -- edit this section to change the watchlist or parameters
# ---------------------------------------------------------------------

WATCHLIST = [
    "BEL",      # Kris's holding
    "CANBK",    # Kris's holding
    "IREDA",    # Kris's holding
    "ITC",      # Kris's holding
    "MAZDOCK",  # Kris's holding
    "SBIN",     # Kris's holding (benchmark-ish, large liquid PSU bank)
    "RELIANCE", # liquid benchmark
    "TCS",      # liquid benchmark
]

PERIOD = "3y"
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE_PCT = 0.01      # 1% of capital risked per trade (full size)
STOP_LOSS_PCT = 0.03           # 3% stop-loss from entry
TAKE_PROFIT_PCT = 0.06         # 6% take-profit from entry (2:1 reward:risk)
DAILY_LOSS_LIMIT_PCT = 0.02    # 2% of capital = one day's loss limit
STRICT_CONFLUENCE = True       # All 4 filters required (set False for looser EMA+RSI only)

# ---------------------------------------------------------------------


def main():
    all_results = {}
    summary_rows = []

    for symbol in WATCHLIST:
        df = load_ohlcv(symbol, period=PERIOD)
        source = df.attrs.get("source", "unknown")

        result = run_backtest(
            df,
            symbol=symbol,
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade_pct=RISK_PER_TRADE_PCT,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            daily_loss_limit_pct=DAILY_LOSS_LIMIT_PCT,
            strict_confluence=STRICT_CONFLUENCE,
        )
        all_results[symbol] = result

        stats = summarize_trades(result.trades)
        final_equity = result.equity_curve.iloc[-1] if len(result.equity_curve) else INITIAL_CAPITAL
        total_return_pct = (final_equity / INITIAL_CAPITAL - 1) * 100

        summary_rows.append(
            {
                "Symbol": symbol,
                "Data Source": source,
                "Trades": stats["num_trades"],
                "Win Rate %": round(stats["win_rate"], 1) if stats["num_trades"] else 0,
                "Avg P&L % / Trade": round(stats["avg_pnl_pct"], 2) if stats["num_trades"] else 0,
                "Total P&L (Rs)": round(stats["total_pnl"], 0),
                "Total Return %": round(total_return_pct, 2),
                "Hard-Stop Events": result.risk_engine.hard_stop_events,
                "Session-Lock Events": result.risk_engine.session_lock_events,
                "Half-Size Trades": result.risk_engine.half_size_trades,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("DECODELABS SYSTEMATIC ENGINE -- BACKTEST SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_summary.csv"), index=False)

    # --- Plot equity curves for all symbols on one chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, result in all_results.items():
        if len(result.equity_curve):
            normalized = result.equity_curve / result.equity_curve.iloc[0] * 100
            ax.plot(normalized.index, normalized.values, label=symbol, linewidth=1.5)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("DecodeLabs Systematic Engine -- Equity Curves (Normalized to 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (Indexed)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "equity_curves.png"), dpi=150)
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'equity_curves.png')}")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'backtest_summary.csv')}")

    # --- Per-symbol trade logs ---
    with open(os.path.join(OUTPUT_DIR, "trade_log.txt"), "w") as f:
        for symbol, result in all_results.items():
            f.write(f"\n{'='*80}\n{symbol} -- Trade Log\n{'='*80}\n")
            if not result.trades:
                f.write("No trades generated under current filter confluence.\n")
                continue
            for t in result.trades:
                f.write(
                    f"{t.entry_date.date()} BUY @ {t.entry_price:.2f} "
                    f"(size x{t.size_multiplier}) -> "
                    f"{t.exit_date.date()} SELL @ {t.exit_price:.2f} "
                    f"[{t.exit_reason}] P&L: {t.pnl_pct:+.2f}% (Rs {t.pnl:+.0f})\n"
                )
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'trade_log.txt')}")

    return summary_df, all_results


if __name__ == "__main__":
    main()

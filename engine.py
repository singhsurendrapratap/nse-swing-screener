"""
engine.py -- core strategy logic (trend + breakout + volume swing strategy)
Shared by both the CLI script and the Streamlit web app, so there's exactly
one source of truth for the rules.

Strategy recap:
  - Only trade WITH the trend (price > SMA50 > SMA200)
  - Only trade WHEN the broader market (Nifty 50) is itself in an uptrend
    (regime filter -- avoids fighting a falling market)
  - Entry trigger: breakout above the prior 20-day high, confirmed by volume
    >= 1.5x the 20-day average
  - Execution is realistic: signal is confirmed on a day's CLOSE, but the
    trade is only actually entered at the NEXT day's OPEN (you can't buy at
    a close price you only know after the close) -- this avoids lookahead bias.
  - Friction (brokerage + STT + slippage) is applied on both entry and exit.
  - Stop-loss / target are ATR-based (volatility-adjusted per stock), not a
    flat percentage.
  - Position size is derived from a fixed % of capital you're willing to
    risk per trade, not a guess.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

BENCHMARK_TICKER = "^NSEI"

DEFAULT_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "JSWSTEEL.NS", "HCLTECH.NS", "TECHM.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS",
    "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "BRITANNIA.NS",
    "DIVISLAB.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "COALINDIA.NS", "BPCL.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "TATACONSUM.NS",
]

# ----------------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Guard: yfinance sometimes returns a partial/incomplete bar for "today" if
    # fetched while the market is open, or a stale row with NaN OHLC. Drop any
    # such rows so they can't corrupt a signal or an exit calculation.
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)
    df["VolAvg20"] = df["Volume"].rolling(20).mean().shift(1)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Volatility contraction (VCP-lite): is the recent 10-day trading range
    # tighter than its own typical 60-day range? A genuine squeeze before a
    # breakout (institutional accumulation) vs. a random spike out of nowhere.
    # FIX: shift(1) so this measures the range BEFORE today, excluding the
    # breakout day itself (which is expansive by definition -- including it
    # was contradicting the very thing this filter is supposed to require).
    range10_raw = df["High"].rolling(10).max() - df["Low"].rolling(10).min()
    df["Range10"] = (range10_raw / df["Close"]).shift(1)
    df["Range10Avg60"] = df["Range10"].rolling(60).mean()

    return df


def get_nifty_close(start_date: str) -> pd.Series:
    """Raw Nifty 50 close series, used for relative-strength comparisons."""
    nifty = yf.download(BENCHMARK_TICKER, start=start_date, auto_adjust=True, progress=False)
    if nifty.empty:
        return pd.Series(dtype=float)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty = nifty.xs(BENCHMARK_TICKER, level=1, axis=1)
    return nifty["Close"]


def add_relative_strength(df: pd.DataFrame, nifty_close: pd.Series, lookback: int = 63) -> pd.DataFrame:
    """
    RS_Diff = stock's % return over `lookback` days minus Nifty's % return over
    the same window. Positive means the stock has been outperforming the index
    (CANSLIM-style relative strength) -- not just moving with market noise.
    """
    df = df.copy()
    if nifty_close.empty:
        df["RS_Diff"] = np.nan
        return df
    aligned = nifty_close.reindex(df.index, method="ffill")
    stock_ret = df["Close"].pct_change(lookback)
    nifty_ret = aligned.pct_change(lookback)
    df["RS_Diff"] = stock_ret - nifty_ret
    return df


def get_market_regime(start_date: str) -> pd.Series:
    """True for dates where Nifty 50 closes above its own 50-day SMA."""
    nifty = yf.download(BENCHMARK_TICKER, start=start_date, auto_adjust=True, progress=False)
    if nifty.empty:
        return pd.Series(dtype=bool)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty = nifty.xs(BENCHMARK_TICKER, level=1, axis=1)
    sma50 = nifty["Close"].rolling(50).mean()
    return (nifty["Close"] > sma50).rename("bullish")


def signal_condition(row, market_ok: bool, params: dict) -> bool:
    base = (
        market_ok
        and row["Close"] > row["SMA50"] > row["SMA200"]
        and row["Close"] > row["High20"]
        and row["Volume"] >= params["volume_mult"] * row["VolAvg20"]
        and params["rsi_low"] <= row["RSI14"] <= params["rsi_high"]
    )
    if not base:
        return False

    if params.get("require_contraction"):
        r10, r60 = row.get("Range10"), row.get("Range10Avg60")
        if pd.isna(r10) or pd.isna(r60):
            return False
        if r10 > r60 * params.get("contraction_threshold", 0.75):
            return False  # range isn't tight enough -- not a real squeeze

    if params.get("require_rs"):
        rs_diff = row.get("RS_Diff")
        if pd.isna(rs_diff) or rs_diff <= 0:
            return False  # stock hasn't been outperforming Nifty

    return True

# ----------------------------------------------------------------------------------
# BACKTEST
# ----------------------------------------------------------------------------------

def backtest_symbol(df: pd.DataFrame, market_regime: pd.Series, params: dict) -> list:
    trades = []
    df = df.reset_index()
    i = 0
    while i < len(df) - 2:
        row = df.iloc[i]
        date = row["Date"]
        mkt_ok = bool(market_regime.get(date, True))

        if pd.notna(row.get("SMA200")) and signal_condition(row, mkt_ok, params):
            entry_day = df.iloc[i + 1]
            entry_price = entry_day["Open"] * (1 + params["friction_pct"])
            atr = row["ATR14"]
            if pd.isna(atr) or atr <= 0:
                i += 1
                continue

            stop = entry_price - params["atr_stop_mult"] * atr
            target = entry_price + params["atr_target_mult"] * atr

            outcome, exit_price, days_held = None, None, 0
            exit_index = i + 1
            for j in range(i + 1, min(i + 1 + params["hold_days"], len(df))):
                day = df.iloc[j]
                days_held += 1
                exit_index = j
                if day["Low"] <= stop:
                    outcome, exit_price = "loss", stop * (1 - params["friction_pct"])
                    break
                if day["High"] >= target:
                    outcome, exit_price = "win", target * (1 - params["friction_pct"])
                    break
            if outcome is None:
                exit_index = min(i + params["hold_days"], len(df) - 1)
                exit_price = df.iloc[exit_index]["Close"] * (1 - params["friction_pct"])
                outcome = "win" if exit_price > entry_price else "loss"

            risk_per_share = entry_price - stop  # FIX: consistent with actual stop used
            r_multiple = (exit_price - entry_price) / risk_per_share if risk_per_share > 0 else 0

            trades.append({
                "symbol": None,  # filled by caller
                "entry_date": entry_day["Date"],
                "outcome": outcome,
                "r_multiple": r_multiple,
                "pct_return": (exit_price - entry_price) / entry_price * 100,
                "days_held": days_held,
            })
            i = exit_index + 1  # FIX: resume right after THIS trade closes, not after the max-hold cap
        else:
            i += 1
    return trades


def backtest_symbol_trailing(df: pd.DataFrame, market_regime: pd.Series, params: dict) -> list:
    """
    Same entry rules as backtest_symbol, but exits differ: no fixed target.
    Instead, the stop trails behind the highest close seen since entry,
    always ATR-adjusted and only ever moving up. Exit only when price
    falls back and hits that trailing stop, or the max hold cap is reached.
    """
    trades = []
    df = df.reset_index()
    i = 0
    while i < len(df) - 2:
        row = df.iloc[i]
        date = row["Date"]
        mkt_ok = bool(market_regime.get(date, True))

        if pd.notna(row.get("SMA200")) and signal_condition(row, mkt_ok, params):
            entry_day = df.iloc[i + 1]
            entry_price = entry_day["Open"] * (1 + params["friction_pct"])
            atr_entry = row["ATR14"]
            if pd.isna(atr_entry) or atr_entry <= 0:
                i += 1
                continue

            initial_stop = entry_price - params["atr_stop_mult"] * atr_entry
            stop = initial_stop
            highest_close = entry_price

            outcome, exit_price, days_held = None, None, 0
            exit_index = i + 1
            for j in range(i + 1, min(i + 1 + params["hold_days"], len(df))):
                day = df.iloc[j]
                days_held += 1
                exit_index = j
                if day["Low"] <= stop:
                    outcome = "win" if stop > entry_price else "loss"
                    exit_price = stop * (1 - params["friction_pct"])
                    break
                highest_close = max(highest_close, day["Close"])
                day_atr = day["ATR14"] if pd.notna(day["ATR14"]) else atr_entry
                trail_candidate = highest_close - params["atr_stop_mult"] * day_atr
                stop = max(stop, trail_candidate)  # ratchet up only, never down

            if outcome is None:
                exit_index = min(i + params["hold_days"], len(df) - 1)
                exit_price = df.iloc[exit_index]["Close"] * (1 - params["friction_pct"])
                outcome = "win" if exit_price > entry_price else "loss"

            risk_per_share = entry_price - initial_stop
            r_multiple = (exit_price - entry_price) / risk_per_share if risk_per_share > 0 else 0

            trades.append({
                "symbol": None,
                "entry_date": entry_day["Date"],
                "outcome": outcome,
                "r_multiple": r_multiple,
                "pct_return": (exit_price - entry_price) / entry_price * 100,
                "days_held": days_held,
            })
            i = exit_index + 1  # FIX: resume right after THIS trade closes, not after the max-hold cap
        else:
            i += 1
    return trades


def run_backtest(universe, years, params) -> tuple[pd.DataFrame, dict]:
    start = (datetime.now() - timedelta(days=365 * years + 250)).strftime("%Y-%m-%d")
    market_regime = get_market_regime(start)
    nifty_close = get_nifty_close(start)
    data = yf.download(universe, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)

    backtest_fn = backtest_symbol_trailing if params.get("use_trailing_stop") else backtest_symbol

    all_trades = []
    for sym in universe:
        try:
            df = data[sym].dropna(how="all")
            if len(df) < 210:
                continue
            df = compute_indicators(df)
            df = add_relative_strength(df, nifty_close, params.get("rs_lookback", 63))
            trades = backtest_fn(df, market_regime, params)
            for t in trades:
                t["symbol"] = sym.replace(".NS", "")
            all_trades.extend(trades)
        except Exception:
            continue

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        return trades_df, {}

    wins = trades_df[trades_df.outcome == "win"]
    losses = trades_df[trades_df.outcome == "loss"]
    summary = {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_win_r": wins.r_multiple.mean() if len(wins) else 0,
        "avg_loss_r": losses.r_multiple.mean() if len(losses) else 0,
        "expectancy_r": trades_df.r_multiple.mean(),
        "avg_days_held": trades_df.days_held.mean(),
        "mode": "TRAILING STOP" if params.get("use_trailing_stop") else "FIXED TARGET",
    }
    return trades_df, summary

# ----------------------------------------------------------------------------------
# TODAY'S SCREENER
# ----------------------------------------------------------------------------------

def screen_today(universe, capital, risk_pct, params) -> tuple[pd.DataFrame, bool]:
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    market_regime = get_market_regime(start)
    is_bullish = bool(market_regime.iloc[-1]) if not market_regime.empty else True

    if not is_bullish:
        return pd.DataFrame(), False

    data = yf.download(universe, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)
    nifty_close = get_nifty_close(start)
    max_risk_amount = capital * risk_pct

    rows = []
    for sym in universe:
        try:
            df = data[sym].dropna(how="all")
            if len(df) < 210:
                continue
            df = compute_indicators(df)
            df = add_relative_strength(df, nifty_close, params.get("rs_lookback", 63))
            last = df.iloc[-1]
            if pd.notna(last.get("SMA200")) and signal_condition(last, is_bullish, params):
                atr = last["ATR14"]
                entry = last["Close"]
                stop = entry - params["atr_stop_mult"] * atr
                risk_per_share = entry - stop
                shares = int(max_risk_amount // risk_per_share) if risk_per_share > 0 else 0
                use_trail = params.get("use_trailing_stop", False)
                target_val = "Trail stop, no fixed target" if use_trail else round(entry + params["atr_target_mult"] * atr, 2)
                rr_val = "n/a (trailing)" if use_trail else round(params["atr_target_mult"] / params["atr_stop_mult"], 2)

                rows.append({
                    "Symbol": sym.replace(".NS", ""),
                    "Buy Near": round(entry, 2),
                    "Initial Stop-Loss": round(stop, 2),
                    "Target (Sell)": target_val,
                    "Qty (at your risk budget)": shares,
                    "Capital Req (Rs)": round(shares * entry, 2),
                    "Risk %": round(risk_per_share / entry * 100, 2),
                    "Reward:Risk": rr_val,
                    "Vol Surge x": round(last["Volume"] / last["VolAvg20"], 2),
                    "RSI14": round(last["RSI14"], 1),
                })
        except Exception:
            continue

    watchlist = pd.DataFrame(rows)
    if not watchlist.empty:
        watchlist = watchlist.sort_values("Vol Surge x", ascending=False).reset_index(drop=True)
    return watchlist, True

# ----------------------------------------------------------------------------------
# OPEN POSITION TRACKING -- sell/exit signals for stocks you already bought
# ----------------------------------------------------------------------------------

def evaluate_positions(positions: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    positions columns required: Symbol, Entry Date, Entry Price, Qty, Stop, Target
    Returns the same rows enriched with current price, P&L, a trailing-stop
    suggestion (only ever ratchets UP, never lowers your protection), and a
    plain Signal: HOLD / SELL - Stop-Loss Hit / SELL - Target Reached /
    CONSIDER EXIT - Trend Weakening.
    """
    if positions.empty:
        return positions

    symbols = [s if s.endswith(".NS") else f"{s}.NS" for s in positions["Symbol"]]
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    data = yf.download(symbols, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)

    out_rows = []
    for _, pos in positions.iterrows():
        sym = str(pos["Symbol"])
        yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        try:
            df = data[yf_sym] if len(symbols) > 1 else data
            df = df.dropna(how="all")
            df = compute_indicators(df)
            last = df.iloc[-1]

            entry_price = float(pos["Entry Price"])
            qty = int(pos["Qty"])
            stop = float(pos["Stop"])
            target = float(pos["Target"])
            current_price = float(last["Close"])
            atr = last["ATR14"]

            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_rs = (current_price - entry_price) * qty

            suggested_stop = stop
            if pd.notna(atr):
                trail_candidate = current_price - params["atr_stop_mult"] * atr
                suggested_stop = max(stop, trail_candidate)  # ratchet up only, never down

            if current_price <= stop:
                signal = "SELL - Stop-Loss Hit"
            elif current_price >= target:
                signal = "SELL - Target Reached"
            elif pd.notna(last.get("SMA50")) and current_price < last["SMA50"]:
                signal = "CONSIDER EXIT - Trend Weakening (below 50 SMA)"
            else:
                signal = "HOLD"

            days_held = None
            try:
                entry_date = pd.to_datetime(pos["Entry Date"])
                days_held = (df.index[-1].tz_localize(None) - entry_date.tz_localize(None)).days \
                    if hasattr(df.index[-1], "tzinfo") and df.index[-1].tzinfo else (df.index[-1] - entry_date).days
            except Exception:
                pass

            out_rows.append({
                "Symbol": sym,
                "Entry Date": pos.get("Entry Date", ""),
                "Days Held": days_held,
                "Entry Price": entry_price,
                "Current Price": round(current_price, 2),
                "Qty": qty,
                "P&L %": round(pnl_pct, 2),
                "P&L (Rs)": round(pnl_rs, 2),
                "Stop (yours)": stop,
                "Suggested Trailing Stop": round(suggested_stop, 2),
                "Target": target,
                "Signal": signal,
            })
        except Exception as e:
            out_rows.append({"Symbol": sym, "Signal": f"Error fetching data: {e}"})

    return pd.DataFrame(out_rows)

"""
engine.py -- core strategy logic (CANSLIM-inspired: fundamentals gate + technical score)
Shared by both the CLI script and the Streamlit web app.

WHERE THIS DESIGN CAME FROM (so future changes stay grounded, not vibes):
  - We backtested strict-AND filtering (contraction AND relative-strength both
    required) and watched trade count collapse from 162 to 6 with no proven
    quality gain -- fixed by switching to a WEIGHTED SCORE instead.
  - We tested fixed-target, pure-trailing, and pure-scaled exits across many
    runs; all landed in the same -0.12R to +0.02R band -- the exit mechanics
    were NOT the bottleneck. Ships ONE well-reasoned layered exit instead of
    three modes to keep re-testing.
  - After ~15 technical-only configurations, every large-sample result
    (n>=60) landed in the same near-breakeven band. That's the real finding:
    a pure PRICE-ACTION breakout has no edge on liquid large-caps. William
    O'Neil's original CANSLIM was never technical-only -- it required real
    earnings growth BEFORE the chart pattern even qualified. We'd only ever
    built the technical half. This version adds the fundamental gate back.
  - IMPORTANT LIMITATION, stated plainly: yfinance only exposes TODAY's
    earnings-growth figure, not a point-in-time historical record. Applying
    today's fundamentals to a trade from 3 years ago would be lookahead bias.
    So the earnings gate is LIVE-SCREENER-ONLY -- the backtest tab stays
    technical-only and says so, rather than faking a number that isn't real.
  - Several component thresholds (breakout volume multiple, contraction
    strictness, RS lookback window, RS margin) never showed a decisive
    effect across many tests, so they're fixed sensible constants now
    instead of exposed sliders -- simpler surface, same behavior.

Strategy, current version:
  MANDATORY GATES for the BACKTEST and the live screener alike:
    - Index regime: Nifty Close > its own 20-EMA > its own 50-SMA
    - Breakout: stock closes above its prior 20-day high
    - RSI sanity band: not deeply oversold, not blown-off overbought

  MANDATORY GATE, live screener ONLY (can't be backtested honestly):
    - Real earnings growth: latest quarterly YoY earnings growth exceeds a
      minimum threshold (the "C" in CANSLIM -- Current earnings growth)

  WEIGHTED SCORE (each worth 2 of 8 points -- NOT 10; regime/breakout/RSI are
  separate gates above and don't add to this total. Take the trade if score
  >= threshold):
    - Stock trend alignment: Close > SMA50 > SMA200
    - Relative strength vs Nifty: stock's 63-day return beats Nifty's by a
      real margin
    - Volatility contraction: ATR5/ATR20 tight AND recent volume dried up,
      measured through YESTERDAY, excluding the breakout day itself
    - Volume expansion: today's volume much higher than its 20-day average

  EXIT (layered, one method):
    - Initial stop: entry - 1.5x ATR
    - At +1.5x ATR: move stop to breakeven
    - At +2.5x ATR: sell 50% of the position, banking real profit
    - Remaining 50% trails behind price using the tighter of a 20-day
      structural low or a 2.0x ATR volatility trail
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

# Representative sample of liquid, established mid/small-caps -- NOT the official
# index list (too many for casual backtesting; many recent IPOs lack history).
# CAVEAT: testing today's known survivors against past years has survivorship
# bias baked in. Treat results here as more optimistic than a true point-in-time
# backtest. Not the default -- pick it deliberately in the sidebar if you want it.
MIDSMALLCAP_UNIVERSE = [
    "FEDERALBNK.NS", "MCX.NS", "SUZLON.NS", "BHEL.NS", "LAURUSLABS.NS", "POLYCAB.NS",
    "ABCAPITAL.NS", "INDIANB.NS", "PAGEIND.NS", "MPHASIS.NS", "PERSISTENT.NS",
    "COFORGE.NS", "LTTS.NS", "TATACOMM.NS", "VOLTAS.NS", "TRENT.NS", "GODREJPROP.NS",
    "OBEROIRLTY.NS", "PHOENIXLTD.NS", "INDHOTEL.NS", "JUBLFOOD.NS", "DEEPAKNTR.NS",
    "PIIND.NS", "SRF.NS", "GUJGASLTD.NS", "ASTRAL.NS", "DIXON.NS", "AUBANK.NS",
    "IDFCFIRSTB.NS", "BANDHANBNK.NS", "RBLBANK.NS", "IEX.NS", "CDSL.NS", "BSE.NS",
    "CROMPTON.NS", "WHIRLPOOL.NS", "ESCORTS.NS", "BALKRISIND.NS", "MOTHERSON.NS",
    "ASHOKLEY.NS", "BHARATFORG.NS", "CUMMINSIND.NS", "LUPIN.NS", "ALKEM.NS",
    "TORNTPHARM.NS", "GLENMARK.NS", "NATIONALUM.NS", "HINDZINC.NS", "JINDALSTEL.NS",
    "NMDC.NS", "SAIL.NS", "RECLTD.NS", "PFC.NS", "CONCOR.NS", "GMRINFRA.NS", "IRCTC.NS",
]

DEFAULT_PARAMS = dict(
    # Exposed to the user -- these had real, visible effects across our testing.
    atr_stop_mult=1.5,
    breakeven_mult=1.5,
    partial_target_mult=2.5,
    runner_trail_mult=2.0,
    hold_days=20,
    friction_pct=0.0015,
    score_threshold=6,   # out of 8 (see setup_score) -- 3 of 4 factors, not all 4
    rsi_low=45, rsi_high=70,
    min_earnings_growth=0.10,   # live screener only -- 10% YoY quarterly earnings growth

    # Fixed constants -- tested repeatedly this session with no decisive effect
    # from tuning, so simplified out of the UI rather than left as clutter.
    volume_mult=1.5,
    rs_lookback=63,
    rs_min_outperformance=0.05,
    contraction_ratio_threshold=0.70,
)

# ----------------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Guard: yfinance sometimes returns a partial/incomplete bar for "today" if
    # fetched while the market is open, or a stale row with NaN OHLC.
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)      # prior 20d high, excludes today
    df["Low20"] = df["Low"].rolling(20).min().shift(1)        # prior 20d low, excludes today
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

    # VCP-style contraction: ATR5/ATR20 ratio AND 5-day volume vs 20-day average,
    # both measured through YESTERDAY (shift(1)) so the breakout day's own
    # expansion can't contaminate the "was it quiet beforehand" measurement.
    atr5_raw = tr.rolling(5).mean()
    atr20_raw = tr.rolling(20).mean()
    df["ATR_ContractionRatio"] = (atr5_raw / atr20_raw).shift(1)
    df["Vol5Avg"] = df["Volume"].rolling(5).mean().shift(1)

    return df


def get_nifty_data(start_date: str) -> pd.DataFrame:
    nifty = yf.download(BENCHMARK_TICKER, start=start_date, auto_adjust=True, progress=False)
    if nifty.empty:
        return pd.DataFrame()
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty = nifty.xs(BENCHMARK_TICKER, level=1, axis=1)
    nifty["EMA20"] = nifty["Close"].ewm(span=20, adjust=False).mean()
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    return nifty


def get_market_regime(nifty_df: pd.DataFrame) -> pd.Series:
    """True only when Nifty is in a real uptrend: Close > 20-EMA > 50-SMA."""
    if nifty_df.empty:
        return pd.Series(dtype=bool)
    cond = (nifty_df["Close"] > nifty_df["EMA20"]) & (nifty_df["EMA20"] > nifty_df["SMA50"])
    return cond.rename("bullish")


def add_relative_strength(df: pd.DataFrame, nifty_close: pd.Series, lookback: int) -> pd.DataFrame:
    """RS_Diff = stock's % return over `lookback` days minus Nifty's % return
    over the same window. A real margin, not a ratio (ratios break when either
    return is negative -- e.g. a stock down 5% vs Nifty down 20% is a big
    relative WIN, but the ratio 0.25 looks like a loser)."""
    df = df.copy()
    if nifty_close.empty:
        df["RS_Diff"] = np.nan
        return df
    aligned = nifty_close.reindex(df.index, method="ffill")
    df["RS_Diff"] = df["Close"].pct_change(lookback) - aligned.pct_change(lookback)
    return df


def get_earnings_growth(symbol: str):
    """
    Latest quarterly YoY earnings growth, as a fraction (0.15 = 15%). None if
    Yahoo doesn't have it for this symbol. LIVE DATA ONLY -- this reflects
    TODAY's most recent reported figure, not what was known on any past date,
    which is exactly why this gate can't be used in the backtest.
    """
    try:
        info = yf.Ticker(symbol).info
        return info.get("earningsQuarterlyGrowth")
    except Exception:
        return None

# ----------------------------------------------------------------------------------
# ENTRY SIGNAL: mandatory gates + weighted score
# ----------------------------------------------------------------------------------

def setup_score(row, market_ok: bool, params: dict) -> int:
    """Returns -1 if a mandatory gate fails (never a valid setup that day),
    otherwise 2-10 reflecting how many quality factors lined up."""
    breakout_ok = (
        row["Close"] > row["High20"]
        and params["rsi_low"] <= row["RSI14"] <= params["rsi_high"]
    )
    if not market_ok or not breakout_ok:
        return -1

    score = 0

    if row["Close"] > row["SMA50"] > row["SMA200"]:
        score += 2  # trend alignment

    rs_diff = row.get("RS_Diff")
    if pd.notna(rs_diff) and rs_diff > params.get("rs_min_outperformance", 0.05):
        score += 2  # meaningful relative strength vs Nifty

    ratio = row.get("ATR_ContractionRatio")
    vol5 = row.get("Vol5Avg")
    vol20 = row.get("VolAvg20")
    if pd.notna(ratio) and pd.notna(vol5) and pd.notna(vol20):
        if ratio < params.get("contraction_ratio_threshold", 0.70) and vol5 < vol20:
            score += 2  # price AND volume both dried up before the breakout

    if row["Volume"] >= params["volume_mult"] * row["VolAvg20"]:
        score += 2  # volume expansion on the breakout day itself

    return score


def entry_signal(row, market_ok: bool, params: dict) -> bool:
    return setup_score(row, market_ok, params) >= params.get("score_threshold", 7)

# ----------------------------------------------------------------------------------
# LAYERED EXIT SIMULATION (used by both backtest and live position tracking)
# ----------------------------------------------------------------------------------

def simulate_layered_exit(df: pd.DataFrame, i: int, entry_price: float, atr_entry: float, params: dict):
    """
    df is the reset-index'd, per-symbol frame; i is the signal-day row index
    (entry happens at df.iloc[i+1]'s open, already priced into entry_price).
    Returns (r_multiple, days_held, exit_index).
    """
    fric = params["friction_pct"]
    initial_stop = entry_price - params["atr_stop_mult"] * atr_entry
    stop = initial_stop
    breakeven_trigger = entry_price + params.get("breakeven_mult", 1.5) * atr_entry
    partial_trigger = entry_price + params.get("partial_target_mult", 2.5) * atr_entry
    runner_trail_mult = params.get("runner_trail_mult", 2.0)

    partial_taken = False
    partial_r = None
    highest_close = entry_price
    r_multiple = None
    days_held = 0
    exit_index = i + 1

    for j in range(i + 1, min(i + 1 + params["hold_days"], len(df))):
        day = df.iloc[j]
        days_held += 1
        exit_index = j
        day_atr = day["ATR14"] if pd.notna(day["ATR14"]) else atr_entry

        if not partial_taken:
            if day["Low"] <= stop:
                exit_price = stop * (1 - fric)
                r_multiple = (exit_price - entry_price) / (entry_price - initial_stop)
                break
            if day["High"] >= breakeven_trigger:
                stop = max(stop, entry_price)  # lock in "can't lose" early
            if day["High"] >= partial_trigger:
                partial_taken = True
                partial_exit = partial_trigger * (1 - fric)
                partial_r = (partial_exit - entry_price) / (entry_price - initial_stop)
                stop = max(stop, entry_price)
                highest_close = max(highest_close, day["Close"], partial_trigger)
        else:
            if day["Low"] <= stop:
                runner_exit = stop * (1 - fric)
                runner_r = (runner_exit - entry_price) / (entry_price - initial_stop)
                r_multiple = 0.5 * partial_r + 0.5 * runner_r
                break
            highest_close = max(highest_close, day["Close"])
            atr_trail = highest_close - runner_trail_mult * day_atr
            structural_trail = day["Low20"] if pd.notna(day.get("Low20")) else atr_trail
            stop = max(stop, atr_trail, structural_trail)  # tighter (higher) of the two

    if r_multiple is None:
        exit_index = min(i + params["hold_days"], len(df) - 1)
        final_close = df.iloc[exit_index]["Close"] * (1 - fric)
        final_r = (final_close - entry_price) / (entry_price - initial_stop)
        r_multiple = (0.5 * partial_r + 0.5 * final_r) if partial_taken else final_r

    return r_multiple, days_held, exit_index

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
        mkt_ok = bool(market_regime.get(date, False))

        if pd.notna(row.get("SMA200")) and entry_signal(row, mkt_ok, params):
            entry_day = df.iloc[i + 1]
            entry_price = entry_day["Open"] * (1 + params["friction_pct"])
            atr_entry = row["ATR14"]
            if pd.isna(atr_entry) or atr_entry <= 0:
                i += 1
                continue

            r_multiple, days_held, exit_index = simulate_layered_exit(df, i, entry_price, atr_entry, params)
            trades.append({
                "symbol": None,
                "entry_date": entry_day["Date"],
                "outcome": "win" if r_multiple > 0 else "loss",
                "r_multiple": r_multiple,
                "days_held": days_held,
                "setup_score": setup_score(row, mkt_ok, params),
            })
            i = exit_index + 1  # resume right after THIS trade closes
        else:
            i += 1
    return trades


def run_backtest(universe, years, params) -> tuple[pd.DataFrame, dict]:
    start = (datetime.now() - timedelta(days=365 * years + 250)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    market_regime = get_market_regime(nifty_df)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)

    data = yf.download(universe, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)

    all_trades = []
    for sym in universe:
        try:
            df = data[sym].dropna(how="all")
            if len(df) < 210:
                continue
            df = compute_indicators(df)
            df = add_relative_strength(df, nifty_close, params.get("rs_lookback", 63))
            trades = backtest_symbol(df, market_regime, params)
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
    }
    return trades_df, summary

# ----------------------------------------------------------------------------------
# TODAY'S SCREENER
# ----------------------------------------------------------------------------------

def screen_today(universe, capital, risk_pct, params) -> tuple[pd.DataFrame, bool]:
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    market_regime = get_market_regime(nifty_df)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)
    is_bullish = bool(market_regime.iloc[-1]) if not market_regime.empty else False

    if not is_bullish:
        return pd.DataFrame(), False

    data = yf.download(universe, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)
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
            if pd.isna(last.get("SMA200")):
                continue
            score = setup_score(last, is_bullish, params)
            if score >= params.get("score_threshold", 7):
                # Fundamental gate, live-only: only spend the extra API call on
                # names that already passed the technical bar.
                earnings_growth = get_earnings_growth(sym)
                min_growth = params.get("min_earnings_growth", 0.10)
                if earnings_growth is None or earnings_growth < min_growth:
                    continue  # no confirmed real earnings growth -- skip, CANSLIM-style

                atr = last["ATR14"]
                entry = last["Close"]
                stop = entry - params["atr_stop_mult"] * atr
                risk_per_share = entry - stop
                shares = int(max_risk_amount // risk_per_share) if risk_per_share > 0 else 0
                breakeven_at = entry + params.get("breakeven_mult", 1.5) * atr
                partial_at = entry + params.get("partial_target_mult", 2.5) * atr

                rows.append({
                    "Symbol": sym.replace(".NS", ""),
                    "Buy Near": round(entry, 2),
                    "Initial Stop-Loss": round(stop, 2),
                    "Move to Breakeven @": round(breakeven_at, 2),
                    "Sell 50% @": round(partial_at, 2),
                    "Then": "trail rest (20d low / 2xATR)",
                    "Qty (at your risk budget)": shares,
                    "Capital Req (Rs)": round(shares * entry, 2),
                    "Risk %": round(risk_per_share / entry * 100, 2),
                    "Earnings Growth YoY": f"{earnings_growth*100:.1f}%",
                    "_score_num": score,
                    "Setup Score": f"{score}/8",
                    "RSI14": round(last["RSI14"], 1),
                })
        except Exception:
            continue

    watchlist = pd.DataFrame(rows)
    if not watchlist.empty:
        watchlist = watchlist.sort_values("_score_num", ascending=False).drop(columns="_score_num").reset_index(drop=True)
    return watchlist, True

# ----------------------------------------------------------------------------------
# OPEN POSITION TRACKING -- sell/exit signals for stocks you already bought
# ----------------------------------------------------------------------------------

def evaluate_positions(positions: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    positions columns required: Symbol, Entry Date, Entry Price, Qty, Stop, Target
    (Target here is treated as the user's own reference/partial-target level --
    this function reports live status, it doesn't re-simulate the layered exit.)
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
                trail_candidate = current_price - params.get("runner_trail_mult", 2.0) * atr
                suggested_stop = max(stop, trail_candidate)

            if current_price <= stop:
                signal = "SELL - Stop-Loss Hit"
            elif current_price >= target:
                signal = "SELL 50% - Partial Target Reached"
            elif pd.notna(last.get("SMA50")) and current_price < last["SMA50"]:
                signal = "CONSIDER EXIT - Trend Weakening (below 50 SMA)"
            else:
                signal = "HOLD"

            days_held = None
            try:
                entry_date = pd.to_datetime(pos["Entry Date"])
                idx = df.index[-1]
                idx = idx.tz_localize(None) if getattr(idx, "tzinfo", None) else idx
                entry_date = entry_date.tz_localize(None) if getattr(entry_date, "tzinfo", None) else entry_date
                days_held = (idx - entry_date).days
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

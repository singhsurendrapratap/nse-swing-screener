"""
engine.py -- NSE swing/positional stock-selection engine

Design goal:
    Trade LESS, but make each trade a higher-quality candidate.

This version changes the original system in four important ways:
1) A 0-100 quality score replaces the coarse 0/2/3 point buckets.
2) Breakout QUALITY is scored, not merely "close > prior 20-day high".
3) The backtest records diagnostics for every trade so winners vs losers can be studied.
4) The live screener ranks candidates and can cap the number of trades per day.

Important:
- The live earnings-growth filter is still LIVE ONLY. It is not used in the
  historical backtest because yfinance does not provide reliable point-in-time
  historical earnings growth in this workflow.
- This is a research tool, not investment advice.
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
    # Risk / exit
    atr_stop_mult=1.5,
    breakeven_r=1.0,
    partial_r=2.0,
    runner_trail_mult=2.0,
    hold_days=20,
    friction_pct=0.0015,

    # New selection engine
    score_threshold=78,
    max_trades_per_day=3,
    rsi_low=45,
    rsi_high=70,
    min_earnings_growth=0.10,

    # Fixed research constants; change only after out-of-sample testing.
    rs_lookback=63,
    rs_min_outperformance=0.05,
    breakout_min_atr=0.25,
    breakout_max_atr=2.50,
    volume_mult=1.50,
    contraction_ratio_threshold=0.80,
    max_gap_pct=4.0,
    max_extension_atr=3.5,
)

# -----------------------------------------------------------------------------
# DATA / INDICATORS
# -----------------------------------------------------------------------------

def _flatten_single_ticker_columns(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Make yfinance output consistently OHLCV for one ticker."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Typical single-ticker download: level 0 = Price, level 1 = Ticker.
        if symbol is not None:
            try:
                out = out.xs(symbol, axis=1, level=-1)
            except Exception:
                try:
                    out = out.xs(symbol, axis=1, level=0)
                except Exception:
                    pass
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_single_ticker_columns(df)
    if df.empty:
        return df
    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["High20"] = df["High"].rolling(20).max().shift(1)
    df["Low20"] = df["Low"].rolling(20).min().shift(1)
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
    df["TR"] = tr
    df["ATR14"] = tr.rolling(14).mean()
    atr5 = tr.rolling(5).mean()
    atr20 = tr.rolling(20).mean()
    df["ATR5"] = atr5
    df["ATR20"] = atr20

    # Pre-breakout contraction is measured through yesterday.
    df["ATR_ContractionRatio"] = (atr5 / atr20).shift(1)
    df["Vol5Avg"] = df["Volume"].rolling(5).mean().shift(1)

    # Multi-horizon momentum / relative strength.
    for n in (20, 63, 126):
        df[f"Return{n}"] = df["Close"].pct_change(n)

    # Trend slopes are normalized so they are comparable across stocks.
    df["EMA20_Slope5"] = df["EMA20"].pct_change(5)
    df["SMA50_Slope20"] = df["SMA50"].pct_change(20)

    # Breakout-day candle quality.
    prev_close = df["Close"].shift(1)
    day_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["GapPct"] = (df["Open"] / prev_close - 1) * 100
    df["CandleBody"] = (df["Close"] - df["Open"]).abs()
    df["BodyATR"] = df["CandleBody"] / df["ATR14"].replace(0, np.nan)
    df["CloseLocation"] = (df["Close"] - df["Low"]) / day_range
    df["UpperWickPct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / day_range

    # How far beyond resistance the close is, in volatility units.
    df["BreakoutATR"] = (df["Close"] - df["High20"]) / df["ATR14"].replace(0, np.nan)
    df["ExtensionATR"] = (df["Close"] - df["EMA20"]) / df["ATR14"].replace(0, np.nan)
    df["VolumeRatio"] = df["Volume"] / df["VolAvg20"].replace(0, np.nan)

    return df


def get_nifty_data(start_date: str) -> pd.DataFrame:
    nifty = yf.download(BENCHMARK_TICKER, start=start_date, auto_adjust=True, progress=False)
    nifty = _flatten_single_ticker_columns(nifty, BENCHMARK_TICKER)
    if nifty.empty or "Close" not in nifty.columns:
        return pd.DataFrame()
    nifty["EMA20"] = nifty["Close"].ewm(span=20, adjust=False).mean()
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["EMA20_Slope5"] = nifty["EMA20"].pct_change(5)
    nifty["SMA50_Slope20"] = nifty["SMA50"].pct_change(20)
    return nifty


def get_market_regime(nifty_df: pd.DataFrame) -> pd.Series:
    """Base long regime: Nifty Close > EMA20 > SMA50."""
    if nifty_df.empty:
        return pd.Series(dtype=bool)
    cond = (
        (nifty_df["Close"] > nifty_df["EMA20"])
        & (nifty_df["EMA20"] > nifty_df["SMA50"])
    )
    return cond.rename("bullish")


def add_relative_strength(df: pd.DataFrame, nifty_close: pd.Series, lookback: int) -> pd.DataFrame:
    df = df.copy()
    if nifty_close.empty:
        for n in (20, 63, 126):
            df[f"RS_Diff{n}"] = np.nan
        return df
    aligned = nifty_close.reindex(df.index, method="ffill")
    for n in (20, 63, 126):
        df[f"RS_Diff{n}"] = df["Close"].pct_change(n) - aligned.pct_change(n)
    # Preserve the old field name for compatibility.
    df["RS_Diff"] = df[f"RS_Diff{lookback}"]
    return df


def get_earnings_growth(symbol: str):
    try:
        info = yf.Ticker(symbol).info
        value = info.get("earningsQuarterlyGrowth")
        return float(value) if value is not None else None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# QUALITY SCORE
# -----------------------------------------------------------------------------

def market_quality(row, market_ok: bool) -> tuple[int, str]:
    if not market_ok:
        return 0, "BEAR/CHOP"
    score = 8  # base bullish structure
    if pd.notna(row.get("EMA20_Slope5")) and row["EMA20_Slope5"] > 0:
        score += 4
    if pd.notna(row.get("SMA50_Slope20")) and row["SMA50_Slope20"] > 0:
        score += 3
    if pd.notna(row.get("EMA20_Slope5")) and row["EMA20_Slope5"] > 0.01:
        score += 2
    if score >= 15:
        return min(score, 15), "A+"
    if score >= 12:
        return min(score, 15), "A"
    return min(score, 15), "B"


def _clip_score(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(x))))


def score_setup(row, market_ok: bool, params: dict, earnings_growth=None, include_earnings=False):
    """Return a detailed 0-100 setup score plus diagnostics.

    The score is deliberately interpretable rather than machine-learned.
    It is intended for ranking and research, not as a claim of predictive certainty.
    """
    components = {}

    # 1. Market regime: 15
    mkt_score, regime = market_quality(row, market_ok)
    components["market"] = mkt_score

    # 2. Trend quality: 15
    trend = 0
    if pd.notna(row.get("Close")) and pd.notna(row.get("EMA20")) and row["Close"] > row["EMA20"]:
        trend += 5
    if pd.notna(row.get("EMA20")) and pd.notna(row.get("SMA50")) and row["EMA20"] > row["SMA50"]:
        trend += 4
    if pd.notna(row.get("SMA50")) and pd.notna(row.get("SMA200")) and row["SMA50"] > row["SMA200"]:
        trend += 4
    if pd.notna(row.get("EMA20_Slope5")) and row["EMA20_Slope5"] > 0:
        trend += 2
    components["trend"] = min(trend, 15)

    # 3. Relative strength: 15
    rs = 0
    rs20, rs63, rs126 = row.get("RS_Diff20"), row.get("RS_Diff63"), row.get("RS_Diff126")
    if pd.notna(rs20) and rs20 > 0:
        rs += 3
    if pd.notna(rs63):
        if rs63 >= 0.10:
            rs += 9
        elif rs63 >= params.get("rs_min_outperformance", 0.05):
            rs += 7
        elif rs63 >= 0:
            rs += 3
    if pd.notna(rs126) and rs126 > 0.05:
        rs += 3
    components["relative_strength"] = min(rs, 15)

    # 4. Breakout quality: 20
    bo = 0
    b_atr = row.get("BreakoutATR")
    if pd.notna(b_atr):
        if b_atr >= 1.0:
            bo += 8
        elif b_atr >= 0.50:
            bo += 6
        elif b_atr >= params.get("breakout_min_atr", 0.25):
            bo += 4
    cl = row.get("CloseLocation")
    if pd.notna(cl):
        if cl >= 0.80:
            bo += 5
        elif cl >= 0.65:
            bo += 3
        elif cl >= 0.50:
            bo += 1
    body_atr = row.get("BodyATR")
    if pd.notna(body_atr):
        if body_atr >= 0.75:
            bo += 4
        elif body_atr >= 0.40:
            bo += 3
        elif body_atr >= 0.20:
            bo += 1
    uw = row.get("UpperWickPct")
    if pd.notna(uw):
        if uw <= 0.15:
            bo += 3
        elif uw <= 0.25:
            bo += 2
    components["breakout"] = min(bo, 20)

    # 5. Volume confirmation: 15
    vol = 0
    vr = row.get("VolumeRatio")
    if pd.notna(vr):
        if vr >= 2.50:
            vol += 15
        elif vr >= 2.00:
            vol += 13
        elif vr >= 1.50:
            vol += 10
        elif vr >= 1.25:
            vol += 5
    components["volume"] = min(vol, 15)

    # 6. Pre-breakout contraction: 10
    con = 0
    vol5, vol20, cr = row.get("Vol5Avg"), row.get("VolAvg20"), row.get("ATR_ContractionRatio")
    if pd.notna(vol5) and pd.notna(vol20) and vol5 < vol20:
        con += 5
    if pd.notna(cr):
        if cr < 0.70:
            con += 5
        elif cr < params.get("contraction_ratio_threshold", 0.80):
            con += 3
        elif cr < 0.90:
            con += 1
    components["contraction"] = min(con, 10)

    # 7. RSI quality: 5
    rsi_score = 0
    rsi = row.get("RSI14")
    if pd.notna(rsi):
        if 50 <= rsi <= 65:
            rsi_score = 5
        elif params.get("rsi_low", 45) <= rsi <= params.get("rsi_high", 70):
            rsi_score = 3
    components["rsi"] = rsi_score

    # 8. Extension / chase control: 5
    ext_score = 0
    ext = row.get("ExtensionATR")
    if pd.notna(ext):
        if 0 <= ext <= 2.0:
            ext_score = 5
        elif 2.0 < ext <= params.get("max_extension_atr", 3.5):
            ext_score = 2
    components["extension"] = ext_score

    technical_score = sum(components.values())

    # Earnings is intentionally an additive live-only filter, not part of the
    # historical technical score. Keep the technical max at 100.
    if include_earnings:
        min_growth = params.get("min_earnings_growth", 0.10)
        earnings_ok = earnings_growth is not None and earnings_growth >= min_growth
    else:
        earnings_ok = True

    mandatory = (
        market_ok
        and pd.notna(row.get("High20"))
        and row["Close"] > row["High20"]
        and pd.notna(row.get("RSI14"))
        and params.get("rsi_low", 45) <= row["RSI14"] <= params.get("rsi_high", 70)
        and pd.notna(row.get("ATR14"))
        and row["ATR14"] > 0
        and pd.notna(row.get("BreakoutATR"))
        and row["BreakoutATR"] >= params.get("breakout_min_atr", 0.25)
        and row["BreakoutATR"] <= params.get("breakout_max_atr", 2.50)
        and pd.notna(row.get("VolumeRatio"))
        and row["VolumeRatio"] >= params.get("volume_mult", 1.50)
        and pd.notna(row.get("ExtensionATR"))
        and row["ExtensionATR"] <= params.get("max_extension_atr", 3.5)
        and (pd.isna(row.get("GapPct")) or abs(row["GapPct"]) <= params.get("max_gap_pct", 4.0))
        and earnings_ok
    )

    return {
        "score": _clip_score(technical_score),
        "components": components,
        "regime": regime,
        "earnings_ok": earnings_ok,
        "mandatory_ok": mandatory,
    }


def setup_score(row, market_ok: bool, params: dict, earnings_growth=None, include_earnings=False) -> int:
    result = score_setup(row, market_ok, params, earnings_growth, include_earnings)
    return result["score"] if result["mandatory_ok"] else -1


def entry_signal(row, market_ok: bool, params: dict) -> bool:
    result = score_setup(row, market_ok, params, include_earnings=False)
    return result["mandatory_ok"] and result["score"] >= params.get("score_threshold", 78)

# -----------------------------------------------------------------------------
# EXIT SIMULATION
# -----------------------------------------------------------------------------

def simulate_layered_exit(df: pd.DataFrame, i: int, entry_price: float, atr_entry: float, params: dict):
    fric = params["friction_pct"]
    one_r = params["atr_stop_mult"] * atr_entry
    initial_stop = entry_price - one_r
    stop = initial_stop
    breakeven_trigger = entry_price + params.get("breakeven_r", 1.0) * one_r
    partial_trigger = entry_price + params.get("partial_r", 2.0) * one_r
    runner_trail_mult = params.get("runner_trail_mult", 2.0)

    partial_taken = False
    partial_r = None
    highest_close = entry_price
    r_multiple = None
    days_held = 0
    exit_index = i + 1
    exit_reason = "time_exit"

    for j in range(i + 1, min(i + 1 + params["hold_days"], len(df))):
        day = df.iloc[j]
        days_held += 1
        exit_index = j
        day_atr = day["ATR14"] if pd.notna(day["ATR14"]) else atr_entry

        if not partial_taken:
            if day["Low"] <= stop:
                exit_price = stop * (1 - fric)
                r_multiple = (exit_price - entry_price) / one_r
                exit_reason = "initial_stop" if stop == initial_stop else "breakeven_stop"
                break
            if day["High"] >= breakeven_trigger:
                stop = max(stop, entry_price)
            if day["High"] >= partial_trigger:
                partial_taken = True
                partial_exit = partial_trigger * (1 - fric)
                partial_r = (partial_exit - entry_price) / one_r
                stop = max(stop, entry_price)
                highest_close = max(highest_close, day["Close"], partial_trigger)
        else:
            if day["Low"] <= stop:
                runner_exit = stop * (1 - fric)
                runner_r = (runner_exit - entry_price) / one_r
                r_multiple = 0.5 * partial_r + 0.5 * runner_r
                exit_reason = "runner_trail"
                break
            highest_close = max(highest_close, day["Close"])
            atr_trail = highest_close - runner_trail_mult * day_atr
            ema_trail = day["EMA20"] if pd.notna(day.get("EMA20")) else atr_trail
            stop = max(stop, atr_trail, ema_trail)

    if r_multiple is None:
        exit_index = min(i + params["hold_days"], len(df) - 1)
        final_close = df.iloc[exit_index]["Close"] * (1 - fric)
        final_r = (final_close - entry_price) / one_r
        r_multiple = (0.5 * partial_r + 0.5 * final_r) if partial_taken else final_r
        exit_reason = "time_exit"

    return r_multiple, days_held, exit_index, exit_reason

# -----------------------------------------------------------------------------
# BACKTEST
# -----------------------------------------------------------------------------

def backtest_symbol(df: pd.DataFrame, market_regime: pd.Series, params: dict) -> list:
    trades = []
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    i = 0
    while i < len(df) - 2:
        row = df.iloc[i]
        date = row["Date"]
        mkt_ok = bool(market_regime.get(date, False))
        result = score_setup(row, mkt_ok, params, include_earnings=False)

        if result["mandatory_ok"] and result["score"] >= params.get("score_threshold", 78):
            entry_day = df.iloc[i + 1]
            entry_price = entry_day["Open"] * (1 + params["friction_pct"])
            atr_entry = row["ATR14"]
            if pd.isna(atr_entry) or atr_entry <= 0:
                i += 1
                continue

            r_multiple, days_held, exit_index, exit_reason = simulate_layered_exit(
                df, i, entry_price, atr_entry, params
            )
            components = result["components"]
            trades.append({
                "symbol": None,
                "signal_date": date,
                "entry_date": entry_day["Date"],
                "outcome": "win" if r_multiple > 0 else "loss",
                "r_multiple": r_multiple,
                "days_held": days_held,
                "setup_score": result["score"],
                "market_regime": result["regime"],
                "exit_reason": exit_reason,
                "entry_price": entry_price,
                "atr_entry": atr_entry,
                "breakout_atr": row.get("BreakoutATR"),
                "volume_ratio": row.get("VolumeRatio"),
                "rs20": row.get("RS_Diff20"),
                "rs63": row.get("RS_Diff63"),
                "rs126": row.get("RS_Diff126"),
                "rsi14": row.get("RSI14"),
                "extension_atr": row.get("ExtensionATR"),
                "gap_pct": row.get("GapPct"),
                "close_location": row.get("CloseLocation"),
                "body_atr": row.get("BodyATR"),
                "upper_wick_pct": row.get("UpperWickPct"),
                "atr_contraction": row.get("ATR_ContractionRatio"),
                "trend_score": components["trend"],
                "rs_score": components["relative_strength"],
                "breakout_score": components["breakout"],
                "volume_score": components["volume"],
                "contraction_score": components["contraction"],
            })
            # No overlapping positions in the same stock.
            i = exit_index + 1
        else:
            i += 1
    return trades


def _extract_symbol_frame(data: pd.DataFrame, sym: str, universe_len: int) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    try:
        if universe_len == 1:
            return _flatten_single_ticker_columns(data, sym).dropna(how="all")
        frame = data[sym]
        return frame.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def run_backtest(universe, years, params, return_candidates=False) -> tuple[pd.DataFrame, dict]:
    start = (datetime.now() - timedelta(days=365 * years + 260)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    market_regime = get_market_regime(nifty_df)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)

    data = yf.download(
        universe, start=start, group_by="ticker", auto_adjust=True,
        progress=False, threads=True
    )

    all_trades = []
    for sym in universe:
        try:
            df = _extract_symbol_frame(data, sym, len(universe))
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

    # Enforce the "trade less" rule across the whole universe.
    # Pick only the highest-quality setups on each signal day.
    if "signal_date" in trades_df.columns:
        trades_df["signal_date"] = pd.to_datetime(trades_df["signal_date"])
        trades_df = (
            trades_df.sort_values(["signal_date", "setup_score", "r_multiple"], ascending=[True, False, False])
            .groupby("signal_date", group_keys=False)
            .head(int(params.get("max_trades_per_day", 3)))
            .sort_values("entry_date")
            .reset_index(drop=True)
        )

    wins = trades_df[trades_df.outcome == "win"]
    losses = trades_df[trades_df.outcome == "loss"]
    gross_win = wins.r_multiple.sum() if len(wins) else 0
    gross_loss_abs = abs(losses.r_multiple.sum()) if len(losses) else 0
    summary = {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_win_r": wins.r_multiple.mean() if len(wins) else 0,
        "avg_loss_r": losses.r_multiple.mean() if len(losses) else 0,
        "expectancy_r": trades_df.r_multiple.mean(),
        "avg_days_held": trades_df.days_held.mean(),
        "profit_factor": gross_win / gross_loss_abs if gross_loss_abs else np.inf,
        "total_r": trades_df.r_multiple.sum(),
        "max_loss_streak": _max_streak((trades_df.outcome == "loss").tolist()),
        "parameters": {k: params[k] for k in params},
    }
    return trades_df.sort_values("entry_date").reset_index(drop=True), summary


def _max_streak(values):
    best = cur = 0
    for v in values:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

# -----------------------------------------------------------------------------
# TODAY'S SCREENER
# -----------------------------------------------------------------------------

def screen_today(universe, capital, risk_pct, params) -> tuple[pd.DataFrame, bool]:
    start = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    market_regime = get_market_regime(nifty_df)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)
    is_bullish = bool(market_regime.iloc[-1]) if not market_regime.empty else False

    if not is_bullish:
        return pd.DataFrame(), False

    data = yf.download(
        universe, start=start, group_by="ticker", auto_adjust=True,
        progress=False, threads=True
    )
    max_risk_amount = capital * risk_pct
    candidates = []

    for sym in universe:
        try:
            df = _extract_symbol_frame(data, sym, len(universe))
            if len(df) < 210:
                continue
            df = compute_indicators(df)
            df = add_relative_strength(df, nifty_close, params.get("rs_lookback", 63))
            last = df.iloc[-1]
            if pd.isna(last.get("SMA200")):
                continue

            # Earnings is checked only after the technical score passes.
            tech = score_setup(last, is_bullish, params, include_earnings=False)
            if not tech["mandatory_ok"] or tech["score"] < params.get("score_threshold", 78):
                continue

            earnings_growth = get_earnings_growth(sym)
            live = score_setup(
                last, is_bullish, params,
                earnings_growth=earnings_growth,
                include_earnings=True,
            )
            if not live["earnings_ok"]:
                continue

            atr = float(last["ATR14"])
            entry = float(last["Close"])
            stop = entry - params["atr_stop_mult"] * atr
            risk_per_share = entry - stop
            shares = int(max_risk_amount // risk_per_share) if risk_per_share > 0 else 0
            breakeven_at = entry + params.get("breakeven_r", 1.0) * risk_per_share
            partial_at = entry + params.get("partial_r", 2.0) * risk_per_share

            candidates.append({
                "Symbol": sym.replace(".NS", ""),
                "Quality Score": tech["score"],
                "Regime": tech["regime"],
                "Buy Near": round(entry, 2),
                "Initial Stop": round(stop, 2),
                "Breakeven": round(breakeven_at, 2),
                "Sell 50%": round(partial_at, 2),
                "Qty": shares,
                "Capital Req": round(shares * entry, 2),
                "Risk / Share": round(risk_per_share, 2),
                "RS 63D": f"{last.get('RS_Diff63', np.nan)*100:.1f}%" if pd.notna(last.get("RS_Diff63")) else "N/A",
                "Breakout": f"{last.get('BreakoutATR', np.nan):.2f} ATR" if pd.notna(last.get("BreakoutATR")) else "N/A",
                "Volume": f"{last.get('VolumeRatio', np.nan):.2f}x" if pd.notna(last.get("VolumeRatio")) else "N/A",
                "RSI": round(last["RSI14"], 1),
                "Gap": f"{last.get('GapPct', np.nan):.1f}%" if pd.notna(last.get("GapPct")) else "N/A",
                "Earnings YoY": f"{earnings_growth*100:.1f}%" if earnings_growth is not None else "N/A",
                "Then": "Trail rest: higher of 20EMA / ATR trail",
                "_score_num": tech["score"],
                "_symbol": sym,
            })
        except Exception:
            continue

    if not candidates:
        return pd.DataFrame(), True

    candidates = sorted(candidates, key=lambda x: (-x["_score_num"], x["_symbol"]))
    limit = int(params.get("max_trades_per_day", 3))
    selected = candidates[:limit]
    watchlist = pd.DataFrame(selected)
    if not watchlist.empty:
        watchlist = watchlist.drop(columns=["_score_num", "_symbol"])
    return watchlist.reset_index(drop=True), True

# -----------------------------------------------------------------------------
# POSITION TRACKING
# -----------------------------------------------------------------------------

def evaluate_positions(positions: pd.DataFrame, params: dict) -> pd.DataFrame:
    if positions.empty:
        return positions

    symbols = [s if str(s).endswith(".NS") else f"{s}.NS" for s in positions["Symbol"]]
    start = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")
    data = yf.download(symbols, start=start, group_by="ticker", auto_adjust=True,
                       progress=False, threads=True)

    out_rows = []
    for _, pos in positions.iterrows():
        sym = str(pos["Symbol"])
        yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        try:
            df = _extract_symbol_frame(data, yf_sym, len(symbols))
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
                ema_candidate = last["EMA20"] if pd.notna(last.get("EMA20")) else trail_candidate
                suggested_stop = max(stop, trail_candidate, ema_candidate)

            if current_price <= stop:
                signal = "SELL - Stop-Loss Hit"
            elif current_price >= target:
                signal = "SELL 50% - Partial Target Reached"
            elif pd.notna(last.get("SMA50")) and current_price < last["SMA50"]:
                signal = "CONSIDER EXIT - Below 50 SMA"
            else:
                signal = "HOLD"

            days_held = None
            try:
                entry_date = pd.to_datetime(pos["Entry Date"])
                idx = df.index[-1]
                if getattr(idx, "tzinfo", None):
                    idx = idx.tz_localize(None)
                if getattr(entry_date, "tzinfo", None):
                    entry_date = entry_date.tz_localize(None)
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
                "Suggested Trail": round(suggested_stop, 2),
                "Target": target,
                "Signal": signal,
            })
        except Exception as e:
            out_rows.append({"Symbol": sym, "Signal": f"Error fetching data: {e}"})

    return pd.DataFrame(out_rows)

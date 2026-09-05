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

DEPLOYMENT: replace the ENTIRE contents of your repo's engine.py with this
file's contents (same for app.py). After several mismatched-deploy incidents
this session, ENGINE_VERSION below exists specifically so you can confirm a
redeploy actually took -- check the sidebar footer against this string.
"""

ENGINE_VERSION = "engine-2026-09-03-l-diagnostics"

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

BENCHMARK_TICKER = "^NSEI"

STABLE_UNIVERSE = [
    # Nifty 50 core
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
    # Nifty Next 50 / other verified large-caps -- widens the "stable" bucket
    # without guessing at names. Verified via NSE/index-factsheet sources.
    "ADANIPOWER.NS", "HAL.NS", "TVSMOTOR.NS", "VBL.NS", "TATAPOWER.NS",
    "CHOLAFIN.NS", "VEDL.NS", "ADANIGREEN.NS", "ADANIENSOL.NS", "PIDILITIND.NS",
    "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "DABUR.NS", "COLPAL.NS",
    "GODREJCP.NS", "BANKBARODA.NS", "CANBK.NS", "PNB.NS", "LTIM.NS",
    "NAUKRI.NS", "ZYDUSLIFE.NS", "UNITDSPR.NS", "SHREECEM.NS", "AMBUJACEM.NS",
    "DMART.NS", "ICICIPRULI.NS", "ICICIGI.NS", "SBICARD.NS", "HDFCAMC.NS",
]

DYNAMIC_UNIVERSE = [
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

# Backward-compatible aliases -- older code/screenshots referenced these names.
DEFAULT_UNIVERSE = STABLE_UNIVERSE
MIDSMALLCAP_UNIVERSE = DYNAMIC_UNIVERSE

# CAVEAT, worth repeating: both lists are TODAY's known liquid, established
# names, tested against PAST years. Stocks that got delisted or crashed out
# of relevance in the meantime aren't included -- this is survivorship bias,
# and it applies to any curated or index-based universe, not just this one.
# We deliberately did NOT expand to a claimed "Nifty 500" list: hardcoding
# 500 tickers from memory risks silently wrong/delisted symbols, and a single
# yf.download() call across 500 tickers risks rate-limiting or timing out on
# Streamlit Cloud's free tier. This ~80-stock-per-bucket expansion roughly
# doubles the previous universe with names we can actually stand behind.

DEFAULT_PARAMS = dict(
    # Risk / exit
    atr_stop_mult=1.5,
    breakeven_r=1.0,
    partial_r=2.0,
    runner_trail_mult=2.0,
    hold_days=20,
    friction_pct=0.0015,
    gap_slippage_frac=0.15,  # extra slippage = 15% of the entry candle's gap-up size,
                              # on top of flat friction -- breakout days gap more than average
    min_breadth_pct=None,  # None = off. If set (e.g. 40), skips entries on days where
                            # fewer than this % of the universe is above its own 50-SMA,
                            # even if Nifty's own trend still looks fine. OFF by default --
                            # untested until proven via the walk-forward/Agent tabs, same
                            # discipline as everything else added tonight.

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


def compute_historical_breadth(universe: list, fetched_data) -> pd.Series:
    """
    UNLIKE get_market_breadth (a live snapshot), this is a real time series --
    for every date in the data, what % of `universe` closed above its OWN
    50-day SMA on THAT date, using only that date's trailing window. No
    future information involved, so -- unlike the liquidity/RS active-universe
    selection -- this one IS safe to use inside the backtest.

    Returns a Series indexed by date, values 0-100. Used as an additional,
    optional market-condition gate: skip new entries on days where breadth is
    weak, even if Nifty's own single-index trend still looks fine (a market
    can be "Nifty up" while only a handful of stocks are actually working).
    """
    per_stock_above = {}
    for sym in universe:
        try:
            df = _extract_symbol_frame(fetched_data, sym, len(universe))
            if len(df) < 210:  # same bar as everywhere else -- don't let thin/gappy
                continue        # stocks that couldn't even be traded dilute the reading
            sma50 = df["Close"].rolling(50).mean()
            # FIX: (Close > NaN) silently evaluates to False in pandas, not "unknown" --
            # that was quietly counting warmup periods and data gaps as "below its
            # average" instead of excluding them. .where(sma50.notna()) keeps the real
            # comparison where sma50 exists and sets NaN everywhere else, so skipna=True
            # below actually skips them instead of treating them as bearish.
            above = (df["Close"] > sma50).where(sma50.notna())
            per_stock_above[sym] = above
        except Exception:
            continue

    if not per_stock_above:
        return pd.Series(dtype=float)

    aligned = pd.DataFrame(per_stock_above).sort_index()
    breadth_pct = aligned.mean(axis=1, skipna=True) * 100
    return breadth_pct.rename("breadth_pct")


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

def backtest_symbol(df: pd.DataFrame, market_regime: pd.Series, params: dict, breadth: pd.Series = None) -> list:
    trades = []
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    breadth = breadth if breadth is not None else pd.Series(dtype=float)

    i = 0
    while i < len(df) - 2:
        row = df.iloc[i]
        date = row["Date"]
        mkt_ok = bool(market_regime.get(date, False))

        min_breadth = params.get("min_breadth_pct")
        if min_breadth and not breadth.empty:
            # asof = most recent breadth reading AT OR BEFORE this date. Robust to
            # breadth's index (a union of many stocks' individual date sets) not
            # having an exact match for every single date -- unlike an exact-match
            # .get(), which would silently fail-closed (block every trade) on any
            # tiny misalignment, exactly the kind of bug that can zero out an
            # entire backtest without an obvious error anywhere.
            try:
                today_breadth = breadth.asof(date)
            except Exception:
                today_breadth = None
            if pd.notna(today_breadth) and today_breadth < min_breadth:
                mkt_ok = False  # market too narrow today, even if Nifty itself looks fine
            # If today_breadth is NaN/unavailable, fail OPEN -- don't let a data gap
            # silently block every trade. The gate should narrow results when it has
            # real data, not erase the whole backtest when it doesn't.

        result = score_setup(row, mkt_ok, params, include_earnings=False)

        if result["mandatory_ok"] and result["score"] >= params.get("score_threshold", 78):
            entry_day = df.iloc[i + 1]
            gap_pct = max(0.0, (entry_day["Open"] / row["Close"] - 1) * 100) if row["Close"] else 0.0
            effective_friction = params["friction_pct"] + gap_pct / 100 * params.get("gap_slippage_frac", 0.0)
            entry_price = entry_day["Open"] * (1 + effective_friction)
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
            frame = _flatten_single_ticker_columns(data, sym).dropna(how="all")
        else:
            frame = data[sym].dropna(how="all")
        # Normalize timezone HERE, once, at the single extraction point every
        # downstream consumer uses (breadth calc, backtest loop, indicators).
        # If even one symbol among many comes back timezone-aware while
        # others don't, building a combined index later (e.g. breadth's
        # union across all stocks) can silently degrade to object dtype --
        # which breaks exact/asof lookups without ever raising a visible
        # error. Stripping tz at the source prevents that whole bug class.
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame = frame.copy()
            frame.index = frame.index.tz_localize(None)
        return frame
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# LIVE-ONLY: liquidity + relative-strength ranking, for auto-narrowing a pool
# -----------------------------------------------------------------------------
#
# IMPORTANT DESIGN NOTE, worth reading before touching this section:
# This ranks TODAY's most liquid, strongest-momentum names out of a larger
# candidate pool -- exactly the "let the AI pick the best names so I don't
# have to look at all of them" idea. It is wired into the LIVE screener only
# (screen_today), never into run_backtest / run_walk_forward_backtest /
# run_auto_optimize. Reason: ranking by TODAY's turnover and momentum and
# then using that ranked list to pick which stocks to backtest over the past
# few years would silently reintroduce a lookahead/survivorship bias --
# today's leaders were not necessarily leaders two years ago. Backtesting
# stays on the full static pool; only the live "what should I look at today"
# step gets the dynamic narrowing.

def compute_liquidity_and_rs(pool: list, lookback_days: int = 90) -> pd.DataFrame:
    """
    For each ticker in `pool`, computes:
      - avg_turnover_20d: 20-day average of Close x Volume (a liquidity proxy)
      - rs_50d: 50-day stock return minus Nifty's 50-day return (momentum,
        same "margin not ratio" logic as add_relative_strength -- robust to
        negative return periods)
    Returns a DataFrame with one row per ticker that had enough history.
    """
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)
    data = yf.download(pool, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)

    rows = []
    for sym in pool:
        try:
            df = _extract_symbol_frame(data, sym, len(pool))
            if len(df) < 55:
                continue
            turnover20 = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
            if len(df) >= 51:
                stock_ret50 = df["Close"].iloc[-1] / df["Close"].iloc[-51] - 1
            else:
                stock_ret50 = np.nan
            aligned = nifty_close.reindex(df.index, method="ffill")
            if len(aligned) >= 51 and pd.notna(aligned.iloc[-51]):
                nifty_ret50 = aligned.iloc[-1] / aligned.iloc[-51] - 1
            else:
                nifty_ret50 = np.nan
            rs50 = (stock_ret50 - nifty_ret50) if pd.notna(stock_ret50) and pd.notna(nifty_ret50) else np.nan
            rows.append({
                "symbol": sym,
                "avg_turnover_20d": turnover20,
                "rs_50d": rs50,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def select_active_universe(pool: list, top_n: int, min_turnover_percentile: float = 0.0) -> tuple:
    """
    LIVE-ONLY. Ranks `pool` by 20-day liquidity and 50-day relative strength,
    optionally drops the bottom `min_turnover_percentile` of names by
    liquidity (illiquid/dormant), then returns the top `top_n` tickers by
    relative strength -- today's "active universe" for screening.

    Returns (active_tickers, ranked_df) so the caller can show the ranking
    table for transparency, not just the final narrowed list.
    """
    ranked = compute_liquidity_and_rs(pool)
    if ranked.empty:
        return pool[:top_n], ranked  # fallback: fixed order if ranking failed entirely

    if min_turnover_percentile > 0:
        cutoff = ranked["avg_turnover_20d"].quantile(min_turnover_percentile)
        ranked = ranked[ranked["avg_turnover_20d"] >= cutoff]

    ranked = ranked.dropna(subset=["rs_50d"]).sort_values("rs_50d", ascending=False).reset_index(drop=True)
    active = ranked["symbol"].head(top_n).tolist()
    return active, ranked


def debug_breadth_application(universe: list, years: float = 3) -> dict:
    """
    diagnose_breadth() checks whether the breadth SERIES itself is healthy.
    This checks something different and more specific: whether the actual
    per-row lookup performed inside backtest_symbol -- breadth.asof(date),
    once per signal day, per symbol -- is actually succeeding, using the
    EXACT SAME data objects the real backtest loop uses. If the series is
    healthy but this lookup silently fails (wrong dtype, tz mismatch, etc.),
    the gate can compute correctly and still never restrict a single trade,
    because our fail-open design swallows lookup failures by design (to
    avoid the earlier "blocks everything" bug) -- which also means a broken
    lookup fails SILENTLY instead of loudly. This is how we'd catch that.
    """
    fetched = _fetch_universe_data(universe, years)
    breadth = fetched.get("breadth", pd.Series(dtype=float))
    if breadth.empty:
        return {"error": "Breadth series is empty -- can't test lookups against nothing."}

    prepared = _prepare_symbol_frames(universe, fetched)
    if not prepared:
        return {"error": "No prepared symbol frames -- can't run the test."}

    sym, df = next(iter(prepared.items()))
    df2 = df.reset_index()
    if "Date" not in df2.columns:
        df2.rename(columns={df2.columns[0]: "Date"}, inplace=True)

    total, succeeded, exceptions = 0, 0, 0
    sample_results = []
    n_check = min(len(df2), 300)
    for i in range(n_check):
        date = df2.iloc[i]["Date"]
        total += 1
        try:
            val = breadth.asof(date)
            ok = pd.notna(val)
            if ok:
                succeeded += 1
            if i < 5 or i >= n_check - 5:
                sample_results.append({"date": str(date), "breadth_value": val if ok else None})
        except Exception as e:
            exceptions += 1
            if len(sample_results) < 10:
                sample_results.append({"date": str(date), "error": f"{type(e).__name__}: {e}"})

    return {
        "symbol_tested": sym,
        "total_checked": total,
        "succeeded": succeeded,
        "failed_or_nan": total - succeeded - exceptions,
        "raised_exception": exceptions,
        "date_column_dtype": str(df2["Date"].dtype),
        "breadth_index_dtype": str(breadth.index.dtype),
        "sample_results": sample_results,
    }


def diagnose_breadth(universe: list, years: float = 3) -> dict:
    """
    Diagnostic tool -- computes the actual historical breadth series and
    returns real statistics about it, so a genuinely-strict reading can be
    told apart from a lookup/coverage bug without guessing. Call this from
    the UI and look at the numbers directly rather than inferring from
    downstream symptoms (like the agent finding zero valid configurations).
    """
    fetched = _fetch_universe_data(universe, years)
    breadth = fetched.get("breadth", pd.Series(dtype=float))
    if breadth.empty:
        return {"error": "Breadth series is completely empty -- computation itself failed, not just strict."}

    valid = breadth.dropna()
    return {
        "n_dates_total": len(breadth),
        "n_dates_valid": len(valid),
        "n_dates_nan": len(breadth) - len(valid),
        "min_pct": float(valid.min()) if len(valid) else None,
        "max_pct": float(valid.max()) if len(valid) else None,
        "mean_pct": float(valid.mean()) if len(valid) else None,
        "median_pct": float(valid.median()) if len(valid) else None,
        "pct_days_below_40": float((valid < 40).mean() * 100) if len(valid) else None,
        "pct_days_below_30": float((valid < 30).mean() * 100) if len(valid) else None,
        "last_10_values": valid.tail(10).round(1).to_dict(),
        "n_stocks_in_universe": len(universe),
    }


def get_market_breadth(universe: list) -> dict:
    """
    LIVE-ONLY, informational (not a hard gate, and its 60%/30% labels are
    illustrative, NOT walk-forward validated -- treat as context, the same
    way a human trader glances at "how many stocks are trending" before
    deciding how aggressive to be today).

    % of `universe` currently trading above their own 50-day SMA -- a simple,
    well-established regime-strength gauge (breadth), independent of Nifty's
    own single-index trend. A market can be "Nifty above its 50-SMA" while
    breadth underneath is narrow (a handful of large stocks propping up the
    index) -- breadth catches that, a single-index check can't.
    """
    start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    data = yf.download(universe, start=start, group_by="ticker", auto_adjust=True,
                        progress=False, threads=True)
    above, total = 0, 0
    for sym in universe:
        try:
            df = _extract_symbol_frame(data, sym, len(universe))
            if len(df) < 55:
                continue
            sma50 = df["Close"].rolling(50).mean().iloc[-1]
            if pd.isna(sma50):
                continue
            total += 1
            if df["Close"].iloc[-1] > sma50:
                above += 1
        except Exception:
            continue

    if total == 0:
        return {}

    breadth_pct = above / total * 100
    if breadth_pct >= 60:
        regime_label = "Aggressive -- breadth > 60%, most of the universe is trending up"
    elif breadth_pct >= 30:
        regime_label = "Selective -- breadth 30-60%, a mixed market, be choosier"
    else:
        regime_label = "Defensive -- breadth < 30%, most of the universe is NOT trending up"

    return {"breadth_pct": breadth_pct, "n_stocks": total, "n_above": above, "regime_label": regime_label}


def _fetch_universe_data(universe, years) -> dict:
    """Network fetch only -- prices + Nifty regime data. Split out from trade
    generation so the auto-optimize agent can fetch ONCE and re-use the same
    data across many parameter combinations, instead of re-downloading for
    every candidate it tests."""
    start = (datetime.now() - timedelta(days=365 * years + 260)).strftime("%Y-%m-%d")
    nifty_df = get_nifty_data(start)
    market_regime = get_market_regime(nifty_df)
    nifty_close = nifty_df["Close"] if not nifty_df.empty else pd.Series(dtype=float)
    data = yf.download(
        universe, start=start, group_by="ticker", auto_adjust=True,
        progress=False, threads=True
    )
    breadth = compute_historical_breadth(universe, data)
    return {"market_regime": market_regime, "nifty_close": nifty_close, "data": data, "breadth": breadth}


def _prepare_symbol_frames(universe, fetched: dict, rs_lookback: int = 63) -> dict:
    """
    Computes indicators + relative strength ONCE per symbol. Neither depends
    on score_threshold, hold_days, or atr_stop_mult -- the three things the
    auto-optimize agent actually searches over -- so doing this once and
    reusing it across all 54 grid combinations (instead of recomputing it 54
    times) is the single biggest available speedup for the agent. Returns
    {symbol: prepared_dataframe}, skipping symbols with too little history.
    """
    nifty_close = fetched["nifty_close"]
    data = fetched["data"]
    prepared = {}
    for sym in universe:
        try:
            df = _extract_symbol_frame(data, sym, len(universe))
            if len(df) < 210:
                continue
            df = compute_indicators(df)
            df = add_relative_strength(df, nifty_close, rs_lookback)
            prepared[sym] = df
        except Exception:
            continue
    return prepared


def _run_backtest_on_prepared(universe, prepared: dict, market_regime, breadth, params) -> pd.DataFrame:
    """The cheap, params-DEPENDENT part -- runs backtest_symbol against
    already-prepared (indicators computed) dataframes. Safe to call many
    times with different params once _prepare_symbol_frames has run once."""
    all_trades = []
    for sym in universe:
        df = prepared.get(sym)
        if df is None:
            continue
        try:
            trades = backtest_symbol(df, market_regime, params, breadth)
            for t in trades:
                t["symbol"] = sym.replace(".NS", "")
            all_trades.extend(trades)
        except Exception:
            continue

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        return trades_df

    if "signal_date" in trades_df.columns:
        trades_df["signal_date"] = pd.to_datetime(trades_df["signal_date"])
        trades_df = (
            trades_df.sort_values(["signal_date", "setup_score", "r_multiple"], ascending=[True, False, False])
            .groupby("signal_date", group_keys=False)
            .head(int(params.get("max_trades_per_day", 3)))
            .sort_values("entry_date")
            .reset_index(drop=True)
        )
    return trades_df.sort_values("entry_date").reset_index(drop=True)


def _generate_trades_from_fetched(universe, fetched: dict, params) -> pd.DataFrame:
    """Pure computation, no network calls -- single-combination convenience
    wrapper used by run_backtest / run_walk_forward_backtest (which only ever
    need ONE parameter combination, so the prepare/run split doesn't matter
    there -- it matters for the agent, which needs 54)."""
    market_regime = fetched["market_regime"]
    breadth = fetched.get("breadth", pd.Series(dtype=float))
    prepared = _prepare_symbol_frames(universe, fetched, params.get("rs_lookback", 63))
    return _run_backtest_on_prepared(universe, prepared, market_regime, breadth, params)


def _generate_all_trades(universe, years, params) -> pd.DataFrame:
    """Core trade generation, shared by run_backtest and the walk-forward split.
    Indicators need continuous lookback history, so this always runs across the
    FULL requested window -- splitting into in-sample/out-of-sample happens
    afterward, on the resulting trades, not by truncating the price history."""
    fetched = _fetch_universe_data(universe, years)
    return _generate_trades_from_fetched(universe, fetched, params)


def _summarize_trades(trades_df: pd.DataFrame, params: dict) -> dict:
    if trades_df.empty:
        return {}
    wins = trades_df[trades_df.outcome == "win"]
    losses = trades_df[trades_df.outcome == "loss"]
    gross_win = wins.r_multiple.sum() if len(wins) else 0
    gross_loss_abs = abs(losses.r_multiple.sum()) if len(losses) else 0
    return {
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


def run_backtest(universe, years, params, return_candidates=False) -> tuple[pd.DataFrame, dict]:
    trades_df = _generate_all_trades(universe, years, params)
    return trades_df, _summarize_trades(trades_df, params)


def run_walk_forward_backtest(universe, years, params, out_sample_frac: float = 0.35, split_date: str = None) -> dict:
    """
    Splits the SAME set of generated trades into an in-sample (earlier) period
    and an out-of-sample (later) period, using the exact same frozen params for
    both -- nothing is re-tuned per half. If the out-of-sample half performs
    meaningfully worse than in-sample, that's a real warning sign the earlier
    tuning was fitting to noise in that specific window rather than finding
    something that generalizes.

    Two split modes:
      - split_date given (e.g. "2025-01-01"): FIXED calendar cutoff. Use this
        when comparing walk-forward results across different `years` settings --
        a % split floats with window length, so a 3-year and 5-year backtest
        end up testing different out-of-sample periods and aren't comparable.
        A fixed date answers the same question every time: "does this hold up
        on 2025-2026 specifically?"
      - split_date not given: falls back to the % split (out_sample_frac of
        trades, chronologically), useful for quick single-run checks.
    """
    trades_df = _generate_all_trades(universe, years, params)
    if trades_df.empty or len(trades_df) < 10:
        return {"trades_df": trades_df, "in_sample": {}, "out_sample": {}, "split_date": None}

    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

    if split_date:
        cutoff = pd.to_datetime(split_date)
        in_sample_df = trades_df[trades_df["entry_date"] < cutoff].reset_index(drop=True)
        out_sample_df = trades_df[trades_df["entry_date"] >= cutoff].reset_index(drop=True)
        actual_split = cutoff
    else:
        split_idx = int(len(trades_df) * (1 - out_sample_frac))
        split_idx = max(1, min(split_idx, len(trades_df) - 1))
        actual_split = trades_df.iloc[split_idx]["entry_date"]
        in_sample_df = trades_df.iloc[:split_idx].reset_index(drop=True)
        out_sample_df = trades_df.iloc[split_idx:].reset_index(drop=True)

    return {
        "trades_df": trades_df,
        "in_sample_df": in_sample_df,
        "out_sample_df": out_sample_df,
        "in_sample": _summarize_trades(in_sample_df, params),
        "out_sample": _summarize_trades(out_sample_df, params),
        "split_date": actual_split,
    }

# -----------------------------------------------------------------------------
# AUTO-OPTIMIZE AGENT
# -----------------------------------------------------------------------------

# Deliberately small and curated -- these are the levers that showed real,
# mechanism-backed effects across a long night of manual testing. Adding a
# 3rd dimension (stop distance) roughly doubles the grid; that's the bound
# we're comfortable with. Going to 5-6 dimensions (thousands of combinations)
# turns "search" into "guaranteed to find noise that looks like edge" -- see
# the profit-factor swing from a SINGLE parameter change this session
# (0.90 -> 0.98 -> 0.96 just from hold_days) for why that risk is real, not
# theoretical.
AUTO_OPTIMIZE_GRID = {
    "score_threshold": [72, 76, 80, 83, 86, 90],
    "hold_days": [20, 30, 40],
    "atr_stop_mult": [1.5, 2.0, 2.5],
}

# Faster option: every other score value, same hold/stop range -- 18 combos
# instead of 54, for a quicker first look (e.g. if Streamlit Cloud's free
# tier struggles with the full search). Real search-space reduction, not
# just a progress-bar trick, so treat results from this as a rougher signal.
AUTO_OPTIMIZE_GRID_QUICK = {
    "score_threshold": [72, 80, 86],
    "hold_days": [20, 30, 40],
    "atr_stop_mult": [1.5, 2.0, 2.5],
}

# Mode presets: different starting risk posture per bucket, matching the
# "stable = tighter/higher win-rate, dynamic = wider/bigger winners" idea.
# These only set the STARTING params the grid search is applied on top of --
# the grid still searches score/hold/stop for whichever mode is selected.
MODE_PRESETS = {
    "stable": dict(breakeven_r=0.75, partial_r=1.75, runner_trail_mult=1.5),
    "dynamic": dict(breakeven_r=1.25, partial_r=3.0, runner_trail_mult=2.5),
}


def run_auto_optimize(
    universe, years, base_params, split_date,
    min_in_sample_trades: int = 15,
    min_out_sample_trades: int = 8,
    mode: str = None,
    quick: bool = False,
    progress_callback=None,
) -> dict:
    """
    Rule-based search agent -- no external AI service, fully deterministic and
    inspectable. Fetches price data ONCE, then evaluates a small curated grid
    of (score_threshold, hold_days, atr_stop_mult) combinations against it.

    mode: "stable" or "dynamic", optional. If given, overlays MODE_PRESETS
    onto base_params before searching (different default risk posture per
    bucket), matching the dual-bucket idea: stable favors tighter, more
    consistent exits; dynamic favors wider stops and bigger runners.

    quick: use the smaller 27-combination grid instead of the full 54 --
    trade-off search breadth for speed, useful if the full search is too
    slow for Streamlit Cloud's free tier.

    The critical discipline this enforces, which manual tuning this session
    did NOT follow: the winning configuration is selected using ONLY
    in-sample data (before split_date). The out-of-sample result for that
    winner is then reported as-is -- data the selection process never saw,
    so it's an honest read of whether the choice generalizes.

    progress_callback(done, total, score_threshold, hold_days, atr_stop_mult),
    if given, is called after each combination so a caller (e.g. a Streamlit
    progress bar) can show live progress during what can be a slow search.

    Returns {"leaderboard": [...], "best": {...} | None, "recommendation_text": str}
    """
    effective_base = dict(base_params)
    if mode in MODE_PRESETS:
        effective_base.update(MODE_PRESETS[mode])

    grid = AUTO_OPTIMIZE_GRID_QUICK if quick else AUTO_OPTIMIZE_GRID

    fetched = _fetch_universe_data(universe, years)
    # THE speedup: compute indicators once per symbol, reuse across every
    # combination below, instead of recomputing them inside every iteration.
    prepared = _prepare_symbol_frames(universe, fetched, effective_base.get("rs_lookback", 63))
    market_regime = fetched["market_regime"]
    breadth = fetched.get("breadth", pd.Series(dtype=float))

    combos = [
        (st_, hd, sm)
        for st_ in grid["score_threshold"]
        for hd in grid["hold_days"]
        for sm in grid["atr_stop_mult"]
    ]

    results = []
    for idx, (score_threshold, hold_days, atr_stop_mult) in enumerate(combos):
        params = dict(effective_base)
        params["score_threshold"] = score_threshold
        params["hold_days"] = hold_days
        params["atr_stop_mult"] = atr_stop_mult

        trades_df = _run_backtest_on_prepared(universe, prepared, market_regime, breadth, params)
        if progress_callback:
            progress_callback(idx + 1, len(combos), score_threshold, hold_days, atr_stop_mult)
        if trades_df.empty:
            continue

        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        cutoff = pd.to_datetime(split_date)
        in_df = trades_df[trades_df["entry_date"] < cutoff].reset_index(drop=True)
        out_df = trades_df[trades_df["entry_date"] >= cutoff].reset_index(drop=True)
        ins = _summarize_trades(in_df, params)
        oos = _summarize_trades(out_df, params)

        if not ins or ins.get("total_trades", 0) < min_in_sample_trades:
            continue  # not enough in-sample evidence to trust this candidate at all

        results.append({
            "score_threshold": score_threshold,
            "hold_days": hold_days,
            "atr_stop_mult": atr_stop_mult,
            "in_sample_trades": ins.get("total_trades", 0),
            "in_sample_win_rate": ins.get("win_rate", 0.0),
            "in_sample_expectancy": ins.get("expectancy_r", 0.0),
            "in_sample_pf": ins.get("profit_factor", 0.0),
            "out_sample_trades": oos.get("total_trades", 0) if oos else 0,
            "out_sample_win_rate": oos.get("win_rate", 0.0) if oos else 0.0,
            "out_sample_expectancy": oos.get("expectancy_r", 0.0) if oos else 0.0,
            "out_sample_pf": oos.get("profit_factor", 0.0) if oos else 0.0,
        })

    if not results:
        return {
            "leaderboard": [],
            "best": None,
            "recommendation_text": (
                "No configuration in the search grid produced enough in-sample trades "
                "to evaluate reliably. Try widening the universe, or lowering the split "
                "date so more history counts as in-sample."
            ),
        }

    # SELECTION uses ONLY in-sample data. Out-of-sample is never consulted
    # here -- selecting by out-of-sample performance would contaminate the
    # one honest test we have (proven by this session's own leaderboard: the
    # config that "won" on out-of-sample alone was a mediocre in-sample
    # performer -- picking it would just relocate the overfitting problem,
    # not fix it).
    #
    # STABILITY ADJUSTMENT: rather than picking the single best in-sample
    # point (which can be a lucky spike -- a narrow combination that looked
    # great by chance), each candidate's score also rewards its immediate
    # grid neighbors also looking decent. A real, broad effect shows up
    # across nearby parameter values; a curve-fit spike doesn't.
    by_key = {(r["score_threshold"], r["hold_days"], r["atr_stop_mult"]): r for r in results}

    def _stability_adjusted_score(r):
        neighbors = []
        for d_score in grid["score_threshold"]:
            for d_hold in grid["hold_days"]:
                for d_stop in grid["atr_stop_mult"]:
                    key = (d_score, d_hold, d_stop)
                    if key == (r["score_threshold"], r["hold_days"], r["atr_stop_mult"]):
                        continue
                    # "adjacent" = differs in exactly one dimension by one grid step
                    diffs = sum([
                        d_score != r["score_threshold"],
                        d_hold != r["hold_days"],
                        d_stop != r["atr_stop_mult"],
                    ])
                    if diffs == 1 and key in by_key:
                        neighbors.append(by_key[key]["in_sample_expectancy"])
        neighbor_avg = sum(neighbors) / len(neighbors) if neighbors else r["in_sample_expectancy"]
        # Blend: mostly the candidate's own result, partly how its neighbors did.
        return 0.7 * r["in_sample_expectancy"] + 0.3 * neighbor_avg

    for r in results:
        r["stability_adjusted_score"] = _stability_adjusted_score(r)

    leaderboard = sorted(results, key=lambda r: r["stability_adjusted_score"], reverse=True)
    best = leaderboard[0]

    oos_n = best["out_sample_trades"]
    oos_exp = best["out_sample_expectancy"]
    ins_exp = best["in_sample_expectancy"]

    if oos_n < min_out_sample_trades:
        verdict = (
            f"Only {oos_n} out-of-sample trades since {split_date} -- too few to trust "
            "either way yet. Treat this as a promising starting point, not a proven system. "
            "Keep tracking it forward before risking real money."
        )
    elif oos_exp >= ins_exp - 0.05 and oos_exp > 0:
        verdict = (
            f"These settings held up on data they never saw during selection: "
            f"{oos_n} trades since {split_date}, {best['out_sample_win_rate']:.0f}% win rate, "
            f"{oos_exp:+.2f}R expectancy per trade. That's a real, if modest, sign this "
            "generalizes rather than being fit to noise in the tuning period."
        )
    elif oos_exp > 0:
        verdict = (
            f"Still profitable out-of-sample ({oos_exp:+.2f}R over {oos_n} trades) but "
            f"noticeably weaker than during selection ({ins_exp:+.2f}R). Treat this with "
            "some caution -- part of the in-sample result may have been luck."
        )
    else:
        verdict = (
            f"Out-of-sample this configuration was NOT profitable ({oos_exp:+.2f}R over "
            f"{oos_n} trades), even though it looked good during selection ({ins_exp:+.2f}R). "
            "This is the overfitting warning sign -- don't trust this configuration yet."
        )

    recommendation_text = (
        f"Best configuration found: quality score \u2265 {best['score_threshold']}, "
        f"maximum hold {best['hold_days']} trading days, stop-loss {best['atr_stop_mult']}x ATR.\n\n"
        f"Chosen using only data before {split_date}, favoring configurations whose nearby "
        f"parameter values also looked decent (not just a single lucky spike): "
        f"{best['in_sample_trades']} trades, {best['in_sample_win_rate']:.0f}% win rate, "
        f"{ins_exp:+.2f}R expectancy.\n\n"
        f"{verdict}"
    )

    return {"leaderboard": leaderboard, "best": best, "recommendation_text": recommendation_text}


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

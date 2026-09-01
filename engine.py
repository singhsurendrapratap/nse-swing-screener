import numpy as np
import pandas as pd
import yfinance as yf

ENGINE_VERSION = "engine-2026-09-01-analytics-v3"

DEFAULT_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LTIM.NS", "LT.NS", "AXISBANK.NS",
]

MIDSMALLCAP_UNIVERSE = [
    "POLYCAB.NS", "DIXON.NS", "KEI.NS", "PERSISTENT.NS", "COFORGE.NS",
    "KPITTECH.NS", "TATAELXSI.NS", "SUZLON.NS", "BSOFT.NS", "MAZDOCK.NS",
]

DEFAULT_PARAMS = {
    "score_threshold": 78,
    "max_trades_per_day": 3,
    "atr_stop_mult": 1.5,
    "breakeven_r": 1.0,
    "partial_r": 2.0,
    "runner_trail_mult": 2.0,
    "rsi_low": 45,
    "rsi_high": 70,
    "min_earnings_growth": 0.10,
    "hold_days": 20,
    "friction_pct": 0.0015,
}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["SMA200"] = close.rolling(window=200).mean()

    tr = np.maximum(
        high - low,
        np.maximum(
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ),
    )
    df["ATR"] = tr.rolling(window=14).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Vol20"] = volume.rolling(window=20).mean()
    df["VolRatio"] = volume / (df["Vol20"] + 1e-9)
    df["High20"] = high.rolling(window=20).max()
    df["Low20"] = low.rolling(window=20).min()
    df["Contraction"] = (df["High20"] - df["Low20"]) / (df["EMA20"] + 1e-9)

    return df


def compute_score(row: pd.Series, params: dict) -> float:
    score = 0.0
    if row["Close"] > row["EMA20"]: score += 10
    if row["EMA20"] > row["EMA50"]: score += 10
    if pd.notna(row["SMA200"]) and row["Close"] > row["SMA200"]: score += 10

    if row["Close"] >= row["High20"] * 0.99: score += 15
    if row["Contraction"] < 0.12: score += 15

    if row["VolRatio"] > 1.5: score += 20
    elif row["VolRatio"] > 1.0: score += 10

    rsi_low = params.get("rsi_low", 45)
    rsi_high = params.get("rsi_high", 70)
    if rsi_low <= row["RSI"] <= rsi_high: score += 20

    return min(score, 100.0)


def download_data(symbol: str, years: int) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=f"{years}y", progress=False)
        if df.empty or len(df) < 200: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return compute_indicators(df)
    except Exception:
        return pd.DataFrame()


def check_market_regime() -> bool:
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
        if nifty.empty: return True
        if isinstance(nifty.columns, pd.MultiIndex): nifty.columns = nifty.columns.get_level_values(0)
        nifty = compute_indicators(nifty)
        last = nifty.iloc[-1]
        return bool(last["Close"] > last["EMA20"] and last["Close"] > last["EMA50"])
    except Exception:
        return True


def screen_today(universe: list, capital: float, risk_pct: float, params: dict):
    market_bullish = check_market_regime()
    watchlist = []

    for sym in universe:
        df = download_data(sym, 1)
        if df.empty: continue
        last = df.iloc[-1]
        score = compute_score(last, params)

        if score >= params.get("score_threshold", 78):
            atr = last["ATR"]
            close = last["Close"]
            stop = close - (atr * params.get("atr_stop_mult", 1.5))
            risk_per_share = close - stop
            if risk_per_share <= 0: continue

            max_risk_amount = capital * risk_pct
            qty = max(1, int(max_risk_amount / risk_per_share))
            target = close + (risk_per_share * params.get("partial_r", 2.0))

            watchlist.append({
                "Symbol": sym,
                "Score": round(score, 1),
                "Close": round(close, 2),
                "ATR": round(atr, 2),
                "Stop": round(stop, 2),
                "Target": round(target, 2),
                "Position Size (Qty)": qty,
                "Risk/Trade (Rs)": round(qty * risk_per_share, 2),
            })

    res_df = pd.DataFrame(watchlist)
    if not res_df.empty:
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return res_df, market_bullish


def run_backtest(universe: list, years: int, params: dict):
    all_trades = []

    for sym in universe:
        df = download_data(sym, years)
        if df.empty: continue

        for i in range(200, len(df) - 1):
            row = df.iloc[i]
            score = compute_score(row, params)

            if score >= params.get("score_threshold", 78):
                entry_date = df.index[i]
                entry_price = df.iloc[i + 1]["Open"]
                atr = row["ATR"]
                stop_dist = atr * params.get("atr_stop_mult", 1.5)
                stop_price = entry_price - stop_dist

                if stop_dist <= 0: continue

                hold_days = params.get("hold_days", 20)
                exit_price = entry_price
                exit_date = entry_date

                for j in range(i + 1, min(i + 1 + hold_days, len(df))):
                    future_row = df.iloc[j]
                    if future_row["Low"] <= stop_price:
                        exit_price = stop_price
                        exit_date = df.index[j]
                        break
                    else:
                        exit_price = future_row["Close"]
                        exit_date = df.index[j]

                r_mult = (exit_price - entry_price) / stop_dist
                r_mult -= params.get("friction_pct", 0.0015)

                all_trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "score": round(score, 1),
                    "r_multiplier": round(r_mult, 2),
                })

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty: return trades_df, {}

    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    trades_df["cum_r"] = trades_df["r_multiplier"].cumsum()

    # Calculate Drawdown metrics
    cum = trades_df["cum_r"]
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = drawdown.min()

    # Calculate Win/Loss statistics
    n_trades = len(trades_df)
    wins = trades_df[trades_df["r_multiplier"] > 0]["r_multiplier"]
    losses = trades_df[trades_df["r_multiplier"] <= 0]["r_multiplier"]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

    win_rate = (len(wins) / n_trades) * 100
    total_r = trades_df["r_multiplier"].sum()
    expectancy_r = trades_df["r_multiplier"].mean()

    summary = {
        "total_trades": n_trades,
        "win_rate": win_rate,
        "expectancy_r": expectancy_r,
        "total_r": total_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_dd,
        "avg_win_r": wins.mean() if not wins.empty else 0.0,
        "avg_loss_r": losses.mean() if not losses.empty else 0.0,
    }

    return trades_df, summary


def run_optimizer_backtest(trades_df: pd.DataFrame, params: dict) -> dict:
    if trades_df.empty: return {"expectancy_r": -999.0, "total_r": -999.0, "win_rate": 0.0}
    filtered = trades_df[trades_df["score"] >= params.get("score_threshold", 78)]
    if filtered.empty: return {"expectancy_r": -999.0, "total_r": -999.0, "win_rate": 0.0}

    base_atr_mult = 1.5
    adj_factor = base_atr_mult / params.get("atr_stop_mult", 1.5)
    adj_r = filtered["r_multiplier"] * adj_factor

    n_trades = len(adj_r)
    wins = (adj_r > 0).sum()

    return {
        "expectancy_r": float(adj_r.mean()),
        "total_r": float(adj_r.sum()),
        "win_rate": float(wins / n_trades) if n_trades > 0 else 0.0,
    }


def run_walk_forward_backtest(universe: list, years: int, params: dict, split_date: str) -> dict:
    trades_df, _ = run_backtest(universe, years, params)
    if trades_df.empty: return {}

    split_dt = pd.to_datetime(split_date)
    in_sample = trades_df[trades_df["entry_date"] < split_dt]
    out_sample = trades_df[trades_df["entry_date"] >= split_dt]

    def get_stats(df_sub):
        if df_sub.empty: return {"total_trades": 0, "expectancy_r": 0.0, "total_r": 0.0, "win_rate": 0.0}
        n = len(df_sub)
        w = (df_sub["r_multiplier"] > 0).sum()
        return {
            "total_trades": n,
            "expectancy_r": float(df_sub["r_multiplier"].mean()),
            "total_r": float(df_sub["r_multiplier"].sum()),
            "win_rate": float(w / n) * 100,
        }

    return {"in_sample": get_stats(in_sample), "out_sample": get_stats(out_sample)}


def evaluate_positions(pos_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    results = []
    for _, row in pos_df.iterrows():
        sym = str(row["Symbol"]).strip()
        yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        try:
            df = yf.download(yf_sym, period="5d", progress=False)
            if df.empty: continue
            curr_price = float(df["Close"].iloc[-1])
            entry_p = float(row["Entry Price"])
            stop_p = float(row["Stop"])
            target_p = float(row["Target"])

            pnl_pct = ((curr_price - entry_p) / entry_p) * 100
            status = "HOLD"
            if curr_price >= target_p > 0: status = "TARGET HIT"
            elif curr_price <= stop_p > 0: status = "STOP HIT"

            results.append({
                "Symbol": sym,
                "Current Price": round(curr_price, 2),
                "PnL (%)": round(pnl_pct, 2),
                "Status": status,
            })
        except Exception:
            continue
    return pd.DataFrame(results)

"""
Automatic Daily Signal Logger
------------------------------
Purpose: build a genuine forward track record automatically, without
ever having to remember to save anything.

Call `log_todays_signals()` once per day (e.g. at market close, via a
scheduled job or the first Streamlit run of the day) and it will:
  1. Take whatever signals your strategy generated today
  2. Store them with a stable (date, symbol) key so re-runs never duplicate
  3. Leave "outcome" fields empty until the trade actually resolves

Call `update_open_outcomes()` on each run to fill in realized R for
signals that have since hit target/stop/time-exit, so the track record
fills itself in going forward without manual entry.
"""

import sqlite3
import json
from datetime import date
import pandas as pd

DB_PATH = "signal_log.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry REAL,
            stop REAL,
            target REAL,
            quality_score REAL,
            params_json TEXT,
            status TEXT DEFAULT 'open',   -- open / win / loss / expired
            realized_r REAL,
            closed_date TEXT,
            UNIQUE(log_date, symbol)
        )
        """)


def log_todays_signals(signals_df: pd.DataFrame, params: dict):
    """
    signals_df columns expected: symbol, entry, stop, target, quality_score
    params: the strategy parameters active today. Worth storing per-signal
    so that once you're running AI-optimized params that change over time,
    you can trace exactly which parameter set produced which signal.
    """
    init_db()
    today = date.today().isoformat()
    with _conn() as c:
        for _, row in signals_df.iterrows():
            try:
                c.execute("""
                    INSERT INTO signals
                    (log_date, symbol, entry, stop, target, quality_score, params_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (today, row["symbol"], row["entry"], row["stop"],
                      row["target"], row["quality_score"], json.dumps(params)))
            except sqlite3.IntegrityError:
                pass  # already logged today for this symbol -- no duplicates


def update_open_outcomes(price_lookup_fn, max_hold_days: int = None):
    """
    price_lookup_fn(symbol) -> latest price.
    Walks every still-open signal and closes it out if price has hit
    stop or target (extend with your own time-exit / trailing logic
    as needed). This is what makes the track record forward and
    unattended -- you never manually mark a trade won or lost.
    """
    init_db()
    today = date.today().isoformat()
    with _conn() as c:
        open_rows = c.execute(
            "SELECT id, symbol, entry, stop, target FROM signals WHERE status='open'"
        ).fetchall()
        for sid, symbol, entry, stop, target in open_rows:
            price = price_lookup_fn(symbol)
            if price is None:
                continue
            risk = abs(entry - stop) or 1e-9
            if price <= stop:
                c.execute(
                    "UPDATE signals SET status='loss', realized_r=?, closed_date=? WHERE id=?",
                    (-1.0, today, sid)
                )
            elif price >= target:
                r = (target - entry) / risk
                c.execute(
                    "UPDATE signals SET status='win', realized_r=?, closed_date=? WHERE id=?",
                    (r, today, sid)
                )


def get_track_record() -> pd.DataFrame:
    init_db()
    with _conn() as c:
        return pd.read_sql("SELECT * FROM signals ORDER BY log_date DESC", c)


def summary_stats() -> dict:
    """Quick forward-track-record stats -- the real, unfaked win rate."""
    df = get_track_record()
    closed = df[df["status"].isin(["win", "loss"])]
    if closed.empty:
        return {"n_trades": 0}
    return {
        "n_trades": len(closed),
        "win_rate": (closed["status"] == "win").mean(),
        "expectancy_r": closed["realized_r"].mean(),
        "total_r": closed["realized_r"].sum(),
    }

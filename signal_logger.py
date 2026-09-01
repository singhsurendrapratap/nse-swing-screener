import sqlite3
from datetime import date
import pandas as pd

DB_FILE = "signal_log.db"


def init_db():
    """Initializes the SQLite database table for signal tracking."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date TEXT,
                symbol TEXT,
                score REAL,
                close_price REAL,
                atr REAL,
                recommended_stop REAL,
                recommended_target REAL,
                recommended_qty INTEGER,
                params_hash TEXT,
                outcome_status TEXT DEFAULT 'OPEN',
                exit_price REAL,
                exit_date TEXT,
                r_multipler REAL,
                UNIQUE(signal_date, symbol)
            )
            """
        )


def log_todays_signals(watchlist_df: pd.DataFrame, params: dict):
    """Logs new candidates to the database."""
    init_db()
    if watchlist_df.empty:
        return

    params_hash = str(sorted(params.items()))
    today_str = str(date.today())

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for _, row in watchlist_df.iterrows():
            cursor.execute(
                """
                INSERT OR IGNORE INTO signal_log (
                    signal_date, symbol, score, close_price, atr,
                    recommended_stop, recommended_target, recommended_qty, params_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    today_str,
                    row.get("Symbol"),
                    row.get("Score"),
                    row.get("Close"),
                    row.get("ATR"),
                    row.get("Stop"),
                    row.get("Target"),
                    row.get("Position Size (Qty)"),
                    params_hash,
                ),
            )
        conn.commit()


def get_track_record() -> pd.DataFrame:
    """Fetches all stored signals from the SQLite database."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(
            "SELECT * FROM signal_log ORDER BY signal_date DESC", conn
        )


def update_open_outcomes(live_price_fetcher):
    """Updates open positions using a price-fetching callback."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM signal_log WHERE outcome_status = 'OPEN'", conn
        )
        if df.empty:
            return

        cursor = conn.cursor()
        for _, row in df.iterrows():
            sym = row["symbol"]
            entry_p = row["close_price"]
            stop_p = row["recommended_stop"]
            target_p = row["recommended_target"]
            row_id = row["id"]

            curr_price = live_price_fetcher(sym)
            if curr_price is None:
                continue

            r_unit = entry_p - stop_p
            if r_unit <= 0:
                continue

            # Check target and stop conditions
            if curr_price >= target_p:
                r_mult = (target_p - entry_p) / r_unit
                cursor.execute(
                    """
                    UPDATE signal_log
                    SET outcome_status = 'CLOSED_TARGET', exit_price = ?, exit_date = ?, r_multipler = ?
                    WHERE id = ?
                    """,
                    (target_p, str(date.today()), r_mult, row_id),
                )
            elif curr_price <= stop_p:
                r_mult = (stop_p - entry_p) / r_unit
                cursor.execute(
                    """
                    UPDATE signal_log
                    SET outcome_status = 'CLOSED_STOP', exit_price = ?, exit_date = ?, r_multipler = ?
                    WHERE id = ?
                    """,
                    (stop_p, str(date.today()), r_mult, row_id),
                )
        conn.commit()


def summary_stats() -> dict:
    """Calculates summary statistics across all closed signals."""
    init_db()
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM signal_log WHERE outcome_status != 'OPEN'", conn
        )

    if df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "total_r": 0.0,
        }

    n_trades = len(df)
    wins = (df["r_multipler"] > 0).sum()
    win_rate = wins / n_trades
    total_r = df["r_multipler"].sum()
    expectancy_r = df["r_multipler"].mean()

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "expectancy_r": expectancy_r,
        "total_r": total_r,
    }

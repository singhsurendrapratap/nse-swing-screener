"""
NSE Swing / Positional Screener -- Streamlit app

Deployment: replace the ENTIRE contents of app.py and engine.py in your repo.
Run with: streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
import pandas as pd
from datetime import date
from engine import (
    DEFAULT_UNIVERSE,
    MIDSMALLCAP_UNIVERSE,
    DEFAULT_PARAMS,
    ENGINE_VERSION,
    run_backtest,
    run_walk_forward_backtest,
    screen_today,
    evaluate_positions,
)

APP_VERSION = "app-2026-08-31-d-fixedsplit"

POSITIONS_FILE = "positions.csv"
POSITIONS_COLS = ["Symbol", "Entry Date", "Entry Price", "Qty", "Stop", "Target"]


def load_positions() -> pd.DataFrame:
    if os.path.exists(POSITIONS_FILE):
        return pd.read_csv(POSITIONS_FILE)
    return pd.DataFrame(columns=POSITIONS_COLS)


def save_positions(df: pd.DataFrame):
    df.to_csv(POSITIONS_FILE, index=False)


st.set_page_config(page_title="NSE A+ Swing Screener", layout="wide")
st.title("📈 NSE A+ Swing / Positional Screener")
st.caption(
    "Selective breakout system: market regime → trend → relative strength → "
    "breakout quality → volume → contraction → anti-chase checks. The app ranks "
    "setups and limits the number of trades instead of buying every qualifying breakout."
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Strategy Settings")

    capital = st.number_input("Total capital (Rs)", min_value=10000, value=500000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100

    st.divider()
    st.subheader("A+ selection")
    score_threshold = st.slider(
        "Minimum quality score (0–100)", 50, 95,
        int(DEFAULT_PARAMS.get("score_threshold", 83)), 1,
        help="Default updated to 83 based on optimization research."
    )
    max_trades = st.slider(
        "Maximum new trades per day", 1, 5,
        int(DEFAULT_PARAMS.get("max_trades_per_day", 3)), 1,
        help="The screener ranks candidates and only returns the best N."
    )

    st.divider()
    st.subheader("Quality gates")
    rsi_low, rsi_high = st.slider(
        "RSI sanity band", 35, 80,
        (DEFAULT_PARAMS.get("rsi_low", 45), DEFAULT_PARAMS.get("rsi_high", 70))
    )
    min_earnings_growth = st.slider(
        "Minimum quarterly earnings growth YoY (%)", 0, 50,
        int(DEFAULT_PARAMS.get("min_earnings_growth", 0.10) * 100), 5,
        help="LIVE screener only."
    ) / 100

    st.divider()
    st.subheader("Exit")
    atr_stop = st.slider(
        "Initial stop (x ATR)", 0.75, 3.0,
        float(DEFAULT_PARAMS.get("atr_stop_mult", 1.5)), 0.1
    )
    breakeven_r = st.slider(
        "Move stop to breakeven at (+R)", 0.5, 2.5,
        float(DEFAULT_PARAMS.get("breakeven_r", 1.0)), 0.1
    )
    partial_r = st.slider(
        "Sell 50% at (+R)", 1.0, 5.0,
        float(DEFAULT_PARAMS.get("partial_r", 2.0)), 0.1
    )
    runner_trail_mult = st.slider(
        "Runner ATR trail", 1.0, 4.0,
        float(DEFAULT_PARAMS.get("runner_trail_mult", 2.0)), 0.1
    )
    hold_days = st.slider(
        "Maximum hold (trading days)", 5, 60,
        int(DEFAULT_PARAMS.get("hold_days", 40))
    )
    friction_pct = st.slider(
        "Friction: brokerage + STT + slippage (%)", 0.0, 0.5, 0.15, 0.05
    ) / 100

    if partial_r <= breakeven_r:
        st.error("Partial target should be above the breakeven trigger.")

    st.divider()
    universe_choice = st.radio(
        "Universe",
        ["Large-cap (Nifty 50-ish)", "Mid/Small-cap (higher momentum, higher risk)"],
        index=0,
    )
    if universe_choice.startswith("Mid"):
        st.warning("Mid/small-cap universe has survivorship bias because today's survivors are used historically.")
        default_universe = MIDSMALLCAP_UNIVERSE
    else:
        default_universe = DEFAULT_UNIVERSE
    universe = st.multiselect("Tickers", default_universe, default=default_universe)
    backtest_years = st.slider("Backtest lookback (years)", 1, 5, 3)

params = dict(DEFAULT_PARAMS)
params.update(
    score_threshold=score_threshold,
    max_trades_per_day=max_trades,
    rsi_low=rsi_low,
    rsi_high=rsi_high,
    min_earnings_growth=min_earnings_growth,
    atr_stop_mult=atr_stop,
    breakeven_r=breakeven_r,
    partial_r=partial_r,
    runner_trail_mult=runner_trail_mult,
    hold_days=hold_days,
    friction_pct=friction_pct,
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 A+ Watchlist", "📊 Research Backtest", "📌 My Positions"])

# -----------------------------------------------------------------------------
# TODAY
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Today's highest-quality long setups")
    st.write(
        f"The screener first requires a bullish Nifty regime, then applies the "
        f"technical quality engine. Only candidates scoring **{score_threshold}/100+** "
        f"and passing the live earnings gate are returned. The list is capped at "
        f"**{max_trades} trade(s)** and ranked by quality."
    )

    if st.button("🔄 Find today's A+ setups", type="primary"):
        if not universe:
            st.error("Select at least one ticker.")
        elif partial_r <= breakeven_r:
            st.error("Fix the exit settings first: partial target must be above breakeven.")
        else:
            with st.spinner("Fetching market data, ranking setups, and checking earnings..."):
                watchlist, market_bullish = screen_today(universe, capital, risk_pct, params)

            if not market_bullish:
                st.warning(
                    "No long trades: Nifty is not in the required bullish regime "
                    "(Close > 20EMA > 50SMA)."
                )
            elif watchlist.empty:
                st.info("No A+ setup passed all filters today. Do not force a trade.")
            else:
                st.success(f"Found {len(watchlist)} A+ candidate(s).")
                st.dataframe(watchlist, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download today's watchlist",
                    watchlist.to_csv(index=False),
                    file_name="todays_Aplus_watchlist.csv",
                    mime="text/csv",
                )

# -----------------------------------------------------------------------------
# BACKTEST
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Research the exact selection + exit rules")
    st.caption(
        f"Quality score ≥ {score_threshold}/100 | Max {max_trades} trades/day | "
        f"Stop {atr_stop}×ATR | BE {breakeven_r}R | Partial {partial_r}R | "
        f"Runner {runner_trail_mult}×ATR | {hold_days}-day max hold"
    )

    if st.button("▶️ Run research backtest", type="primary"):
        if not universe:
            st.error("Select at least one ticker.")
        elif partial_r <= breakeven_r:
            st.error("Fix the exit settings first.")
        else:
            with st.spinner(f"Backtesting {len(universe)} stocks over {backtest_years} years..."):
                trades_df, summary = run_backtest(universe, backtest_years, params)

            if not summary:
                st.info("No trades passed the filters in this period.")
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Trades", summary.get("total_trades", 0))
                c2.metric("Win rate", f"{summary.get('win_rate', 0.0):.1f}%")
                c3.metric("Expectancy", f"{summary.get('expectancy_r', 0.0):.2f}R")

                pf = summary.get("profit_factor", 0.0)
                c4.metric("Profit factor", f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf))

                c5.metric("Total R", f"{summary.get('total_r', 0.0):.2f}R")

                c6, c7, c8 = st.columns(3)
                c6.metric("Avg win", f"{summary.get('avg_win_r', 0.0):.2f}R")
                c7.metric("Avg loss", f"{summary.get('avg_loss_r', 0.0):.2f}R")
                c8.metric("Max losing streak", summary.get("max_loss_streak", 0))

                if summary.get("total_trades", 0) < 50:
                    st.warning("Small sample size.")
                elif summary.get("expectancy_r", 0.0) <= 0:
                    st.error("Negative expectancy.")
                else:
                    st.success("Positive expectancy in this test window.")

                st.divider()
                st.subheader("🔬 Winner vs loser diagnostics")
                if not trades_df.empty:
                    numeric_cols = [
                        "setup_score", "breakout_atr", "volume_ratio", "rs20", "rs63", "rs126",
                        "rsi14", "extension_atr", "gap_pct", "close_location", "body_atr",
                        "upper_wick_pct", "atr_contraction",
                    ]
                    available = [c for c in numeric_cols if c in trades_df.columns]
                    diag = trades_df.groupby("outcome")[available].mean().T
                    st.dataframe(diag.round(3), use_container_width=True)

                    st.subheader("Trade-level research table")
                    st.dataframe(
                        trades_df.sort_values("entry_date", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "⬇️ Download full diagnostic backtest CSV",
                        trades_df.to_csv(index=False),
                        file_name="diagnostic_backtest.csv",
                        mime="text/csv",
                    )

    st.divider()
    st.subheader("🧪 Walk-forward validation")
    split_mode = st.radio(
        "Split method", ["Fixed calendar date (comparable across runs)", "% of trades (floats with window length)"],
        index=0,
    )
    if split_mode.startswith("Fixed"):
        split_date_input = st.date_input(
            "Out-of-sample starts on", value=date(2025, 1, 1),
        )
        out_sample_pct = None
    else:
        out_sample_pct = st.slider("Out-of-sample size (most recent %, held out)", 20, 50, 35, 5)
        split_date_input = None

    if st.button("🧪 Run walk-forward validation"):
        if not universe:
            st.error("Select at least one ticker.")
        elif partial_r <= breakeven_r:
            st.error("Fix the exit settings first.")
        else:
            with st.spinner(f"Generating trades across {backtest_years} years, then splitting..."):
                if split_date_input:
                    wf = run_walk_forward_backtest(universe, backtest_years, params,
                                                    split_date=split_date_input.isoformat())
                else:
                    wf = run_walk_forward_backtest(universe, backtest_years, params,
                                                    out_sample_frac=out_sample_pct / 100)

            if not wf.get("in_sample") or not wf.get("out_sample"):
                st.info("Not enough trades in this window to split meaningfully -- widen years or universe.")
            else:
                ins, oos = wf["in_sample"], wf["out_sample"]
                split_dt_str = str(wf['split_date'].date()) if wf.get('split_date') and hasattr(wf['split_date'], 'date') else str(wf.get('split_date', 'N/A'))
                st.caption(f"Split point: **{split_dt_str}** -- everything before is in-sample, everything from that date onward is out-of-sample.")

                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("**In-sample (earlier, tuned on)**")
                    st.metric("Trades", ins.get("total_trades", 0))
                    st.metric("Win rate", f"{ins.get('win_rate', 0):.1f}%")
                    st.metric("Expectancy", f"{ins.get('expectancy_r', 0):.2f}R")
                    pf_in = ins.get("profit_factor", 0)
                    st.metric("Profit factor", f"{pf_in:.2f}" if isinstance(pf_in, (int, float)) else str(pf_in))
                with col_out:
                    st.markdown("**Out-of-sample (later, untouched)**")
                    st.metric("Trades", oos.get("total_trades", 0))
                    st.metric("Win rate", f"{oos.get('win_rate', 0):.1f}%")
                    st.metric("Expectancy", f"{oos.get('expectancy_r', 0):.2f}R")
                    pf_out = oos.get("profit_factor", 0)
                    st.metric("Profit factor", f"{pf_out:.2f}" if isinstance(pf_out, (int, float)) else str(pf_out))

                exp_in = ins.get("expectancy_r", 0)
                exp_out = oos.get("expectancy_r", 0)
                if oos.get("total_trades", 0) < 15:
                    st.warning(
                        f"Only {oos.get('total_trades', 0)} out-of-sample trades -- too few to draw a real conclusion."
                    )
                elif exp_out >= exp_in - 0.05:
                    st.success("Out-of-sample expectancy holds up close to or above in-sample.")
                else:
                    st.error(f"Out-of-sample expectancy ({exp_out:.2f}R) is worse than in-sample ({exp_in:.2f}R).")

                # Safe tagging for export
                export_df = wf["trades_df"].copy()
                if "entry_date" in export_df.columns and wf.get("split_date"):
                    export_df["sample"] = np.where(
                        pd.to_datetime(export_df["entry_date"]) < pd.to_datetime(wf["split_date"]),
                        "in_sample",
                        "out_sample"
                    )

                st.download_button(
                    "⬇️ Download walk-forward trade data (tagged in/out-of-sample)",
                    export_df.to_csv(index=False),
                    file_name="walk_forward_backtest.csv",
                    mime="text/csv",
                )

# -----------------------------------------------------------------------------
# POSITIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Track open positions")

    positions = load_positions()

    with st.form("add_position", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol (e.g. RELIANCE)").strip().upper()
            entry_date = st.date_input("Entry date", value=date.today())
        with c2:
            entry_price = st.number_input("Entry price (Rs)", min_value=0.0, step=0.5)
            qty = st.number_input("Qty", min_value=1, step=1)
        with c3:
            stop = st.number_input("Initial stop (Rs)", min_value=0.0, step=0.5)
            target = st.number_input("Partial target (Rs)", min_value=0.0, step=0.5)
        submitted = st.form_submit_button("➕ Add position")
        if submitted and symbol and entry_price > 0:
            new_row = pd.DataFrame([{
                "Symbol": symbol,
                "Entry Date": entry_date.isoformat(),
                "Entry Price": entry_price,
                "Qty": qty,
                "Stop": stop,
                "Target": target,
            }])
            positions = pd.concat([positions, new_row], ignore_index=True)
            save_positions(positions)
            st.success(f"Added {symbol}.")
            st.rerun()

    st.divider()
    if positions.empty:
        st.info("No open positions tracked yet.")
    else:
        if st.button("🔄 Refresh position signals", type="primary"):
            with st.spinner("Fetching latest prices..."):
                status_df = evaluate_positions(positions, params)
            st.session_state["position_status"] = status_df

        if "position_status" in st.session_state:
            status_df = st.session_state["position_status"]

            def _highlight(row):
                signal = str(row.get("Signal", ""))
                if "SELL" in signal:
                    return ["background-color: #ffdddd"] * len(row)
                if "CONSIDER" in signal:
                    return ["background-color: #fff6cc"] * len(row)
                return [""] * len(row)

            st.dataframe(
                status_df.style.apply(_highlight, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Click Refresh position signals.")

        st.divider()
        to_remove = st.selectbox("Closed position to remove", options=[""] + positions["Symbol"].tolist())
        if st.button("🗑️ Remove") and to_remove:
            positions = positions[positions["Symbol"] != to_remove]
            save_positions(positions)
            st.session_state.pop("position_status", None)
            st.success(f"Removed {to_remove}.")
            st.rerun()

st.divider()
st.caption(f"Build check: `{APP_VERSION}` (app) / `{ENGINE_VERSION}` (engine)")

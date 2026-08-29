"""
NSE Swing Screener -- Web App
"""

import os
import streamlit as st
import pandas as pd
from datetime import date
from engine import (
    DEFAULT_UNIVERSE, MIDSMALLCAP_UNIVERSE, DEFAULT_PARAMS,
    run_backtest, screen_today, evaluate_positions,
)

POSITIONS_FILE = "positions.csv"
POSITIONS_COLS = ["Symbol", "Entry Date", "Entry Price", "Qty", "Stop", "Target"]


def load_positions() -> pd.DataFrame:
    if os.path.exists(POSITIONS_FILE):
        return pd.read_csv(POSITIONS_FILE)
    return pd.DataFrame(columns=POSITIONS_COLS)


def save_positions(df: pd.DataFrame):
    df.to_csv(POSITIONS_FILE, index=False)


st.set_page_config(page_title="NSE Swing Screener", layout="wide")
st.title("📈 NSE Swing / Positional Screener")
st.caption(
    "Trend + breakout, filtered by a weighted setup score, with a layered exit "
    "(breakeven-early, partial profit, trail the rest). Not financial advice -- "
    "a transparent tool so you can see and test the logic yourself."
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    capital = st.number_input("Total capital (Rs)", min_value=10000, value=500000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100

    st.divider()
    st.subheader("Setup score")
    score_threshold = st.slider(
        "Minimum score required (out of 10)", 2, 10, DEFAULT_PARAMS["score_threshold"], 2,
        help="Trend alignment, relative strength, volatility contraction, and volume "
             "expansion each contribute 2 points. Market regime, breakout, and RSI "
             "band are separate mandatory gates -- always required regardless of score.",
    )
    with st.expander("Score component thresholds"):
        rs_min_outperformance = st.slider(
            "Min relative-strength margin vs Nifty (%)", 0, 30,
            int(DEFAULT_PARAMS["rs_min_outperformance"] * 100), 1,
        ) / 100
        contraction_ratio_threshold = st.slider(
            "Contraction strictness (ATR5/ATR20 must be below)", 0.4, 1.0,
            DEFAULT_PARAMS["contraction_ratio_threshold"], 0.05,
        )
        volume_mult = st.slider(
            "Breakout volume multiple", 1.0, 3.0, DEFAULT_PARAMS["volume_mult"], 0.1,
        )
        rs_lookback = st.slider(
            "Relative strength lookback (days)", 20, 120, DEFAULT_PARAMS["rs_lookback"], 1,
        )

    st.divider()
    st.subheader("Exit (layered)")
    with st.expander("Exit thresholds"):
        atr_stop = st.slider("Initial stop-loss (x ATR)", 0.5, 3.0, DEFAULT_PARAMS["atr_stop_mult"], 0.1)
        breakeven_mult = st.slider("Move to breakeven at (x ATR)", 0.5, 3.0, DEFAULT_PARAMS["breakeven_mult"], 0.1)
        partial_target_mult = st.slider("Sell 50% at (x ATR)", 1.0, 5.0, DEFAULT_PARAMS["partial_target_mult"], 0.1)
        runner_trail_mult = st.slider("Trail the rest (x ATR)", 1.0, 4.0, DEFAULT_PARAMS["runner_trail_mult"], 0.1)
        hold_days = st.slider("Max hold (trading days)", 5, 60, DEFAULT_PARAMS["hold_days"])
        friction_pct = st.slider("Friction: brokerage+STT+slippage (%)", 0.0, 0.5, 0.15, 0.05) / 100
        rsi_low, rsi_high = st.slider("RSI sanity band", 0, 100, (DEFAULT_PARAMS["rsi_low"], DEFAULT_PARAMS["rsi_high"]))

    st.divider()
    universe_choice = st.radio(
        "Universe",
        ["Large-cap (Nifty 50-ish)", "Mid/Small-cap (higher momentum, higher risk)"],
        index=0,
    )
    if universe_choice.startswith("Mid"):
        st.caption(
            "⚠️ Representative sample of TODAY's liquid mid/smallcaps, tested against "
            "PAST years -- delisted/crashed-out stocks aren't included."
        )
        default_universe = MIDSMALLCAP_UNIVERSE
    else:
        default_universe = DEFAULT_UNIVERSE
    universe = st.multiselect("Tickers", default_universe, default=default_universe)
    backtest_years = st.slider("Backtest lookback (years)", 1, 5, 3)

params = dict(
    volume_mult=volume_mult, rsi_low=rsi_low, rsi_high=rsi_high,
    atr_stop_mult=atr_stop, breakeven_mult=breakeven_mult,
    partial_target_mult=partial_target_mult, runner_trail_mult=runner_trail_mult,
    hold_days=hold_days, friction_pct=friction_pct,
    score_threshold=score_threshold, rs_lookback=rs_lookback,
    rs_min_outperformance=rs_min_outperformance,
    contraction_ratio_threshold=contraction_ratio_threshold,
)

tab1, tab2, tab3 = st.tabs(["🎯 Today's Watchlist", "📊 Backtest", "📌 My Positions"])

# ---------------------------------------------------------------------------
# TAB 1 -- TODAY'S PICKS
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Today's qualifying setups")
    st.write(
        f"Screens the latest close for every stock in your universe, scores each "
        f"against the 4 weighted factors, and shows only those scoring "
        f"**{score_threshold}/10 or higher** (plus mandatory gates)."
    )
    if st.button("🔄 Screen today's market", type="primary"):
        with st.spinner("Fetching latest NSE data and screening..."):
            watchlist, market_bullish = screen_today(universe, capital, risk_pct, params)

        if not market_bullish:
            st.warning(
                "**Market regime filter: OFF.** Nifty 50 isn't in a confirmed uptrend "
                "(needs Close > 20-EMA > 50-SMA) -- no long setups suggested today."
            )
        elif watchlist.empty:
            st.info(
                "No setups scored high enough today. That's a normal, valid outcome."
            )
        else:
            st.success(f"{len(watchlist)} setup(s) found.")
            st.dataframe(watchlist, use_container_width=True, hide_index=True)
            st.download_button(
                "Download as CSV", watchlist.to_csv(index=False),
                file_name="todays_watchlist.csv", mime="text/csv",
            )

# ---------------------------------------------------------------------------
# TAB 2 -- BACKTEST
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Historical performance of this exact strategy")
    st.caption(
        f"Weighted score, needs **{score_threshold}/10** | Layered exit "
        f"| Universe: {universe_choice}"
    )
    if st.button("▶️ Run backtest"):
        with st.spinner(f"Backtesting {len(universe)} stocks over {backtest_years} years..."):
            trades_df, summary = run_backtest(universe, backtest_years, params)

        if not summary:
            st.info("No trades generated in this period with these settings.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total trades", summary["total_trades"])
            c2.metric("Win rate", f"{summary['win_rate']:.1f}%")
            c3.metric("Expectancy / trade", f"{summary['expectancy_r']:.2f}R")
            c4.metric("Avg days held", f"{summary['avg_days_held']:.1f}")
            st.markdown(f"Avg win: **{summary['avg_win_r']:.2f}R** &nbsp;|&nbsp; "
                        f"Avg loss: **{summary['avg_loss_r']:.2f}R**")
            st.divider()
            st.dataframe(trades_df.sort_values("entry_date", ascending=False),
                        use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 -- MY POSITIONS
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Track positions you've already bought")
    positions = load_positions()

    with st.form("add_position", clear_on_submit=True):
        st.markdown("**Add a position**")
        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol (e.g. RELIANCE)").strip().upper()
            entry_date = st.date_input("Entry date", value=date.today())
        with c2:
            entry_price = st.number_input("Entry price (Rs)", min_value=0.0, step=0.5)
            qty = st.number_input("Qty", min_value=1, step=1)
        with c3:
            stop = st.number_input("Stop-loss (Rs)", min_value=0.0, step=0.5)
            target = st.number_input("Partial target (Rs)", min_value=0.0, step=0.5)
        submitted = st.form_submit_button("➕ Add position")
        if submitted and symbol and entry_price > 0:
            new_row = pd.DataFrame([{
                "Symbol": symbol, "Entry Date": entry_date.isoformat(),
                "Entry Price": entry_price, "Qty": qty, "Stop": stop, "Target": target,
            }])
            positions = pd.concat([positions, new_row], ignore_index=True)
            save_positions(positions)
            st.success(f"Added {symbol}.")
            st.rerun()

    st.divider()

    if positions.empty:
        st.info("No open positions tracked yet -- add one above.")
    else:
        if st.button("🔄 Refresh signals for my positions", type="primary"):
            with st.spinner("Fetching latest prices..."):
                status_df = evaluate_positions(positions, params)
            st.session_state["position_status"] = status_df

        if "position_status" in st.session_state:
            status_df = st.session_state["position_status"]

            def _highlight(row):
                if "SELL" in str(row["Signal"]):
                    return ["background-color: #ffdddd"] * len(row)
                if "CONSIDER" in str(row["Signal"]):
                    return ["background-color: #fff6cc"] * len(row)
                return [""] * len(row)

            st.dataframe(status_df.style.apply(_highlight, axis=1),
                        use_container_width=True, hide_index=True)
        else:
            st.info("Click 'Refresh signals' to check your positions against live prices.")

        st.divider()
        st.markdown("**Remove a closed position**")
        to_remove = st.selectbox("Symbol", options=[""] + positions["Symbol"].tolist())
        if st.button("🗑️ Remove") and to_remove:
            positions = positions[positions["Symbol"] != to_remove]
            save_positions(positions)
            st.session_state.pop("position_status", None)
            st.success(f"Removed {to_remove}.")
            st.rerun()

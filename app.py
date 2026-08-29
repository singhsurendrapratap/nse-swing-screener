"""
NSE Swing Screener -- Web App
==============================
Run with:  streamlit run app.py

Deploy for free (so you get a real URL, open on phone/laptop anytime) at
https://streamlit.io/cloud -- push these files to a GitHub repo, connect it,
done. No server management needed.
"""

import os
import streamlit as st
import pandas as pd
from datetime import date
from engine import DEFAULT_UNIVERSE, run_backtest, screen_today, evaluate_positions

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
    "Trend + breakout + volume strategy, with a Nifty-50 market-regime filter, "
    "realistic next-day-open execution, friction costs, and risk-based position sizing. "
    "Not financial advice -- a transparent tool so you can see and test the logic yourself."
)

# ---------------------------------------------------------------------------
# SIDEBAR -- all the knobs, nothing hidden
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    capital = st.number_input("Total capital (Rs)", min_value=10000, value=500000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100
    st.divider()
    st.subheader("Strategy parameters")
    volume_mult = st.slider("Breakout volume multiple", 1.0, 3.0, 1.5, 0.1)
    rsi_low, rsi_high = st.slider("RSI band", 0, 100, (45, 70))
    atr_stop = st.slider("Stop-loss (x ATR)", 0.5, 3.0, 1.5, 0.1)
    use_trailing_stop = st.checkbox(
        "Use trailing stop instead of fixed target",
        value=False,
        help="Instead of exiting at a fixed 2:1/3:1 target, let the stop ratchet up "
             "behind the price and only exit when it's hit. Lets winners run further, "
             "at the cost of a less predictable exit price.",
    )
    atr_target = st.slider("Target (x ATR)", 1.0, 6.0, 3.0, 0.1, disabled=use_trailing_stop)
    hold_days = st.slider("Max hold (trading days)", 5, 40, 15)
    friction_pct = st.slider("Friction: brokerage+STT+slippage (%)", 0.0, 0.5, 0.15, 0.05) / 100
    st.divider()
    universe = st.multiselect("Universe (NSE tickers)", DEFAULT_UNIVERSE, default=DEFAULT_UNIVERSE)
    backtest_years = st.slider("Backtest lookback (years)", 1, 5, 3)

params = dict(
    volume_mult=volume_mult, rsi_low=rsi_low, rsi_high=rsi_high,
    atr_stop_mult=atr_stop, atr_target_mult=atr_target,
    hold_days=hold_days, friction_pct=friction_pct,
    use_trailing_stop=use_trailing_stop,
)

tab1, tab2, tab3 = st.tabs(["🎯 Today's Watchlist", "📊 Backtest", "📌 My Positions"])

# ---------------------------------------------------------------------------
# TAB 1 -- today's picks, with buy/sell info
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Today's qualifying setups")
    st.write(
        "Click below to pull the **latest close** for every stock in your universe and "
        "screen it against the strategy right now. Kept intentionally short (quality over quantity)."
    )
    if st.button("🔄 Screen today's market", type="primary"):
        with st.spinner("Fetching latest NSE data and screening..."):
            watchlist, market_bullish = screen_today(universe, capital, risk_pct, params)

        if not market_bullish:
            st.warning(
                "**Market regime filter: OFF.** Nifty 50 is currently below its 50-day average, "
                "so no long setups are being suggested today -- this strategy sits out downtrends "
                "on purpose rather than fighting the broader market."
            )
        elif watchlist.empty:
            st.info(
                "No qualifying setups today. That's a normal, valid outcome for a selective "
                "strategy -- don't force a trade because the screen came up empty."
            )
        else:
            st.success(f"{len(watchlist)} setup(s) found. 'Buy Near' = suggested entry; "
                       "'Stop-Loss' and 'Target' are your planned exits.")
            st.dataframe(watchlist, use_container_width=True, hide_index=True)
            st.download_button(
                "Download as CSV", watchlist.to_csv(index=False),
                file_name="todays_watchlist.csv", mime="text/csv",
            )
            st.caption(
                "Qty is sized so that if the stop-loss is hit, you lose only your chosen "
                "risk % of total capital on that trade -- not a full-conviction bet."
            )

# ---------------------------------------------------------------------------
# TAB 2 -- backtest, so claims are checkable
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Historical performance of this exact strategy")
    if params["use_trailing_stop"]:
        st.success("🟢 Mode: **TRAILING STOP** (no fixed target — stop ratchets up behind price)")
    else:
        st.info("🔵 Mode: **FIXED TARGET** (exits at a fixed 3x-ATR target)")
    st.write(
        "Runs the same rules over the past N years so you can see the real win rate and "
        "expectancy before trusting today's picks."
    )
    if st.button("▶️ Run backtest"):
        with st.spinner(f"Backtesting {len(universe)} stocks over {backtest_years} years..."):
            trades_df, summary = run_backtest(universe, backtest_years, params)

        if not summary:
            st.info("No trades generated in this period with these settings.")
        else:
            st.caption(f"**These results were generated using: {summary['mode']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total trades", summary["total_trades"])
            c2.metric("Win rate", f"{summary['win_rate']:.1f}%")
            c3.metric("Expectancy / trade", f"{summary['expectancy_r']:.2f}R")
            c4.metric("Avg days held", f"{summary['avg_days_held']:.1f}")
            st.caption(
                "R = multiples of amount risked per trade. Expectancy matters more than win "
                "rate: a 45% win rate with +0.6R expectancy beats a 65% win rate with -0.1R."
            )
            st.markdown(f"Avg win: **{summary['avg_win_r']:.2f}R** &nbsp;|&nbsp; "
                        f"Avg loss: **{summary['avg_loss_r']:.2f}R**")
            st.divider()
            st.dataframe(trades_df.sort_values("entry_date", ascending=False),
                        use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 -- track your open positions, get live sell/exit signals
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Track positions you've already bought")
    st.write(
        "Add a stock you're holding once. Every time you open this app, it re-checks the "
        "latest price against your stop/target and tells you HOLD or SELL -- and suggests a "
        "trailing stop that only ever moves up, so you can lock in gains without babysitting it."
    )

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
            target = st.number_input("Target (Rs)", min_value=0.0, step=0.5)
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
            st.caption(
                "Red = sell signal triggered. Yellow = trend weakening, worth a look. "
                "'Suggested Trailing Stop' only ever moves up from your original stop -- "
                "update your broker's stop-loss order to match it if you want to lock in gains."
            )
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

st.divider()
st.caption(
    "⚠️ Educational tool, not investment advice. Past performance does not guarantee future "
    "results. Always size positions to a risk % you can afford to lose repeatedly. "
    "Note: on free cloud hosting, the positions file may reset if the app restarts/redeploys -- "
    "download a backup periodically if that matters to you."
)

"""
NSE Swing / Positional Screener -- Streamlit app

Deployment: this repo needs exactly two files, named exactly app.py and
engine.py -- replace their entire contents with these two files (this one,
and engine.py) each time. Check the version string in the sidebar footer
after any redeploy to confirm it actually took.

Run with: streamlit run app.py
"""

import os
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

APP_VERSION = "app-2026-08-31-c-calendarsplit"

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
        int(DEFAULT_PARAMS.get("score_threshold", 78)), 1,
        help="Start around 78. Do not optimize this to the historical sample. "
             "Use walk-forward/out-of-sample testing before changing it."
    )
    max_trades = st.slider(
        "Maximum new trades per day", 1, 5,
        int(DEFAULT_PARAMS.get("max_trades_per_day", 3)), 1,
        help="The screener ranks candidates and only returns the best N. "
             "This is deliberately selective."
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
        help="LIVE screener only. It is not used in the historical backtest because "
             "this workflow does not have point-in-time historical earnings data."
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
        int(DEFAULT_PARAMS.get("hold_days", 20))
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
    st.info(
        "Best practice: run the daily screen after the market close if you want a "
        "clean completed daily candle. If you trade intraday, treat the result as a "
        "research signal, not a confirmed end-of-day breakout."
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
                st.caption(
                    "Position size is based on the initial stop and selected risk budget. "
                    "The backtest assumes next-session open entry, so avoid treating the "
                    "screening close as a guaranteed fill price."
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
    st.warning(
        "The historical backtest is TECHNICAL ONLY. The live earnings-growth gate is "
        "not applied historically because using today's earnings data for old trades "
        "would create lookahead bias."
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
                    st.warning("Small sample. Do not trust the win rate/expectancy yet.")
                elif summary.get("expectancy_r", 0.0) <= 0:
                    st.error(
                        "Negative expectancy. Do not use the system live yet. "
                        "Use the diagnostics below to improve selection, then validate out-of-sample."
                    )
                else:
                    st.success("Positive expectancy in this test window — still validate out-of-sample before live use.")

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

                    st.subheader("Performance by quality score")
                    score_bins = pd.cut(
                        trades_df["setup_score"],
                        bins=[0, 69, 77, 84, 89, 94, 100],
                        labels=["<70", "70–77", "78–84", "85–89", "90–94", "95–100"],
                        include_lowest=True,
                    )
                    by_score = trades_df.assign(score_band=score_bins).groupby("score_band", observed=False).agg(
                        Trades=("r_multiple", "size"),
                        WinRate=("outcome", lambda x: (x == "win").mean() * 100),
                        ExpectancyR=("r_multiple", "mean"),
                        TotalR=("r_multiple", "sum"),
                    )
                    st.dataframe(by_score.round(3), use_container_width=True)

                    st.subheader("Performance by year")
                    tmp = trades_df.copy()
                    tmp["Year"] = pd.to_datetime(tmp["entry_date"]).dt.year
                    by_year = tmp.groupby("Year").agg(
                        Trades=("r_multiple", "size"),
                        WinRate=("outcome", lambda x: (x == "win").mean() * 100),
                        ExpectancyR=("r_multiple", "mean"),
                        TotalR=("r_multiple", "sum"),
                    )
                    st.dataframe(by_year.round(3), use_container_width=True)

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
    st.write(
        "Everything above was tuned by eye against one fixed historical window -- the "
        "classic way a backtest quietly overfits. This runs the exact same frozen "
        "settings (whatever the sidebar is currently set to) on trades split "
        "chronologically: an EARLIER 'in-sample' period, and a LATER 'out-of-sample' "
        "period that plays no part in how these settings were chosen. If performance "
        "holds up on the untouched later period, that's real evidence, not a lucky fit."
    )
    split_mode = st.radio(
        "Split method", ["Fixed calendar date (comparable across runs)", "% of trades (floats with window length)"],
        index=0,
        help="Fixed date lets you compare a 3-year and 5-year backtest on the SAME "
             "out-of-sample period. % split is quicker but the cutoff moves depending "
             "on how many years you backtest, so results aren't directly comparable.",
    )
    if split_mode.startswith("Fixed"):
        split_date_input = st.date_input(
            "Out-of-sample starts on", value=date(2025, 1, 1),
            help="Everything before this date is in-sample (tuned on). Everything "
                 "from this date onward is out-of-sample (untouched, the real test).",
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
                if split_mode.startswith("Fixed") and split_date_input:
                    wf = run_walk_forward_backtest(
                        universe=universe,
                        years=backtest_years,
                        params=params,
                        split_date=str(split_date_input)
                    )
                else:
                    wf = run_walk_forward_backtest(
                        universe=universe,
                        years=backtest_years,
                        params=params,
                        out_sample_frac=float(out_sample_pct / 100)
                    )

            if wf:
                in_s = wf["in_sample"]
                out_s = wf["out_sample"]

                col_in, col_out = st.columns(2)
                with col_in:
                    st.markdown("### 🟢 In-Sample (Training)")
                    st.write(f"**Split Date:** Before {wf['split_date']}")
                    st.metric("Trades", in_s.get("total_trades", 0))
                    st.metric("Win Rate", f"{in_s.get('win_rate', 0.0):.1f}%")
                    st.metric("Expectancy", f"{in_s.get('expectancy_r', 0.0):.2f}R")
                    st.metric("Total R", f"{in_s.get('total_r', 0.0):.2f}R")

                with col_out:
                    st.markdown("### 🔵 Out-Of-Sample (Validation)")
                    st.write(f"**Split Date:** On/After {wf['split_date']}")
                    st.metric("Trades", out_s.get("total_trades", 0))
                    st.metric("Win Rate", f"{out_s.get('win_rate', 0.0):.1f}%")
                    st.metric("Expectancy", f"{out_s.get('expectancy_r', 0.0):.2f}R")
                    st.metric("Total R", f"{out_s.get('total_r', 0.0):.2f}R")

# -----------------------------------------------------------------------------
# MY POSITIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Track active positions")
    pos_df = load_positions()

    with st.form("add_position_form"):
        st.write("Add New Position")
        col1, col2, col3 = st.columns(3)
        sym = col1.text_input("Symbol (e.g. RELIANCE.NS)")
        entry_d = col2.date_input("Entry Date", value=date.today())
        entry_p = col3.number_input("Entry Price", min_value=0.0, value=0.0)
        
        col4, col5, col6 = st.columns(3)
        qty = col4.number_input("Quantity", min_value=1, value=1)
        stop = col5.number_input("Stop Price", min_value=0.0, value=0.0)
        target = col6.number_input("Target Price", min_value=0.0, value=0.0)
        
        submitted = st.form_submit_button("Add Position")
        if submitted and sym:
            new_row = pd.DataFrame([{
                "Symbol": sym.upper().strip(),
                "Entry Date": str(entry_d),
                "Entry Price": entry_p,
                "Qty": qty,
                "Stop": stop,
                "Target": target
            }])
            pos_df = pd.concat([pos_df, new_row], ignore_index=True)
            save_positions(pos_df)
            st.success(f"Added {sym.upper()}")
            st.rerun()

    if pos_df.empty:
        st.info("No active positions tracked.")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        if st.button("Evaluate Positions"):
            eval_results = evaluate_positions(pos_df)
            st.dataframe(eval_results, use_container_width=True)

        if st.button("Clear All Positions"):
            save_positions(pd.DataFrame(columns=POSITIONS_COLS))
            st.success("Cleared all positions.")
            st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"App Version: {APP_VERSION} | Engine Version: {ENGINE_VERSION}")

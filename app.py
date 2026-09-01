import os
from datetime import date
import pandas as pd
import streamlit as st
import yfinance as yf

from engine import (
    DEFAULT_PARAMS,
    DEFAULT_UNIVERSE,
    ENGINE_VERSION,
    MIDSMALLCAP_UNIVERSE,
    evaluate_positions,
    run_backtest,
    run_optimizer_backtest,
    run_walk_forward_backtest,
    screen_today,
)
from optimizer import run_ai_optimizer
from signal_logger import (
    get_track_record,
    log_todays_signals,
    summary_stats,
    update_open_outcomes,
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


def live_price_lookup(symbol: str):
    yf_sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    try:
        t = yf.Ticker(yf_sym)
        hist = t.history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


st.set_page_config(page_title="NSE A+ Swing Screener", layout="wide")
st.title("📈 NSE A+ Swing / Positional Screener")
st.caption(
    "Selective breakout system: market regime → trend → relative strength → "
    "breakout quality → volume → contraction → anti-chase checks."
)

with st.sidebar:
    st.header("Strategy Controls")
    mode = st.radio("Optimization Mode", ["Manual Control", "AI Auto-Optimizer"])

    capital = st.number_input("Total capital (Rs)", min_value=10000, value=500000, step=10000)
    risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25) / 100

    st.divider()

    if mode == "Manual Control":
        st.subheader("Manual Parameters")
        score_threshold = st.slider("Minimum quality score (0–100)", 50, 95, 78, 1)
        max_trades = st.slider("Maximum new trades per day", 1, 5, 3, 1)
        atr_stop = st.slider("Initial stop (x ATR)", 0.75, 4.0, 1.5, 0.1)
        breakeven_r = st.slider("Move stop to BE (+R)", 0.5, 3.0, 1.0, 0.1)
        partial_r = st.slider("Sell 50% at (+R)", 1.0, 5.0, 2.0, 0.1)
        runner_trail_mult = st.slider("Runner ATR trail", 1.0, 4.0, 2.0, 0.1)

        active_params = dict(DEFAULT_PARAMS)
        active_params.update(
            score_threshold=score_threshold,
            max_trades_per_day=max_trades,
            atr_stop_mult=atr_stop,
            breakeven_r=breakeven_r,
            partial_r=partial_r,
            runner_trail_mult=runner_trail_mult,
        )
    else:
        st.subheader("AI Optimization Settings (Walk-Forward)")
        target_metric = st.selectbox("Optimize For", ["Expectancy R", "Total R", "Win Rate"])
        min_trades = st.slider("Minimum Trades per Fold", 10, 100, 20)
        n_folds = st.slider("Walk-Forward Folds", 3, 8, 5)
        n_trials = st.number_input("Number of AI Search Trials", min_value=10, max_value=500, value=50)

        active_params = dict(DEFAULT_PARAMS)
        if st.button("🚀 Run AI Parameter Optimization"):
            param_space = {
                "atr_stop_mult": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
                "breakeven_r": {"type": "float", "low": 0.5, "high": 3.0, "step": 0.1},
                "partial_r": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
                "score_threshold": {"type": "int", "low": 50, "high": 90},
            }
            with st.spinner("Generating base trade history for AI Optimizer..."):
                trades_df, _ = run_backtest(DEFAULT_UNIVERSE, 3, active_params)

            if not trades_df.empty:
                with st.spinner("Running walk-forward parameter search across folds..."):
                    opt_res = run_ai_optimizer(
                        param_space, trades_df, run_optimizer_backtest,
                        target_metric, min_trades, n_trials, n_folds
                    )
                st.session_state["optimized_params"] = opt_res["best_params"]
                st.success(f"Best OOS {target_metric}: {opt_res['oos_median_score']:.3f}")
                if opt_res["overfit_warning"]:
                    st.warning(f"⚠️ High fold variance (ratio={opt_res['overfit_ratio']:.2f})")
                st.json(opt_res["best_params"])
                st.caption("Out-of-sample Score by Fold:")
                st.line_chart(opt_res["oos_fold_scores"])
            else:
                st.error("No trade history available to run optimizer.")

        if "optimized_params" in st.session_state:
            active_params.update(st.session_state["optimized_params"])

    st.divider()
    st.subheader("Quality Gates & Universe")
    rsi_low, rsi_high = st.slider("RSI sanity band", 35, 80, (45, 70))
    min_earnings_growth = st.slider("Min Earnings Growth YoY (%)", 0, 50, 10, 5) / 100
    hold_days = st.slider("Max hold (days)", 5, 60, 20)
    friction_pct = st.slider("Friction (%)", 0.0, 0.5, 0.15, 0.05) / 100

    active_params.update(
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        min_earnings_growth=min_earnings_growth,
        hold_days=hold_days,
        friction_pct=friction_pct,
    )

    universe_choice = st.radio("Universe", ["Large-cap", "Mid/Small-cap"], index=0)
    default_universe = MIDSMALLCAP_UNIVERSE if universe_choice.startswith("Mid") else DEFAULT_UNIVERSE
    universe = st.multiselect("Tickers", default_universe, default=default_universe)
    backtest_years = st.slider("Backtest lookback (years)", 1, 5, 3)

    st.divider()
    st.subheader("Forward Track Record")
    if st.button("🔄 Refresh Outcomes"):
        update_open_outcomes(live_price_lookup)
        st.success("Outcomes refreshed.")

    stats = summary_stats()
    if stats["n_trades"] > 0:
        c1, c2 = st.columns(2)
        c1.metric("Forward Trades", stats["n_trades"])
        c2.metric("Win Rate", f"{stats['win_rate']:.1%}")
        st.metric("Expectancy (R)", f"{stats['expectancy_r']:.2f}")
    else:
        st.caption("No closed forward trades logged.")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 A+ Watchlist", "📊 Research Backtest", "📌 Active Positions", "📜 Forward Log"])

with tab1:
    st.subheader("Today's Highest Quality Setups")
    if st.button("🔄 Screen Today's Setups", type="primary"):
        if not universe:
            st.error("Select at least one ticker.")
        else:
            with st.spinner("Fetching data and evaluating live setups..."):
                watchlist, market_bullish = screen_today(universe, capital, risk_pct, active_params)

            if not market_bullish:
                st.warning("Market condition non-bullish (Nifty Close < EMA20 / SMA50).")
            elif watchlist.empty:
                st.info("No candidates passed all filters today.")
            else:
                st.success(f"Found {len(watchlist)} candidate(s).")
                st.dataframe(watchlist, use_container_width=True, hide_index=True)
                log_todays_signals(watchlist, active_params)
                st.caption("Signals logged to tracking DB.")

with tab2:
    st.subheader("Backtest Current Parameters")
    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Executing historical simulation..."):
            trades_df, summary = run_backtest(universe, backtest_years, active_params)

        if not summary:
            st.info("No trades found in selected parameters.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", summary.get("total_trades", 0))
            c2.metric("Win Rate", f"{summary.get('win_rate', 0.0):.1f}%")
            c3.metric("Expectancy", f"{summary.get('expectancy_r', 0.0):.2f}R")
            c4.metric("Total R", f"{summary.get('total_r', 0.0):.2f}R")
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧪 Walk-Forward Split Validation")
    split_date_input = st.date_input("Out-of-sample start date", value=date(2025, 1, 1))
    if st.button("🧪 Run Split Validation"):
        wf = run_walk_forward_backtest(
            universe=universe, years=backtest_years, params=active_params, split_date=str(split_date_input)
        )
        if wf:
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("### 🟢 In-Sample")
                st.write(f"Trades: {wf['in_sample'].get('total_trades', 0)}")
                st.write(f"Expectancy: {wf['in_sample'].get('expectancy_r', 0.0):.2f}R")
            with col_out:
                st.markdown("### 🔵 Out-Of-Sample")
                st.write(f"Trades: {wf['out_sample'].get('total_trades', 0)}")
                st.write(f"Expectancy: {wf['out_sample'].get('expectancy_r', 0.0):.2f}R")

with tab3:
    st.subheader("Manage Active Positions")
    pos_df = load_positions()
    with st.form("add_pos_form"):
        col1, col2, col3 = st.columns(3)
        sym = col1.text_input("Symbol")
        entry_d = col2.date_input("Entry Date", value=date.today())
        entry_p = col3.number_input("Entry Price", min_value=0.0, value=0.0)
        col4, col5, col6 = st.columns(3)
        qty = col4.number_input("Qty", min_value=1, value=1)
        stop = col5.number_input("Stop", min_value=0.0, value=0.0)
        target = col6.number_input("Target", min_value=0.0, value=0.0)
        if st.form_submit_button("Add Position") and sym:
            new_pos = pd.DataFrame([{
                "Symbol": sym.upper().strip(), "Entry Date": str(entry_d),
                "Entry Price": entry_p, "Qty": qty, "Stop": stop, "Target": target
            }])
            pos_df = pd.concat([pos_df, new_pos], ignore_index=True)
            save_positions(pos_df)
            st.rerun()

    if not pos_df.empty:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        if st.button("Evaluate Open Positions"):
            eval_res = evaluate_positions(pos_df, active_params)
            st.dataframe(eval_res, use_container_width=True)
        if st.button("Clear Positions"):
            save_positions(pd.DataFrame(columns=POSITIONS_COLS))
            st.rerun()

with tab4:
    st.subheader("Forward Signal Tracking Database")
    st.dataframe(get_track_record(), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(f"App Version: {APP_VERSION} | Engine Version: {ENGINE_VERSION}")

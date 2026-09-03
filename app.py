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
    STABLE_UNIVERSE,
    DYNAMIC_UNIVERSE,
    DEFAULT_PARAMS,
    ENGINE_VERSION,
    run_backtest,
    run_walk_forward_backtest,
    run_auto_optimize,
    select_active_universe,
    get_market_breadth,
    screen_today,
    evaluate_positions,
)

APP_VERSION = "app-2026-09-03-i-speedup"

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
st.info(
    "**Two ways to use this app.** Don't want to think about sliders? Go straight to "
    "**🤖 AI Agent** — it runs the research for you and explains what it found in plain "
    "language. Want to see and control everything yourself? The sidebar on the left has "
    "every setting exposed, and the Research Backtest tab shows the full diagnostics. "
    "Both use the exact same underlying engine — the Agent just automates the process "
    "we'd otherwise do by hand.",
    icon="🧭",
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
        key="score_threshold_key",
        help="Start around 78. Do not optimize this to the historical sample. "
             "Use walk-forward/out-of-sample testing before changing it. "
             "The AI Agent tab can set this for you automatically."
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
        int(DEFAULT_PARAMS.get("hold_days", 20)),
        key="hold_days_key",
        help="The AI Agent tab can set this for you automatically."
    )
    friction_pct = st.slider(
        "Base friction: brokerage + STT + slippage (%)", 0.0, 0.5, 0.15, 0.05
    ) / 100
    gap_slippage_frac = st.slider(
        "Extra slippage on gap-up entries (% of the gap size)", 0, 50,
        int(DEFAULT_PARAMS.get("gap_slippage_frac", 0.15) * 100), 5,
        help="Breakout days often gap up more than a typical day, so a flat friction "
             "rate understates real cost on the entry candle specifically. This adds "
             "extra slippage proportional to how big that day's actual gap was -- a "
             "2% gap with this at 15% adds another 0.3% cost on top of base friction.",
    ) / 100

    use_breadth_gate = st.checkbox(
        "🧪 Require minimum market breadth to trade (untested -- try via Agent/Backtest tabs)",
        value=False,
        help="Skips new entries on days where too few stocks in the universe are above "
             "their own 50-day SMA, even if Nifty itself still looks fine -- catches a "
             "market propped up by a handful of large stocks. This is a NEW, unvalidated "
             "lever: test it via walk-forward before trusting it, same as everything else.",
    )
    min_breadth_pct = None
    if use_breadth_gate:
        min_breadth_pct = st.slider("Minimum breadth required to trade (%)", 10, 70, 40, 5)

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
    gap_slippage_frac=gap_slippage_frac,
    min_breadth_pct=min_breadth_pct,
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab0, tab1, tab2, tab3 = st.tabs(["🤖 AI Agent", "🎯 A+ Watchlist", "📊 Research Backtest", "📌 My Positions"])

# -----------------------------------------------------------------------------
# AI AGENT
# -----------------------------------------------------------------------------
with tab0:
    st.subheader("🤖 Let the agent find your settings")
    st.write(
        "This runs a systematic search behind the scenes and explains the result in "
        "plain language — you don't need to understand what a 'quality score' or "
        "'ATR multiple' means to use it."
    )
    with st.expander("What exactly does it do? (for the curious)"):
        st.markdown(
            "1. Downloads price data for your selected universe **once**.\n"
            "2. Tests a curated set of **54 configurations** (how selective to be, "
            "how long to let winners run, how wide the stop-loss is) against that "
            "data — these three settings showed the biggest real effects during a "
            "long night of manual testing, so the search is deliberately narrow "
            "rather than an unlimited grid, to avoid finding noise dressed up as edge.\n"
            "3. For each configuration, splits results into an **earlier period** "
            "(used to pick the winner) and a **later period** (never looked at "
            "during selection).\n"
            "4. Picks the winner using **only** the earlier period, then reports "
            "how it performed on the later, untouched period — an honest test, "
            "not a number that's already seen the answer.\n"
            "5. Explains the result and lets you apply it with one tap."
        )

    st.markdown("**Choose a mode**")
    agent_mode_label = st.radio(
        "Mode",
        ["🛡️ Defensive / Stable (large-cap, tighter exits, steadier)",
         "⚡ Aggressive / Growth (mid/small-cap, wider exits, bigger swings)"],
        index=0,
        label_visibility="collapsed",
    )
    agent_mode = "stable" if agent_mode_label.startswith("🛡️") else "dynamic"
    agent_universe = STABLE_UNIVERSE if agent_mode == "stable" else DYNAMIC_UNIVERSE
    if agent_mode == "dynamic":
        st.caption(
            "⚡ Mid/small-cap universe: today's known liquid survivors tested against "
            "past years, so treat results as more optimistic than a true point-in-time "
            "backtest (survivorship bias). Wider stops and bigger profit targets by default."
        )
    else:
        st.caption("🛡️ Large-cap universe. Tighter stops and profit targets by default, aiming for steadier results.")
    st.caption(f"This mode searches within {len(agent_universe)} tickers, independent of the sidebar's universe choice.")

    agent_split_date = st.date_input(
        "Treat data from this date onward as the honest test",
        value=date(2025, 1, 1),
        help="Everything before this date is used to pick the best configuration. "
             "Everything from this date onward is the real, unbiased test of it.",
    )
    agent_quick = st.checkbox(
        "⚡ Quick search (27 combinations instead of 54 -- faster, less thorough)",
        value=False,
        help="Use this if the full search times out or feels too slow on your connection. "
             "Trades search breadth for speed; treat quick-mode results as a rougher signal.",
    )

    if st.button("🤖 Run AI Agent — Find My Best Settings", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _update_progress(done, total, score_threshold_try, hold_days_try, atr_stop_try):
            progress_bar.progress(done / total)
            status_text.caption(
                f"Testing configuration {done}/{total}: "
                f"quality score ≥ {score_threshold_try}, max hold {hold_days_try} days, "
                f"stop {atr_stop_try}x ATR..."
            )

        try:
            with st.spinner("Downloading data once, then testing configurations..."):
                agent_result = run_auto_optimize(
                    agent_universe, backtest_years, params,
                    split_date=agent_split_date.isoformat(),
                    mode=agent_mode,
                    quick=agent_quick,
                    progress_callback=_update_progress,
                )
            progress_bar.empty()
            status_text.empty()
            agent_result["mode"] = agent_mode
            st.session_state["agent_result"] = agent_result
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(
                f"The search hit an error and couldn't finish: `{type(e).__name__}: {e}`\n\n"
                "This is a real failure, not a silent timeout -- if this keeps happening, "
                "try Quick search, a shorter backtest window, or a smaller universe."
            )
            st.session_state.pop("agent_result", None)

    if "agent_result" in st.session_state:
        agent_result = st.session_state["agent_result"]
        best = agent_result.get("best")

        if best is None:
            st.warning(agent_result["recommendation_text"])
        else:
            st.divider()
            st.subheader("Recommendation")

            verdict_text = agent_result["recommendation_text"]
            if best["out_sample_trades"] >= 8 and best["out_sample_expectancy"] > 0:
                st.success(verdict_text)
            elif best["out_sample_trades"] < 8:
                st.warning(verdict_text)
            else:
                st.error(verdict_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Quality score ≥", best["score_threshold"])
            c2.metric("Max hold (days)", best["hold_days"])
            c3.metric("In-sample expectancy", f"{best['in_sample_expectancy']:+.2f}R")
            c4.metric("Out-of-sample expectancy", f"{best['out_sample_expectancy']:+.2f}R")

            st.caption(
                f"In-sample: {best['in_sample_trades']} trades, "
                f"{best['in_sample_win_rate']:.0f}% win rate, PF {best['in_sample_pf']:.2f}  |  "
                f"Out-of-sample: {best['out_sample_trades']} trades, "
                f"{best['out_sample_win_rate']:.0f}% win rate, PF {best['out_sample_pf']:.2f}"
            )

            if st.button("✅ Use this configuration"):
                st.session_state["score_threshold_key"] = best["score_threshold"]
                st.session_state["hold_days_key"] = best["hold_days"]
                st.success("Applied. The sidebar sliders now reflect this configuration.")
                st.rerun()

            with st.expander(f"See all {len(agent_result['leaderboard'])} tested configurations"):
                lb_df = pd.DataFrame(agent_result["leaderboard"])
                st.dataframe(lb_df.round(3), use_container_width=True, hide_index=True)
                st.caption(
                    "Sorted by in-sample expectancy (what the agent used to choose). "
                    "Compare the out-of-sample columns yourself if you want a second opinion."
                )
    else:
        st.info("Click the button above to run the agent. It takes roughly a minute.")



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

    with st.expander("📊 Check market breadth first (optional context)"):
        st.caption(
            "Informational only, not a hard gate -- and the 60%/30% labels below are "
            "illustrative, not walk-forward validated. A market can be 'Nifty above its "
            "50-SMA' while breadth underneath is narrow (a few large stocks propping up "
            "the index) -- this catches that."
        )
        if st.button("Check breadth now"):
            with st.spinner("Checking how many stocks in the universe are trending..."):
                breadth = get_market_breadth(universe)
            if breadth:
                c1, c2 = st.columns(2)
                c1.metric("% of universe above 50-day SMA", f"{breadth['breadth_pct']:.0f}%")
                c2.metric("Stocks counted", f"{breadth['n_above']}/{breadth['n_stocks']}")
                st.write(f"**{breadth['regime_label']}**")
            else:
                st.warning("Couldn't compute breadth right now -- try again in a moment.")

    use_smart_universe = st.checkbox(
        "🔄 Smart universe narrowing (auto-pick the most liquid, trending names from the pool)",
        value=False,
        help="Ranks the sidebar's full universe by 20-day liquidity and 50-day relative "
             "strength vs Nifty, RIGHT NOW, and only screens the top N -- so a stock that's "
             "gone quiet or illiquid naturally rotates out without you doing anything. "
             "LIVE-ONLY: this uses today's data to pick today's active list, so it's not "
             "used in the Research Backtest tab (using today's leaders to pick which stocks "
             "get backtested over past years would quietly bias the backtest).",
    )
    smart_top_n = None
    if use_smart_universe:
        smart_top_n = st.slider(
            "How many top-ranked stocks to actually screen", 10, min(80, len(universe)),
            min(40, len(universe)),
        )

    if st.button("🔄 Find today's A+ setups", type="primary"):
        if not universe:
            st.error("Select at least one ticker.")
        elif partial_r <= breakeven_r:
            st.error("Fix the exit settings first: partial target must be above breakeven.")
        else:
            screening_universe = universe
            if use_smart_universe:
                with st.spinner("Ranking the pool by liquidity and momentum..."):
                    active_list, ranked_df = select_active_universe(universe, smart_top_n)
                if active_list:
                    screening_universe = active_list
                    with st.expander(f"Active universe selected today ({len(active_list)} of {len(universe)} tickers)"):
                        st.dataframe(ranked_df.head(smart_top_n).round(3), use_container_width=True, hide_index=True)
                else:
                    st.warning("Ranking returned nothing usable -- falling back to the full universe.")

            with st.spinner("Fetching market data, ranking setups, and checking earnings..."):
                watchlist, market_bullish = screen_today(screening_universe, capital, risk_pct, params)

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
                st.caption(f"Split point: **{wf['split_date'].date()}** -- everything before is in-sample, "
                           f"everything from that date onward is out-of-sample.")

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
                        f"Only {oos.get('total_trades', 0)} out-of-sample trades -- too few to "
                        "draw a real conclusion either way. Widen backtest years or the universe."
                    )
                elif exp_out >= exp_in - 0.05:
                    st.success(
                        "Out-of-sample expectancy holds up close to (or above) in-sample. "
                        "That's a real, if modest, sign this generalizes rather than being fit "
                        "to noise in the tuning window."
                    )
                else:
                    st.error(
                        f"Out-of-sample expectancy ({exp_out:.2f}R) is meaningfully worse than "
                        f"in-sample ({exp_in:.2f}R). That's the signature of overfitting -- the "
                        "settings were likely fit to noise in the earlier period, not a real, "
                        "durable edge. Treat this configuration with real skepticism."
                    )

                st.download_button(
                    "⬇️ Download walk-forward trade data (tagged in/out-of-sample)",
                    wf["trades_df"].assign(
                        sample=lambda d: ["in_sample"] * len(wf["in_sample_df"]) + ["out_sample"] * len(wf["out_sample_df"])
                    ).to_csv(index=False),
                    file_name="walk_forward_backtest.csv",
                    mime="text/csv",
                )

    st.caption(
        "The diagnostic CSV is intentionally detailed. Its purpose is to let us "
        "study which characteristics separate winners from losers instead of "
        "blindly optimizing indicator thresholds."
    )

# -----------------------------------------------------------------------------
# POSITIONS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Track open positions")
    st.write("Add a position once. Refresh to get a live stop/trailing recommendation.")

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
st.caption(
    "Educational research tool, not investment advice. A positive backtest is not proof "
    "of future profitability. Use out-of-sample / walk-forward validation before risking money."
)
st.caption(f"Build check: `{APP_VERSION}` (app) / `{ENGINE_VERSION}` (engine) -- "
           f"if this doesn't match what you just pasted, the redeploy didn't take.")

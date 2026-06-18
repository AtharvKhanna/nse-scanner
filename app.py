"""NSE Stock Scanner — Streamlit dashboard.

Two modes:
  • 📈 Long-Term Investing (default): Buy/Hold/Avoid, target, time-to-target, stop-loss.
  • ⚡ Intraday Momentum: the original breakout scanner.

Run:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from scanner import config, scan, longterm, swing

st.set_page_config(page_title="NSE Stock Scanner", page_icon="📈", layout="wide")

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _ist_now_str():
    """Current time in IST (cloud servers run in UTC) — used for the data stamp."""
    return dt.datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")


def _updated_caption(res):
    when = res.get("fetched_at", "—")
    st.caption(f"🕒 **Last updated:** {when}  ·  data refreshes once per day  ·  "
               "click **🔄 Refresh data** in the sidebar for a fresh pull now.")

SIGNAL_COLORS = {"STRONG BUY": "#16c784", "WATCH": "#f3b13e", "AVOID": "#7a7f8c"}
REC_COLORS = {"STRONG BUY": "#16c784", "BUY": "#1fb86e", "ACCUMULATE": "#f3b13e",
              "HOLD": "#9aa0ac", "AVOID": "#ea3943"}


def _throttle_note(df, scope):
    """Warn if the data source returned far fewer rows than expected (cloud rate-limit)."""
    expected_min = 50 if scope == "nifty500" else 200
    if len(df) < expected_min:
        st.warning(
            f"⚠️ Only {len(df)} stocks loaded — the free data source (Yahoo/NSE) likely "
            "rate-limited this server. Wait about a minute and click **🔄 Refresh data** "
            "in the sidebar. Data updates once per day, so this is usually a one-time hiccup."
        )


# ==========================================================================
# Cached scans
# ==========================================================================
@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def cached_intraday(scope, with_news, real_vwap, news_limit, _stamp):
    bar = st.progress(0.0, text="Starting…")
    res = scan.run_scan(scope=scope, with_news=with_news, real_vwap_shortlist=real_vwap,
                        news_limit=news_limit,
                        progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
    res["fetched_at"] = _ist_now_str()
    bar.empty()
    return res


@st.cache_data(ttl=config.DAILY_TTL, show_spinner=False)
def cached_longterm(scope, with_news, news_limit, _stamp):
    bar = st.progress(0.0, text="Starting…")
    res = longterm.run_longterm_scan(scope=scope, with_news=with_news, news_limit=news_limit,
                                     progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
    res["fetched_at"] = _ist_now_str()
    bar.empty()
    return res


@st.cache_data(ttl=config.DAILY_TTL, show_spinner=False)
def cached_swing(scope, with_news, news_limit, _stamp):
    bar = st.progress(0.0, text="Starting…")
    res = swing.run_swing_scan(scope=scope, with_news=with_news, news_limit=news_limit,
                               progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
    res["fetched_at"] = _ist_now_str()
    bar.empty()
    return res


# ==========================================================================
# LONG-TERM VIEW
# ==========================================================================
LT_COLS = {
    "rank": "#", "symbol": "SYMBOL", "name": "COMPANY", "industry": "SECTOR",
    "price": "PRICE ₹", "recommendation": "VERDICT", "target": "TARGET ₹",
    "upside_pct": "UPSIDE %", "months_to_target": "≈MONTHS", "stop_loss": "STOP ₹",
    "downside_pct": "RISK %", "rr": "R:R", "horizon": "HORIZON",
    "pe": "P/E", "roe": "ROE", "news_label": "NEWS", "score": "SCORE",
}


def _lt_prep(df):
    if df.empty:
        return df
    d = df.copy()
    if "roe" in d:
        d["roe"] = pd.to_numeric(d["roe"], errors="coerce") * 100
    cols = [c for c in LT_COLS if c in d.columns]
    return d[cols].rename(columns=LT_COLS)


def _lt_style(df):
    fmt = {"PRICE ₹": "{:.1f}", "TARGET ₹": "{:.1f}", "UPSIDE %": "{:+.1f}",
           "≈MONTHS": "{:.0f}", "STOP ₹": "{:.1f}", "RISK %": "{:.1f}",
           "R:R": "{:.2f}", "P/E": "{:.1f}", "ROE": "{:.0f}%", "SCORE": "{:.1f}"}
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    sty = df.style.format(fmt, na_rep="—")
    if "UPSIDE %" in df.columns:
        sty = sty.apply(lambda s: ["color:#16c784" if v > 0 else "color:#ea3943"
                                   for v in s], subset=["UPSIDE %"])
    if "VERDICT" in df.columns:
        def vcol(s):
            out = []
            for v in s:
                key = next((k for k in REC_COLORS if k in str(v)), None)
                out.append(f"color:{REC_COLORS.get(key,'')};font-weight:600")
            return out
        sty = sty.apply(vcol, subset=["VERDICT"])
    return sty


def render_longterm(scope, with_news, news_limit, stamp,
                    min_upside, min_score, sectors, only_buy):
    res = cached_longterm(scope, with_news, news_limit, stamp)
    df, news_map = res["df"], res["news"]

    st.title("📈 NSE Long-Term Investing — Buy / Target / Stop-Loss")
    if df.empty:
        st.warning("No data. Click 🔄 Refresh or check your connection.")
        st.stop()
    _throttle_note(df, scope)

    # filters
    d = df.copy()
    d = d[d["score"] >= min_score]
    d = d[pd.to_numeric(d["upside_pct"], errors="coerce").fillna(-999) >= min_upside]
    if sectors:
        d = d[d["industry"].isin(sectors)]
    if only_buy:
        d = d[d["recommendation"].str.contains("BUY")]

    buy = d[d["recommendation"].str.contains("BUY")]
    acc = d[d["recommendation"].str.contains("ACCUMULATE")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Buy now", len(buy))
    c2.metric("🟡 Accumulate on dips", len(acc))
    c3.metric("Stocks shown", len(d))
    avg_up = pd.to_numeric(buy["upside_pct"], errors="coerce").mean() if len(buy) else 0
    c4.metric("Avg upside (buys)", f"{avg_up:+.1f}%")
    _updated_caption(res)
    st.caption("Targets = analyst consensus (sanity-checked); time-to-target & stop-loss are "
               "model estimates. Screening tool, not investment advice.")

    t1, t2, t3, t4 = st.tabs(["🟢 Buy Now", "🟡 Accumulate / Watch",
                              "📊 All Stocks", "🔎 Stock Detail"])
    with t1:
        st.caption("Recommendation = BUY or STRONG BUY. Confirm the thesis before investing.")
        st.dataframe(_lt_style(_lt_prep(buy)), use_container_width=True, hide_index=True,
                     height=min(650, 60 + 35 * max(len(buy), 1)))
    with t2:
        st.caption("Fundamentally sound but extended / fully valued — better to add on dips.")
        st.dataframe(_lt_style(_lt_prep(acc)), use_container_width=True, hide_index=True,
                     height=min(650, 60 + 35 * max(len(acc), 1)))
    with t3:
        st.dataframe(_lt_style(_lt_prep(d)), use_container_width=True, hide_index=True,
                     height=650)
        st.download_button("⬇️ Download CSV", d.to_csv(index=False).encode("utf-8"),
                           file_name=f"nse_longterm_{stamp[:10]}.csv", mime="text/csv")
    with t4:
        _lt_detail(d, news_map)


def _fmt(v, suffix="", pct=False, mult=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v*mult:.1f}{'%' if pct else ''}{suffix}"


def _lt_detail(df, news_map):
    syms = df["symbol"].tolist()
    if not syms:
        st.info("No stocks match the current filters.")
        return
    pick = st.selectbox("Select a stock", syms)
    r = df[df["symbol"] == pick].iloc[0]
    rec = r["recommendation"]

    st.markdown(f"## {r['symbol']} — {r['name']}")
    st.markdown(f"### {rec}  ·  Score {r['score']:.0f}/100  ·  {r['industry']}")
    if r.get("news_risk"):
        terms = ", ".join(r.get("flag_terms", [])) or "negative news"
        st.error(f"⚠️ **News override:** recent headlines mention **{terms}** — this blocks "
                 "the Buy rating despite the score. Read the news below before any decision.")

    # The plan the user asked for
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Current price", f"₹{r['price']:.1f}")
    p2.metric("Target", f"₹{r['target']:.1f}", f"{r['upside_pct']:+.1f}% upside")
    p3.metric("Stop-loss", f"₹{r['stop_loss']:.1f}", f"-{r['downside_pct']:.1f}% risk",
              delta_color="inverse")
    rr = r["rr"]
    p4.metric("Risk : Reward", f"1 : {rr:.1f}" if rr and rr > 0 else "—")
    q1, q2, q3 = st.columns(3)
    q1.metric("Should you buy now?",
              "YES" if "BUY" in rec else ("ON DIPS" if "ACCUMULATE" in rec else "NO"))
    q2.metric("Est. time to target",
              f"≈ {r['months_to_target']:.0f} months" if r.get("months_to_target") else "—")
    q3.metric("Holding horizon", r["horizon"])
    st.caption(f"Target basis: {r['target_src']}. Analyst range "
               f"₹{_fmt(r.get('target_low'))}–₹{_fmt(r.get('target_high'))} "
               f"from {r.get('n_analysts') or '—'} analysts ({r.get('analyst_rating') or '—'}). "
               f"Time-to-target assumes the historical/expected trend continues — a rough estimate.")

    a, b = st.columns([1, 1])
    with a:
        st.markdown("**Why — score breakdown**")
        c = r["contrib"]
        bd = pd.DataFrame({
            "factor": ["Analyst (25)", "Quality (25)", "Valuation (20)", "Trend (20)", "News (±10)"],
            "points": [c.get("analyst", 0), c.get("quality", 0), c.get("valuation", 0),
                       c.get("trend", 0), c.get("news", 0)],
        }).set_index("factor")
        st.bar_chart(bd, horizontal=True)

        st.markdown("**Fundamentals**")
        fund = {
            "P/E (TTM)": _fmt(r.get("pe")), "Forward P/E": _fmt(r.get("fwd_pe")),
            "PEG": _fmt(r.get("peg")), "P/B": _fmt(r.get("pb")),
            "ROE": _fmt(r.get("roe"), pct=True, mult=100),
            "Debt/Equity": _fmt(r.get("de")),
            "Revenue growth": _fmt(r.get("rev_growth"), pct=True, mult=100),
            "Earnings growth": _fmt(r.get("earn_growth"), pct=True, mult=100),
            "Profit margin": _fmt(r.get("margin"), pct=True, mult=100),
            "Div yield": _fmt(r.get("div_yield"), pct=True, mult=100),
            "Beta": _fmt(r.get("beta")),
            "12-mo return": _fmt(r.get("ret_12m_pct"), pct=True),
            "Above 200-DMA": "Yes ✅" if r.get("above_200dma") else "No ⚠️",
            "52-wk position": _fmt(r.get("pos_52w_pct"), pct=True),
        }
        st.table(pd.DataFrame(fund.items(), columns=["Metric", "Value"]).set_index("Metric"))

    with b:
        st.markdown("**📈 Price (1 year)**")
        try:
            import yfinance as yf
            h = yf.download(r["ticker"], period="1y", interval="1d", progress=False,
                            auto_adjust=False)
            if not h.empty:
                cl = h["Close"]
                cl = cl.iloc[:, 0] if hasattr(cl, "columns") else cl
                chart = pd.DataFrame({"Close": cl})
                chart["200-DMA"] = cl.rolling(200).mean()
                st.line_chart(chart)
        except Exception:
            st.caption("Chart unavailable.")

        st.markdown("**📰 News & sentiment**")
        n = news_map.get(pick)
        if n and n["headlines"]:
            st.markdown(f"Overall: **{n['label']}** (news points {n['points']:+d})")
            for h in n["headlines"]:
                emo = "🟢" if h["sent"] > 0.1 else ("🔴" if h["sent"] < -0.1 else "⚪")
                st.markdown(f"{emo} [{h['title']}]({h['link']})  \n"
                            f"<span style='color:#888;font-size:0.8em'>{h['published']} · "
                            f"sentiment {h['sent']:+.2f}</span>", unsafe_allow_html=True)
        else:
            st.caption("News is fetched only for buy/accumulate candidates.")


# ==========================================================================
# INTRADAY VIEW (original scanner)
# ==========================================================================
INTRA_COLS = {
    "rank": "RANK", "symbol": "SYMBOL", "name": "COMPANY", "cmp": "CMP ₹",
    "price_change_pct": "CHG %", "gap_pct": "GAP %", "rvol": "RVOL",
    "vwap_pos": "VWAP", "rsi": "RSI", "atr_pct": "ATR %", "deliv_per": "DELIV %",
    "trend": "TREND", "breakout_status": "BREAKOUT", "smart_money": "SMART MONEY",
    "news_label": "NEWS", "score": "SCORE", "signal": "SIGNAL", "tier": "TIER",
}


def _intra_prep(df):
    if df.empty:
        return df
    d = df.copy()
    d["vwap_pos"] = d["above_vwap"].map({True: "ABOVE", False: "BELOW"})
    for c in ["cmp", "price_change_pct", "gap_pct", "rvol", "rsi", "atr_pct", "deliv_per", "score"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    cols = [c for c in INTRA_COLS if c in d.columns]
    return d[cols].rename(columns=INTRA_COLS)


def _intra_style(df):
    fmt = {"CMP ₹": "{:.1f}", "CHG %": "{:+.2f}", "GAP %": "{:+.2f}", "RVOL": "{:.2f}",
           "RSI": "{:.0f}", "ATR %": "{:.1f}", "DELIV %": "{:.0f}", "SCORE": "{:.1f}"}
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    sty = df.style.format(fmt, na_rep="—")
    if "SIGNAL" in df.columns:
        sty = sty.apply(lambda s: [f"color:{SIGNAL_COLORS.get(v,'')};font-weight:600"
                                   for v in s], subset=["SIGNAL"])
    if "CHG %" in df.columns:
        sty = sty.apply(lambda s: ["color:#16c784" if v > 0 else "color:#ea3943"
                                   for v in s], subset=["CHG %"])
    return sty


def render_intraday(scope, with_news, real_vwap, news_limit, stamp,
                    min_score, min_rvol, only_inst):
    res = cached_intraday(scope, with_news, real_vwap, news_limit, stamp)
    df, regime, news_map = res["df"], res["regime"], res["news"]

    st.title("⚡ NSE Intraday Momentum & Breakout Scanner")
    if df.empty:
        st.warning("No data. Click 🔄 Refresh or check your connection.")
        st.stop()
    _throttle_note(df, scope)

    d = df[df["score"] >= min_score]
    if min_rvol > 0:
        d = d[d["rvol"].fillna(0) >= min_rvol]
    if only_inst:
        d = d[d["smart_money"] == "INSTITUTIONAL BUYING"]
    watchlist, watch_tom, full = scan.split_views(d)

    reg = regime["regime"].upper()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Regime", reg, f"{regime['change_pct']:+.2f}% Nifty")
    c2.metric("Strong Buy", len(watchlist))
    c3.metric("Watch Tomorrow", len(watch_tom))
    c4.metric("Scanned", len(df))
    _updated_caption(res)

    t1, t2, t3, t4 = st.tabs(["🔥 Watchlist", "👀 Watch Tomorrow", "📊 Full Data", "🔎 Detail"])
    with t1:
        if watchlist.empty:
            st.info("No STRONG BUY signals right now. Check 👀 Watch Tomorrow.")
        else:
            st.dataframe(_intra_style(_intra_prep(watchlist)), use_container_width=True,
                         hide_index=True, height=min(600, 60 + 35 * len(watchlist)))
    with t2:
        st.dataframe(_intra_style(_intra_prep(watch_tom)), use_container_width=True,
                     hide_index=True, height=min(600, 60 + 35 * max(len(watch_tom), 1)))
    with t3:
        st.dataframe(_intra_style(_intra_prep(full)), use_container_width=True,
                     hide_index=True, height=650)
        st.download_button("⬇️ Download CSV", full.to_csv(index=False).encode("utf-8"),
                           file_name=f"nse_intraday_{stamp[:10]}.csv", mime="text/csv")
    with t4:
        syms = full["symbol"].tolist()
        pick = st.selectbox("Select a stock", syms) if syms else None
        if pick:
            row = full[full["symbol"] == pick].iloc[0]
            st.markdown(f"### {row['symbol']} — {row['name']}  ·  {row['signal']} "
                        f"(score {row['score']:.0f})")
            c = row["contrib"]
            bd = pd.DataFrame({"factor": ["RVol (25)", "Momentum (20)", "Location (20)",
                                          "Smart Money (15)", "RSI (10)", "News (±10)"],
                               "points": [c.get("rvol", 0), c.get("momentum", 0),
                                          c.get("location", 0), c.get("smart_money", 0),
                                          c.get("rsi", 0), c.get("news", 0)]}).set_index("factor")
            st.bar_chart(bd, horizontal=True)
            n = news_map.get(pick)
            if n and n["headlines"]:
                st.markdown(f"**News: {n['label']}** ({n['points']:+d})")
                for h in n["headlines"]:
                    emo = "🟢" if h["sent"] > 0.1 else ("🔴" if h["sent"] < -0.1 else "⚪")
                    st.markdown(f"{emo} [{h['title']}]({h['link']}) "
                                f"<span style='color:#888;font-size:0.8em'>· {h['sent']:+.2f}</span>",
                                unsafe_allow_html=True)


# ==========================================================================
# SWING / SHORT-TERM VIEW  (≈15 days to 2 months)
# ==========================================================================
SW_COLS = {
    "rank": "#", "symbol": "SYMBOL", "name": "COMPANY", "industry": "SECTOR",
    "price": "PRICE ₹", "recommendation": "VERDICT", "target": "TARGET ₹",
    "upside_pct": "UPSIDE %", "days_to_target": "≈DAYS", "stop_loss": "STOP ₹",
    "downside_pct": "RISK %", "rr": "R:R", "ret_1m_pct": "1M %", "ret_3m_pct": "3M %",
    "rs_3m_pct": "vs NIFTY", "rsi": "RSI", "rvol": "RVOL", "news_label": "NEWS",
    "score": "SCORE",
}


def _sw_prep(df):
    if df.empty:
        return df
    cols = [c for c in SW_COLS if c in df.columns]
    return df[cols].rename(columns=SW_COLS)


def _sw_style(df):
    fmt = {"PRICE ₹": "{:.1f}", "TARGET ₹": "{:.1f}", "UPSIDE %": "{:+.1f}",
           "≈DAYS": "{:.0f}", "STOP ₹": "{:.1f}", "RISK %": "{:.1f}", "R:R": "{:.2f}",
           "1M %": "{:+.1f}", "3M %": "{:+.1f}", "vs NIFTY": "{:+.1f}", "RSI": "{:.0f}",
           "RVOL": "{:.2f}", "SCORE": "{:.1f}"}
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    sty = df.style.format(fmt, na_rep="—")
    for col in ["UPSIDE %", "1M %", "3M %", "vs NIFTY"]:
        if col in df.columns:
            sty = sty.apply(lambda s: ["color:#16c784" if v > 0 else "color:#ea3943"
                                       for v in s], subset=[col])
    if "VERDICT" in df.columns:
        def vcol(s):
            out = []
            for v in s:
                key = next((k for k in REC_COLORS if k in str(v)), None)
                if "WATCH" in str(v):
                    key = "ACCUMULATE"
                out.append(f"color:{REC_COLORS.get(key,'')};font-weight:600")
            return out
        sty = sty.apply(vcol, subset=["VERDICT"])
    return sty


def render_swing(scope, with_news, news_limit, stamp, min_upside, min_score, only_buy):
    res = cached_swing(scope, with_news, news_limit, stamp)
    df, news_map = res["df"], res["news"]

    st.title("📅 NSE Swing Trades — 15 days to 2 months")
    if df.empty:
        st.warning("No data. Click 🔄 Refresh or check your connection.")
        st.stop()
    _throttle_note(df, scope)

    d = df.copy()
    d = d[d["score"] >= min_score]
    d = d[pd.to_numeric(d["upside_pct"], errors="coerce").fillna(-999) >= min_upside]
    if only_buy:
        d = d[d["recommendation"].str.contains("BUY")]
    buy = d[d["recommendation"].str.contains("BUY")]
    watch = d[d["recommendation"].str.contains("WATCH")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Buy now", len(buy))
    c2.metric("🟡 Watch (near setup)", len(watch))
    c3.metric("Stocks shown", len(d))
    c4.metric("Nifty 3-mo", f"{res['nifty_ret_3m']:+.1f}%")
    _updated_caption(res)
    st.caption("Targets & stops are ATR-based; holding window ≈ 2–10 weeks. "
               "Screening tool — not investment advice.")

    t1, t2, t3, t4 = st.tabs(["🟢 Buy Now", "🟡 Watch", "📊 All Stocks", "🔎 Stock Detail"])
    with t1:
        st.caption("Medium-term uptrend + momentum + breakout structure. Enter near the "
                   "level, keep the stop tight.")
        st.dataframe(_sw_style(_sw_prep(buy)), use_container_width=True, hide_index=True,
                     height=min(650, 60 + 35 * max(len(buy), 1)))
    with t2:
        st.caption("Setting up but not confirmed — watch for a volume breakout.")
        st.dataframe(_sw_style(_sw_prep(watch)), use_container_width=True, hide_index=True,
                     height=min(650, 60 + 35 * max(len(watch), 1)))
    with t3:
        st.dataframe(_sw_style(_sw_prep(d)), use_container_width=True, hide_index=True, height=650)
        st.download_button("⬇️ Download CSV", d.to_csv(index=False).encode("utf-8"),
                           file_name=f"nse_swing_{stamp[:10]}.csv", mime="text/csv")
    with t4:
        syms = d["symbol"].tolist()
        if not syms:
            st.info("No stocks match the current filters.")
            return
        pick = st.selectbox("Select a stock", syms)
        r = d[d["symbol"] == pick].iloc[0]
        rec = r["recommendation"]
        st.markdown(f"## {r['symbol']} — {r['name']}")
        st.markdown(f"### {rec}  ·  Score {r['score']:.0f}/100  ·  {r['industry']}")
        if r.get("news_risk"):
            terms = ", ".join(r.get("flag_terms", [])) or "negative news"
            st.error(f"⚠️ **News override:** recent headlines mention **{terms}** — this blocks "
                     "the Buy rating despite the score. Read the news below before any trade.")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Current price", f"₹{r['price']:.1f}")
        p2.metric("Target", f"₹{r['target']:.1f}", f"{r['upside_pct']:+.1f}%")
        p3.metric("Stop-loss", f"₹{r['stop_loss']:.1f}", f"-{r['downside_pct']:.1f}%",
                  delta_color="inverse")
        p4.metric("Risk : Reward", f"1 : {r['rr']:.1f}" if r['rr'] else "—")
        q1, q2, q3 = st.columns(3)
        q1.metric("Should you buy now?",
                  "YES" if "BUY" in rec else ("WATCH" if "WATCH" in rec else "NO"))
        q2.metric("Est. time to target", f"≈ {r['days_to_target']:.0f} days")
        q3.metric("Momentum (1M / 3M)",
                  f"{_fmt(r.get('ret_1m_pct'),pct=True)} / {_fmt(r.get('ret_3m_pct'),pct=True)}")

        a, b = st.columns([1, 1])
        with a:
            st.markdown("**Why — score breakdown**")
            c = r["contrib"]
            bd = pd.DataFrame({
                "factor": ["Trend (25)", "Momentum/RS (25)", "Breakout (20)",
                           "Volume (10)", "RSI (10)", "News (±10)"],
                "points": [c.get("trend", 0), c.get("momentum", 0), c.get("breakout", 0),
                           c.get("volume", 0), c.get("rsi", 0), c.get("news", 0)],
            }).set_index("factor")
            st.bar_chart(bd, horizontal=True)
            tech = {
                "RSI (14)": _fmt(r.get("rsi")), "ATR %": _fmt(r.get("atr_pct"), pct=True),
                "RVOL": _fmt(r.get("rvol")),
                "1-mo return": _fmt(r.get("ret_1m_pct"), pct=True),
                "3-mo return": _fmt(r.get("ret_3m_pct"), pct=True),
                "vs Nifty (3-mo)": _fmt(r.get("rs_3m_pct"), pct=True),
                "Above 50-EMA": "Yes ✅" if r.get("above_ema50") else "No ⚠️",
                "Below 52-wk high by": _fmt(r.get("dist_52w_high_pct"), pct=True),
            }
            st.table(pd.DataFrame(tech.items(), columns=["Metric", "Value"]).set_index("Metric"))
        with b:
            st.markdown("**📈 Price (6 months)**")
            try:
                import yfinance as yf
                h = yf.download(r["ticker"], period="6mo", interval="1d", progress=False,
                                auto_adjust=False)
                if not h.empty:
                    cl = h["Close"]
                    cl = cl.iloc[:, 0] if hasattr(cl, "columns") else cl
                    ch = pd.DataFrame({"Close": cl})
                    ch["50-EMA"] = cl.ewm(span=50, adjust=False).mean()
                    st.line_chart(ch)
            except Exception:
                st.caption("Chart unavailable.")
            st.markdown("**📰 News & sentiment**")
            n = news_map.get(pick)
            if n and n["headlines"]:
                st.markdown(f"Overall: **{n['label']}** (news points {n['points']:+d})")
                for h in n["headlines"]:
                    emo = "🟢" if h["sent"] > 0.1 else ("🔴" if h["sent"] < -0.1 else "⚪")
                    st.markdown(f"{emo} [{h['title']}]({h['link']})  \n"
                                f"<span style='color:#888;font-size:0.8em'>{h['published']} · "
                                f"sentiment {h['sent']:+.2f}</span>", unsafe_allow_html=True)
            else:
                st.caption("News is fetched only for buy/watch candidates.")


# ==========================================================================
# Sidebar + routing
# ==========================================================================
st.sidebar.title("📈 NSE Scanner")
mode = st.sidebar.radio("Mode", ["Swing (15d – 2 months)", "Long-Term Investing",
                                 "Intraday Momentum"])
scope = st.sidebar.radio("Universe", ["nifty500", "all"],
                         format_func=lambda s: "Nifty 500" if s == "nifty500" else "All (~2,300)")
with_news = st.sidebar.toggle("Apply News factor", value=True)
news_limit = st.sidebar.slider("News depth (top N)", 10, 80, 40, step=10)

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    cached_intraday.clear()
    cached_longterm.clear()
    cached_swing.clear()
    st.session_state["stamp"] = dt.datetime.now().isoformat()
stamp = st.session_state.get("stamp", dt.date.today().isoformat())

st.sidebar.markdown("---")
if mode == "Long-Term Investing":
    st.sidebar.markdown("**Filters**")
    min_upside = st.sidebar.slider("Min upside %", -20, 60, 0, step=5)
    min_score = st.sidebar.slider("Min score", 0, 100, 0, step=5)
    only_buy = st.sidebar.checkbox("Buy-rated only", value=False)
    # sector filter needs the data; fetch then filter inside render — pass empty here
    sectors_sel = []
    st.sidebar.caption("Data: Yahoo Finance fundamentals + analyst targets, NSE list, "
                       "Google News sentiment. Long-term data cached for the day.")
    render_longterm(scope, with_news, news_limit, stamp,
                    min_upside, min_score, sectors_sel, only_buy)
elif mode == "Swing (15d – 2 months)":
    st.sidebar.markdown("**Filters**")
    sw_min_upside = st.sidebar.slider("Min upside %", 0, 40, 0, step=2)
    sw_min_score = st.sidebar.slider("Min score", 0, 100, 0, step=5)
    sw_only_buy = st.sidebar.checkbox("Buy-rated only", value=False)
    st.sidebar.caption("Technical swing model (trend + relative strength + breakout + "
                       "volume + RSI) with ATR-based targets/stops. Holding ≈ 2–10 weeks.")
    render_swing(scope, with_news, news_limit, stamp,
                 sw_min_upside, sw_min_score, sw_only_buy)
else:
    st.sidebar.markdown("**Filters**")
    real_vwap = st.sidebar.toggle("Real intraday VWAP (shortlist)", value=True)
    min_score = st.sidebar.slider("Min score", 0, 100, 0, step=5)
    min_rvol = st.sidebar.slider("Min RVOL", 0.0, 10.0, 0.0, step=0.5)
    only_inst = st.sidebar.checkbox("Institutional (Tier 1) only", value=False)
    st.sidebar.caption("Data: Yahoo Finance (~15-min delayed) + NSE delivery bhavcopy.")
    render_intraday(scope, with_news, real_vwap, news_limit, stamp,
                    min_score, min_rvol, only_inst)

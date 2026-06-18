"""Orchestrator: run a full scan = fetch -> features -> score -> (news on shortlist).

`run_scan` is pure (no Streamlit) and accepts a `progress` callback so the UI can
show a progress bar. The app wraps it with caching.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import pandas as pd

from . import config, data, news as news_mod, signals, universe


def run_scan(scope: str = "nifty500",
             with_news: bool = True,
             news_limit: int = 40,
             real_vwap_shortlist: bool = True,
             progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Return {'df': DataFrame, 'regime': dict, 'news': {symbol: news_dict}}."""
    def report(p, msg):
        if progress:
            progress(p, msg)

    report(0.02, "Loading universe…")
    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    meta = {u["ticker"]: u for u in uni}

    report(0.08, "Fetching market regime (Nifty 50)…")
    regime = data.fetch_index_regime()

    report(0.15, "Downloading delivery % (NSE bhavcopy)…")
    deliv_map = data.fetch_delivery_map()

    report(0.25, f"Downloading prices for {len(tickers)} stocks…")
    daily = data.fetch_daily(tickers, config.HISTORY_DAYS)
    report(0.6, f"Scoring {len(daily)} stocks…")

    rows = []
    for t, df in daily.items():
        m = meta.get(t, {})
        sym = m.get("symbol", t.replace(".NS", ""))
        f = signals.build_features(
            t, df, sym, m.get("name", ""), m.get("industry", ""),
            delivery=deliv_map.get(sym),
        )
        if f is None:
            continue
        rows.append(signals.score_features(f, regime["regime"], news=None,
                                           apply_news=False))

    df = signals.to_dataframe(rows)
    news_map = {}

    if not df.empty and with_news:
        # Shortlist = best scoring (Strong Buy + Watch lean), fetch news for them
        shortlist = df[df["signal"].isin(["STRONG BUY", "WATCH"])].head(news_limit)
        items = [{"symbol": r["symbol"], "name": r["name"]}
                 for _, r in shortlist.iterrows()]
        report(0.75, f"Reading news for {len(items)} shortlisted stocks…")
        news_map = news_mod.fetch_news_batch(items)

        # Optional real intraday VWAP for the shortlist (more accurate location)
        idx = df.index[df["signal"].isin(["STRONG BUY", "WATCH"])][:news_limit]
        for i in idx:
            sym = df.at[i, "symbol"]
            tkr = df.at[i, "ticker"]
            n = news_map.get(sym)
            vwap = data.fetch_intraday_vwap(tkr) if real_vwap_shortlist else float("nan")
            # rebuild feature with real vwap, then re-score with news
            f = signals.build_features(
                tkr, daily[tkr], sym, df.at[i, "name"], df.at[i, "industry"],
                delivery=deliv_map.get(sym), intraday_vwap=vwap,
            )
            if f is None:
                continue
            scored = signals.score_features(f, regime["regime"], news=n, apply_news=True)
            for k, v in scored.items():
                df.at[i, k] = v
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    report(1.0, "Done")
    return {"df": df, "regime": regime, "news": news_map}


def split_views(df: pd.DataFrame):
    """Return (watchlist, watch_tomorrow, full) mirroring the Excel tabs."""
    if df.empty:
        return df, df, df
    watchlist = df[df["signal"] == "STRONG BUY"].copy()
    watch_tom = df[df["signal"] == "WATCH"].copy()
    return watchlist, watch_tom, df

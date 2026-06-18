"""Data acquisition: Yahoo Finance (prices) + NSE bhavcopy (delivery %).

All functions are plain (no Streamlit) so they're testable; the app wraps them
with st.cache_data. Everything is best-effort: a failed optional source degrades
to neutral rather than crashing the scan.
"""
from __future__ import annotations

import datetime as dt
import io
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf

from . import config

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# --------------------------------------------------------------------------
# Daily OHLCV history (batched)
# --------------------------------------------------------------------------
def fetch_daily(tickers: List[str], days: int = config.HISTORY_DAYS,
                chunk: int = 120) -> Dict[str, pd.DataFrame]:
    """Return {ticker: daily OHLCV DataFrame (oldest->newest)}. Skips failures."""
    import time

    out: Dict[str, pd.DataFrame] = {}
    period = f"{max(days, 30)}d"
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        df = None
        for attempt in range(2):  # retry once on transient cloud rate-limits
            try:
                df = yf.download(
                    batch, period=period, interval="1d", group_by="ticker",
                    threads=True, progress=False, auto_adjust=False,
                )
                if df is not None and len(df):
                    break
            except Exception:
                df = None
            time.sleep(1.5 * (attempt + 1))
        if df is None or len(df) == 0:
            continue
        multi = isinstance(df.columns, pd.MultiIndex)
        for t in batch:
            try:
                if multi:
                    if t not in df.columns.get_level_values(0):
                        continue
                    sub = df[t]            # works for single- and multi-ticker batches
                else:
                    sub = df
                sub = sub.dropna(how="all")
                if len(sub) >= 20 and sub["Close"].notna().any():
                    out[t] = sub
            except Exception:
                continue
    return out


def fetch_intraday_vwap(ticker: str) -> float:
    """Real session VWAP from today's 5-min bars (for the shortlist)."""
    from .indicators import intraday_vwap
    try:
        bars = yf.download(ticker, period="1d", interval="5m", progress=False,
                           auto_adjust=False)
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        return intraday_vwap(bars)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# NSE delivery % + turnover (smart-money proxy)
# --------------------------------------------------------------------------
def fetch_delivery_map(max_lookback: int = 6) -> Dict[str, dict]:
    """{SYMBOL: {'deliv_per': float, 'turnover_cr': float}} from latest bhavcopy."""
    sess = requests.Session()
    sess.headers.update(_NSE_HEADERS)
    for back in range(max_lookback):
        d = dt.date.today() - dt.timedelta(days=back)
        ds = d.strftime("%d%m%Y")
        url = (
            "https://nsearchives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{ds}.csv"
        )
        try:
            r = sess.get(url, timeout=20)
            if r.status_code != 200 or len(r.content) < 1000:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
            df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
            out = {}
            for _, row in df.iterrows():
                try:
                    deliv = float(str(row.get("DELIV_PER")).strip())
                except (ValueError, TypeError):
                    deliv = float("nan")
                try:
                    turn_cr = float(str(row.get("TURNOVER_LACS")).strip()) / 100.0
                except (ValueError, TypeError):
                    turn_cr = float("nan")
                out[row["SYMBOL"]] = {"deliv_per": deliv, "turnover_cr": turn_cr}
            if out:
                return out
        except Exception:
            continue
    return {}


# --------------------------------------------------------------------------
# Market regime (Nifty 50)
# --------------------------------------------------------------------------
def fetch_index_regime() -> dict:
    """Classify the broad market: 'bull' / 'neutral' / 'bear' from ^NSEI."""
    try:
        df = yf.download("^NSEI", period="60d", interval="1d", progress=False,
                         auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        last = close.iloc[-1]
        prev = close.iloc[-2]
        chg = (last - prev) / prev * 100
        above = last > ema20
        if above and chg > -0.3:
            regime = "bull"
        elif not above and chg < -0.3:
            regime = "bear"
        else:
            regime = "neutral"
        return {"regime": regime, "nifty": float(last), "change_pct": float(chg),
                "above_ema20": bool(above)}
    except Exception:
        return {"regime": "neutral", "nifty": float("nan"), "change_pct": float("nan"),
                "above_ema20": True}

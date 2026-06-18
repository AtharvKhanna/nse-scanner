"""Fundamentals + analyst data from Yahoo Finance (threaded, disk-cached).

`.info` is one request per ticker, so we fetch in a thread pool and persist the
result to data/cache for the day — long-term data doesn't need minute-by-minute
refresh.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import yfinance as yf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(HERE, "data", "cache")

INFO_FIELDS = [
    "longName", "sector", "industry", "currentPrice", "previousClose", "marketCap",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice", "recommendationKey",
    "numberOfAnalystOpinions", "trailingPE", "forwardPE", "priceToBook", "pegRatio",
    "returnOnEquity", "debtToEquity", "earningsGrowth", "revenueGrowth",
    "profitMargins", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "beta", "dividendYield",
]


def _cache_path(scope: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = dt.date.today().isoformat()
    return os.path.join(CACHE_DIR, f"fundamentals_{scope}_{today}.pkl")


def _fetch_one(ticker: str) -> Optional[dict]:
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("currentPrice") is None:
            return None
        out = {k: info.get(k) for k in INFO_FIELDS}
        out["ticker"] = ticker
        return out
    except Exception:
        return None


def fetch_fundamentals(tickers: List[str], scope: str = "nifty500",
                       use_cache: bool = True, workers: int = 10,
                       progress=None) -> Dict[str, dict]:
    """Return {ticker: info_dict}. Cached to disk per scope per day."""
    path = _cache_path(scope)
    # Only trust the cache if it covers most of the requested universe — guards
    # against a partial/poisoned cache (e.g. from a small test run).
    min_coverage = max(20, int(0.5 * len(tickers)))
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if cached and len(cached) >= min_coverage:
                if progress:
                    progress(1.0, f"Loaded {len(cached)} cached fundamentals")
                return cached
        except Exception:
            pass

    out: Dict[str, dict] = {}
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_fetch_one, tickers):
            done += 1
            if res:
                out[res["ticker"]] = res
            if progress and done % 25 == 0:
                progress(done / total, f"Fundamentals {done}/{total}…")

    try:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    except Exception:
        pass
    return out

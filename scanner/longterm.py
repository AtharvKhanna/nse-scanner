"""Long-term investing model: score a stock, then derive a concrete plan —
recommendation, target price, expected time-to-target, stop-loss and risk:reward.

Inputs: Yahoo `.info` fundamentals/analyst data + daily price history + news.
Everything is approximate by nature and clearly labelled as such in the UI.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, data, fundamentals, indicators, news as news_mod, universe


def _num(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------
# Sub-scores (0..1)
# --------------------------------------------------------------------------
def _analyst_sub(price, target_mean, rating, n_analysts):
    if not price or not target_mean:
        return 0.5, None  # neutral when no coverage
    upside = target_mean / price - 1
    up_score = _clamp(upside / config.LT_UPSIDE_FULL)
    rating_bonus = {"strong_buy": 1.0, "buy": 0.8, "hold": 0.4,
                    "underperform": 0.15, "sell": 0.0}.get(
        (rating or "").lower(), 0.5)
    coverage = _clamp((n_analysts or 0) / 20.0)
    score = 0.6 * up_score + 0.3 * rating_bonus + 0.1 * coverage
    return _clamp(score), upside


def _quality_sub(info):
    roe = _num(info.get("returnOnEquity"))
    margin = _num(info.get("profitMargins"))
    de = _num(info.get("debtToEquity"))
    eg = _num(info.get("earningsGrowth"))
    rg = _num(info.get("revenueGrowth"))
    parts = []
    if roe is not None:
        parts.append(_clamp(roe / config.LT_ROE_GOOD))
    if margin is not None:
        parts.append(_clamp(margin / config.LT_MARGIN_GOOD))
    if de is not None:
        # 0 debt -> 1.0 ; LT_DE_GOOD -> ~0.7 ; LT_DE_POOR+ -> ~0
        parts.append(_clamp(1 - (de - config.LT_DE_GOOD) /
                            (config.LT_DE_POOR - config.LT_DE_GOOD) * 0.7
                            if de > config.LT_DE_GOOD else 1.0))
    growth = [g for g in (eg, rg) if g is not None]
    if growth:
        parts.append(_clamp(max(growth) / config.LT_GROWTH_GOOD))
    return float(np.mean(parts)) if parts else 0.5


def _valuation_sub(info):
    peg = _num(info.get("pegRatio"))
    pb = _num(info.get("priceToBook"))
    fpe = _num(info.get("forwardPE"))
    parts = []
    if peg is not None and peg > 0:
        parts.append(_clamp(1 - (peg - config.LT_PEG_CHEAP) /
                            (config.LT_PEG_EXPENSIVE - config.LT_PEG_CHEAP)))
    if pb is not None and pb > 0:
        parts.append(_clamp(1 - (pb - config.LT_PB_CHEAP) /
                            (config.LT_PB_EXPENSIVE - config.LT_PB_CHEAP)))
    if fpe is not None and fpe > 0:
        parts.append(_clamp(1 - (fpe - 12) / (45 - 12)))  # 12x cheap, 45x rich
    return float(np.mean(parts)) if parts else 0.5


def _trend_sub(price, sma50, sma200, ret_12m, pos_52w):
    parts = []
    if sma200:
        parts.append(1.0 if price > sma200 else 0.0)
    if sma50 and sma200:
        parts.append(1.0 if sma50 > sma200 else 0.0)        # golden cross
    if ret_12m is not None:
        parts.append(_clamp((ret_12m + 0.10) / 0.40))        # -10%..+30% -> 0..1
    if pos_52w is not None:
        # reward room to run but avoid falling knives: peak around 40-80% of range
        parts.append(_clamp(1 - abs(pos_52w - 0.6) / 0.6))
    return float(np.mean(parts)) if parts else 0.5


# --------------------------------------------------------------------------
# Plan: target, stop-loss, days-to-target
# --------------------------------------------------------------------------
def _target_price(price, info, upside, ret_12m):
    tmean = _num(info.get("targetMeanPrice"))
    lo, hi = price * config.LT_TARGET_MIN_MULT, price * config.LT_TARGET_MAX_MULT
    if tmean and lo <= tmean <= hi:
        return tmean, "analyst consensus"
    # fallback: grow price by expected fundamental growth (capped)
    eg = _num(info.get("earningsGrowth")) or _num(info.get("revenueGrowth")) or 0.10
    g = _clamp(eg, 0.05, 0.30)
    return price * (1 + g), "model (growth-based)"


def _stop_loss(price, sma200, atr_pct):
    base = price * (1 - config.LT_MAX_RISK)              # 15% floor
    sl = base
    if sma200 and sma200 < price:
        sl = max(base, sma200 * 0.97)                   # tighter if 200-DMA near
    # never tighter than LT_MIN_RISK
    sl = min(sl, price * (1 - config.LT_MIN_RISK))
    return sl


def _days_to_target(upside_pct, ret_12m_pct):
    speed = max(upside_pct, ret_12m_pct or 0, config.LT_SPEED_FLOOR)
    if upside_pct <= 0:
        return None
    days = 365.0 * upside_pct / speed
    return int(_clamp(days, config.LT_DAYS_MIN, config.LT_DAYS_MAX))


def _recommendation(score, price, sma200, upside):
    below_200 = sma200 and price < sma200
    if score >= config.LT_STRONG_BUY and not below_200:
        return "🟢 STRONG BUY"
    if score >= config.LT_BUY:
        return "🟢 BUY"
    if score >= config.LT_ACCUMULATE:
        return "🟡 ACCUMULATE ON DIPS"
    if score >= config.LT_HOLD:
        return "⚪ HOLD"
    return "🔴 AVOID"


# --------------------------------------------------------------------------
# Score one stock
# --------------------------------------------------------------------------
def evaluate(symbol, name, industry, ticker, info, daily, news=None,
             regime="neutral", apply_news=True) -> Optional[dict]:
    price = _num(info.get("currentPrice"))
    if price is None and daily is not None and len(daily):
        price = float(daily["Close"].iloc[-1])
    if price is None:
        return None

    close = daily["Close"].dropna() if daily is not None else pd.Series(dtype=float)
    sma50 = indicators.ema(close, 50) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    atrp = indicators.atr_pct(daily) if daily is not None and len(daily) > 15 else None
    ret_12m = None
    if len(close) >= 230:
        ret_12m = float(close.iloc[-1] / close.iloc[-min(252, len(close))] - 1)
    hi52 = _num(info.get("fiftyTwoWeekHigh"))
    lo52 = _num(info.get("fiftyTwoWeekLow"))
    pos_52w = ((price - lo52) / (hi52 - lo52)) if (hi52 and lo52 and hi52 > lo52) else None

    a_sub, upside = _analyst_sub(price, _num(info.get("targetMeanPrice")),
                                 info.get("recommendationKey"),
                                 info.get("numberOfAnalystOpinions"))
    q_sub = _quality_sub(info)
    v_sub = _valuation_sub(info)
    t_sub = _trend_sub(price, sma50, sma200, ret_12m, pos_52w)

    w = config.LT_WEIGHTS
    contrib = {
        "analyst": round(a_sub * w["analyst"], 1),
        "quality": round(q_sub * w["quality"], 1),
        "valuation": round(v_sub * w["valuation"], 1),
        "trend": round(t_sub * w["trend"], 1),
    }
    news_points = int((news or {}).get("points", 0)) if (apply_news and news) else 0
    contrib["news"] = news_points
    score = _clamp(sum(contrib.values()), 0, 100)

    target, target_src = _target_price(price, info, upside, ret_12m)
    upside_pct = (target / price - 1) * 100
    sl = _stop_loss(price, sma200, atrp)
    downside_pct = (1 - sl / price) * 100
    rr = (upside_pct / downside_pct) if downside_pct > 0 else None
    days = _days_to_target(upside_pct, (ret_12m or 0) * 100)
    horizon = ("Short (<9m)" if days and days < 270 else
               "Medium (9-24m)" if days and days < 730 else "Long (24m+)")

    return {
        "symbol": symbol, "name": info.get("longName") or name,
        "industry": info.get("sector") or industry, "ticker": ticker,
        "price": round(price, 2), "score": round(score, 1), "contrib": contrib,
        "recommendation": _recommendation(score, price, sma200, upside),
        "target": round(target, 2), "target_src": target_src,
        "upside_pct": round(upside_pct, 1),
        "stop_loss": round(sl, 2), "downside_pct": round(downside_pct, 1),
        "rr": round(rr, 2) if rr else None,
        "days_to_target": days, "months_to_target": round(days / 30.0, 1) if days else None,
        "horizon": horizon,
        # context fields
        "pe": _num(info.get("trailingPE")), "fwd_pe": _num(info.get("forwardPE")),
        "peg": _num(info.get("pegRatio")), "pb": _num(info.get("priceToBook")),
        "roe": _num(info.get("returnOnEquity")), "de": _num(info.get("debtToEquity")),
        "rev_growth": _num(info.get("revenueGrowth")),
        "earn_growth": _num(info.get("earningsGrowth")),
        "margin": _num(info.get("profitMargins")),
        "div_yield": _num(info.get("dividendYield")),
        "mcap": _num(info.get("marketCap")),
        "beta": _num(info.get("beta")),
        "analyst_rating": info.get("recommendationKey"),
        "n_analysts": info.get("numberOfAnalystOpinions"),
        "target_low": _num(info.get("targetLowPrice")),
        "target_high": _num(info.get("targetHighPrice")),
        "sma200": round(sma200, 2) if sma200 else None,
        "above_200dma": bool(sma200 and price > sma200),
        "ret_12m_pct": round(ret_12m * 100, 1) if ret_12m is not None else None,
        "pos_52w_pct": round(pos_52w * 100, 0) if pos_52w is not None else None,
        "news_points": news_points, "news_label": (news or {}).get("label", "—"),
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def run_longterm_scan(scope="nifty500", with_news=True, news_limit=40,
                      use_cache=True,
                      progress: Optional[Callable[[float, str], None]] = None) -> dict:
    def report(p, msg):
        if progress:
            progress(p, msg)

    report(0.03, "Loading universe…")
    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    meta = {u["ticker"]: u for u in uni}

    report(0.08, f"Fetching fundamentals for {len(tickers)} stocks…")
    funds = fundamentals.fetch_fundamentals(
        tickers, scope=scope, use_cache=use_cache,
        progress=lambda p, m: report(0.08 + 0.42 * p, m))

    report(0.55, "Downloading price history…")
    daily = data.fetch_daily(tickers, config.LT_HISTORY_DAYS)

    report(0.7, "Scoring…")
    rows = []
    for t in tickers:
        info = funds.get(t)
        if not info:
            continue
        m = meta.get(t, {})
        ev = evaluate(m.get("symbol", t.replace(".NS", "")), m.get("name", ""),
                      m.get("industry", ""), t, info, daily.get(t),
                      news=None, apply_news=False)
        if ev:
            rows.append(ev)

    df = pd.DataFrame(rows)
    news_map = {}
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        if with_news:
            buyable = df[df["recommendation"].str.contains("BUY|ACCUMULATE")].head(news_limit)
            items = [{"symbol": r["symbol"], "name": r["name"]}
                     for _, r in buyable.iterrows()]
            report(0.85, f"Reading news for {len(items)} candidates…")
            news_map = news_mod.fetch_news_batch(items)
            for i in df.index[df["recommendation"].str.contains("BUY|ACCUMULATE")][:news_limit]:
                sym = df.at[i, "symbol"]
                n = news_map.get(sym)
                if not n:
                    continue
                info = funds.get(df.at[i, "ticker"])
                ev = evaluate(sym, df.at[i, "name"], df.at[i, "industry"],
                              df.at[i, "ticker"], info, daily.get(df.at[i, "ticker"]),
                              news=n, apply_news=True)
                if ev:
                    for k, v in ev.items():
                        df.at[i, k] = v
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))

    report(1.0, "Done")
    return {"df": df, "news": news_map}

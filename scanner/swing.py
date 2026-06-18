"""Swing / short-term model (holding ≈ 15 days to 2 months).

Purely technical (medium-term trend, relative strength, breakout structure,
volume, RSI) + news. Produces the same plan shape as the long-term view:
buy-now verdict, target, upside %, ≈ days to target, stop-loss, risk:reward.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, data, indicators, news as news_mod, universe


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _atr_value(daily, period=14):
    if len(daily) < period + 1:
        return float("nan")
    high, low, close = daily["High"], daily["Low"], daily["Close"]
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1])


def _ret(close, days):
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1) * 100


# --------------------------------------------------------------------------
# Sub-scores (0..1)
# --------------------------------------------------------------------------
def _trend_sub(price, ema20, ema50):
    parts = []
    if ema20:
        parts.append(1.0 if price > ema20 else 0.0)
    if ema50:
        parts.append(1.0 if price > ema50 else 0.0)
    if ema20 and ema50:
        parts.append(1.0 if ema20 > ema50 else 0.0)
    return float(np.mean(parts)) if parts else 0.5


def _momentum_sub(ret_1m, ret_3m, rs_3m):
    parts = []
    if ret_3m is not None:
        parts.append(_clamp(ret_3m / config.SWING_MOM_FULL))
    if ret_1m is not None:
        parts.append(_clamp(ret_1m / (config.SWING_MOM_FULL * 0.6)))
    if rs_3m is not None:
        parts.append(_clamp((rs_3m + config.SWING_RS_FULL) / (2 * config.SWING_RS_FULL)))
    return float(np.mean(parts)) if parts else 0.0


def _breakout_sub(price, high_20d, high_52w):
    parts = []
    if high_20d:
        parts.append(_clamp(1 - (high_20d - price) / (high_20d * 0.06)))  # within 6% of 20d high
    if high_52w:
        parts.append(_clamp(1 - (high_52w - price) / (high_52w * 0.15)))  # within 15% of 52w high
    return float(np.mean(parts)) if parts else 0.0


def _volume_sub(rvol):
    if rvol is None or math.isnan(rvol):
        return 0.4
    return _clamp((rvol - 0.8) / (config.SWING_VOL_FULL - 0.8))


def _rsi_sub(rsi):
    if rsi is None or math.isnan(rsi):
        return 0.5
    if config.SWING_RSI_LOW <= rsi <= config.SWING_RSI_HIGH:
        return 1.0
    if rsi < config.SWING_RSI_LOW:
        return _clamp((rsi - 35) / (config.SWING_RSI_LOW - 35))
    return _clamp(1 - (rsi - config.SWING_RSI_HIGH) /
                  (config.SWING_RSI_OVERBOUGHT - config.SWING_RSI_HIGH))


# --------------------------------------------------------------------------
# Plan: target / stop / days
# --------------------------------------------------------------------------
def _plan(price, atr_val, atr_pct, ret_1m, high_20d, high_52w):
    # expected % move = blend of ATR-multiple and recent momentum, bounded
    atr_move_pct = (config.SWING_ATR_TARGET_MULT * atr_val / price * 100) if atr_val and not math.isnan(atr_val) else 8.0
    mom_pct = (ret_1m or 0) * 0.6
    exp_pct = _clamp(max(atr_move_pct, mom_pct), config.SWING_TARGET_MIN_PCT, config.SWING_TARGET_MAX_PCT)
    target = price * (1 + exp_pct / 100)
    # if a clear resistance (20d/52w high) sits just above, prefer it as the target
    for res in [high_20d, high_52w]:
        if res and price < res <= target * 1.05:
            target = max(target, res * 1.01)
    upside_pct = (target / price - 1) * 100

    sl_pct = _clamp(config.SWING_ATR_SL_MULT * (atr_pct or 4.0),
                    config.SWING_SL_MIN_PCT, config.SWING_SL_MAX_PCT)
    sl = price * (1 - sl_pct / 100)
    downside_pct = sl_pct

    drift_per_day = max((ret_1m or 0) / 21.0, (atr_pct or 3.0) * 0.22, 0.3)
    days = int(_clamp(upside_pct / drift_per_day, config.SWING_DAYS_MIN, config.SWING_DAYS_MAX))
    return target, upside_pct, sl, downside_pct, days


def _recommendation(score, price, ema50):
    if score >= config.SWING_BUY and (not ema50 or price > ema50):
        return "🟢 BUY"
    if score >= config.SWING_WATCH:
        return "🟡 WATCH (near setup)"
    return "🔴 AVOID"


# --------------------------------------------------------------------------
# Score one stock
# --------------------------------------------------------------------------
def evaluate(symbol, name, industry, ticker, daily, nifty_ret_3m=0.0,
             news=None, apply_news=True) -> Optional[dict]:
    if daily is None or len(daily) < 40:
        return None
    close = daily["Close"].dropna()
    price = float(close.iloc[-1])
    ema20 = indicators.ema(close, 20)
    ema50 = indicators.ema(close, 50)
    rsi = indicators.wilder_rsi(close)
    atr_pct = indicators.atr_pct(daily)
    atr_val = _atr_value(daily)
    ret_1m = _ret(close, 21)
    ret_3m = _ret(close, 63)
    rs_3m = (ret_3m - nifty_ret_3m) if ret_3m is not None else None
    high_20d = float(daily["High"].iloc[-20:].max())
    high_52w = float(daily["High"].max())
    avg_vol = indicators.avg_volume(daily["Volume"], config.AVG_VOLUME_DAYS)
    vol = float(daily["Volume"].iloc[-1])
    rvol = vol / avg_vol if avg_vol and not math.isnan(avg_vol) and avg_vol > 0 else float("nan")

    subs = {
        "trend": _trend_sub(price, ema20, ema50),
        "momentum": _momentum_sub(ret_1m, ret_3m, rs_3m),
        "breakout": _breakout_sub(price, high_20d, high_52w),
        "volume": _volume_sub(rvol),
        "rsi": _rsi_sub(rsi),
    }
    w = config.SWING_WEIGHTS
    contrib = {k: round(subs[k] * w[k], 1) for k in subs}
    news_points = int((news or {}).get("points", 0)) if (apply_news and news) else 0
    contrib["news"] = news_points
    score = _clamp(sum(contrib.values()), 0, 100)

    target, upside_pct, sl, downside_pct, days = _plan(
        price, atr_val, atr_pct, ret_1m, high_20d, high_52w)
    rr = upside_pct / downside_pct if downside_pct > 0 else None

    rec = _recommendation(score, price, ema50)
    # NEWS VETO: bad news (e.g. big loss, fraud, downgrade) blocks a Buy/Watch
    news_risk = bool(apply_news and news and news.get("red_flag"))
    flag_terms = (news or {}).get("flag_terms", []) if news_risk else []
    if news_risk:
        rec = "🔴 AVOID — ⚠️ NEWS RISK"

    return {
        "symbol": symbol, "name": name, "industry": industry, "ticker": ticker,
        "price": round(price, 2), "score": round(score, 1), "contrib": contrib,
        "recommendation": rec, "news_risk": news_risk, "flag_terms": flag_terms,
        "target": round(target, 2), "upside_pct": round(upside_pct, 1),
        "stop_loss": round(sl, 2), "downside_pct": round(downside_pct, 1),
        "rr": round(rr, 2) if rr else None,
        "days_to_target": days,
        "horizon": f"≈ {days} days",
        "rsi": round(rsi, 0) if not math.isnan(rsi) else None,
        "atr_pct": round(atr_pct, 1) if atr_pct and not math.isnan(atr_pct) else None,
        "ret_1m_pct": round(ret_1m, 1) if ret_1m is not None else None,
        "ret_3m_pct": round(ret_3m, 1) if ret_3m is not None else None,
        "rs_3m_pct": round(rs_3m, 1) if rs_3m is not None else None,
        "rvol": round(rvol, 2) if not math.isnan(rvol) else None,
        "above_ema50": bool(ema50 and price > ema50),
        "dist_52w_high_pct": round((high_52w - price) / high_52w * 100, 1) if high_52w else None,
        "news_points": news_points, "news_label": (news or {}).get("label", "—"),
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def run_swing_scan(scope="nifty500", with_news=True, news_limit=40,
                   progress: Optional[Callable[[float, str], None]] = None) -> dict:
    def report(p, msg):
        if progress:
            progress(p, msg)

    report(0.03, "Loading universe…")
    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    meta = {u["ticker"]: u for u in uni}

    report(0.1, "Fetching Nifty 50 (relative strength benchmark)…")
    nifty_ret_3m = 0.0
    try:
        nidx = data.fetch_daily(["^NSEI"], config.SWING_HISTORY_DAYS).get("^NSEI")
        if nidx is not None and len(nidx) > 63:
            nifty_ret_3m = _ret(nidx["Close"].dropna(), 63) or 0.0
    except Exception:
        pass

    report(0.2, f"Downloading prices for {len(tickers)} stocks…")
    daily = data.fetch_daily(tickers, config.SWING_HISTORY_DAYS)

    report(0.65, "Scoring…")
    rows = []
    for t, df in daily.items():
        m = meta.get(t, {})
        ev = evaluate(m.get("symbol", t.replace(".NS", "")), m.get("name", ""),
                      m.get("industry", ""), t, df, nifty_ret_3m,
                      news=None, apply_news=False)
        if ev:
            rows.append(ev)

    df = pd.DataFrame(rows)
    news_map = {}
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        if with_news:
            cand = df[df["recommendation"].str.contains("BUY|WATCH")].head(news_limit)
            items = [{"symbol": r["symbol"], "name": r["name"]} for _, r in cand.iterrows()]
            report(0.85, f"Reading news for {len(items)} candidates…")
            news_map = news_mod.fetch_news_batch(items)
            for i in df.index[df["recommendation"].str.contains("BUY|WATCH")][:news_limit]:
                sym = df.at[i, "symbol"]
                n = news_map.get(sym)
                if not n:
                    continue
                ev = evaluate(sym, df.at[i, "name"], df.at[i, "industry"], df.at[i, "ticker"],
                              daily[df.at[i, "ticker"]], nifty_ret_3m, news=n, apply_news=True)
                if ev:
                    for k, v in ev.items():
                        df.at[i, k] = v
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))

    report(1.0, "Done")
    return {"df": df, "nifty_ret_3m": nifty_ret_3m, "news": news_map}

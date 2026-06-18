"""v2 scoring engine: build factor features, score them (with breakdown), classify.

Score = RVol(25) + Momentum(20) + Location(20) + SmartMoney(15) + RSI(10)
        + News(±10), then × market-regime multiplier, clamped 0-100.
Hard gates (liquidity) force AVOID regardless of score.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd

from . import config, indicators


# --------------------------------------------------------------------------
# Feature extraction (raw factor values, Excel-parity columns)
# --------------------------------------------------------------------------
def build_features(ticker: str, daily: pd.DataFrame, symbol: str, name: str,
                   industry: str = "", delivery: Optional[dict] = None,
                   intraday_vwap: float = float("nan")) -> Optional[dict]:
    if daily is None or len(daily) < 20:
        return None
    close = daily["Close"]
    cmp_ = float(close.iloc[-1])
    open_ = float(daily["Open"].iloc[-1])
    high = float(daily["High"].iloc[-1])
    low = float(daily["Low"].iloc[-1])
    volume = float(daily["Volume"].iloc[-1])
    prev_close = float(close.iloc[-2])
    prev_high = float(daily["High"].iloc[-2])

    avg_vol = indicators.avg_volume(daily["Volume"], config.AVG_VOLUME_DAYS)
    rvol = volume / avg_vol if avg_vol and not math.isnan(avg_vol) and avg_vol > 0 else float("nan")
    rsi = indicators.wilder_rsi(close)
    ema20 = indicators.ema(close, 20)
    ema50 = indicators.ema(close, 50)
    atrp = indicators.atr_pct(daily)

    vwap = intraday_vwap if not math.isnan(intraday_vwap) else indicators.pivot_vwap(high, low, cmp_)
    vwap_kind = "real" if not math.isnan(intraday_vwap) else "pivot"

    deliv = (delivery or {}).get("deliv_per", float("nan"))
    turnover_cr = (delivery or {}).get("turnover_cr", float("nan"))
    if math.isnan(turnover_cr):  # fallback estimate from price*volume
        turnover_cr = cmp_ * volume / 1e7

    return {
        "symbol": symbol, "name": name, "industry": industry, "ticker": ticker,
        "cmp": cmp_, "open": open_, "high": high, "low": low, "volume": volume,
        "prev_close": prev_close, "prev_high": prev_high,
        "price_change_pct": indicators.pct_change(cmp_, prev_close),
        "gap_pct": indicators.gap_pct(open_, prev_close),
        "avg_volume": avg_vol, "rvol": rvol,
        "rsi": rsi, "ema20": ema20, "ema50": ema50, "atr_pct": atrp,
        "vwap": vwap, "vwap_kind": vwap_kind, "above_vwap": cmp_ > vwap,
        "deliv_per": deliv, "turnover_cr": turnover_cr,
    }


# --------------------------------------------------------------------------
# Individual factor sub-scores (each returns 0..1, news is signed -1..1)
# --------------------------------------------------------------------------
def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _rvol_sub(rvol):
    if rvol is None or math.isnan(rvol):
        return 0.0
    return _clamp((rvol - config.RVOL_MIN) / (config.RVOL_HIGH - config.RVOL_MIN))


def _momentum_sub(price_change, atr_pct):
    if price_change is None or math.isnan(price_change) or price_change <= 0:
        return 0.0
    if atr_pct and not math.isnan(atr_pct) and atr_pct > 0:
        norm = price_change / atr_pct
    else:
        norm = price_change / 2.0
    return _clamp(norm / config.MOM_ATR_FULL)


def _location_sub(above_vwap, cmp_, prev_high):
    score = 0.6 if above_vwap else 0.0
    if prev_high and cmp_ > prev_high:        # breaking prior-day high
        score += 0.4
    elif prev_high and cmp_ > prev_high * 0.99:  # knocking on it
        score += 0.2
    return _clamp(score)


def _smart_money_sub(deliv, turnover_cr):
    if deliv is None or math.isnan(deliv):
        deliv_part = 0.4                       # neutral when unknown
    else:
        deliv_part = _clamp(deliv / config.DELIVERY_STRONG)
    turn_part = _clamp((turnover_cr or 0) / config.TURNOVER_STRONG_CR)
    return _clamp(0.7 * deliv_part + 0.3 * turn_part)


def _rsi_sub(rsi):
    if rsi is None or math.isnan(rsi):
        return 0.5
    if config.RSI_SWEET_LOW <= rsi <= config.RSI_SWEET_HIGH:
        return 1.0
    if rsi < config.RSI_SWEET_LOW:             # ramp up from 40 -> sweet low
        return _clamp((rsi - 40) / (config.RSI_SWEET_LOW - 40))
    # overbought decay: 1.0 at sweet_high -> 0 at OVERBOUGHT+ (exhaustion)
    return _clamp(1 - (rsi - config.RSI_SWEET_HIGH) /
                  (config.RSI_OVERBOUGHT - config.RSI_SWEET_HIGH))


# --------------------------------------------------------------------------
# Descriptive (Excel-parity) labels
# --------------------------------------------------------------------------
def _descriptors(f):
    rvol = f["rvol"] if not math.isnan(f["rvol"]) else 0
    pchg = f["price_change_pct"] if not math.isnan(f["price_change_pct"]) else 0
    strong_rsi = (not math.isnan(f["rsi"])) and f["rsi"] >= config.RSI_SWEET_LOW

    breakout = (f["above_vwap"] and strong_rsi and rvol >= config.BREAKOUT_RVOL
                and pchg >= config.BREAKOUT_PRICE_CHG)
    momentum = pchg >= config.BREAKOUT_PRICE_CHG and rvol >= config.BREAKOUT_RVOL
    institutional = (rvol >= config.INSTITUTIONAL_RVOL and pchg > 0) or \
                    ((not math.isnan(f["deliv_per"])) and f["deliv_per"] >= config.DELIVERY_STRONG and rvol >= 2)
    return {
        "breakout_status": "BREAKOUT READY" if breakout else "WAIT",
        "momentum_status": "HIGH MOMENTUM" if momentum else "LOW MOMENTUM",
        "smart_money": "INSTITUTIONAL BUYING" if institutional else "NORMAL",
        "rsi_label": "RSI STRONG" if strong_rsi else "RSI WEAK",
        "trend": "UPTREND" if (f["ema20"] > f["ema50"]) else "DOWNTREND",
        "tier": "⭐ INSTITUTIONAL" if institutional else ("🟨 MOMENTUM" if momentum else "—"),
    }


def _fake_breakout(f, breakout):
    # breakout claimed but price slipped back below VWAP, or huge gap (>6%) likely to fill
    if not breakout:
        return "VALID"
    if not f["above_vwap"]:
        return "FAKE BREAKOUT"
    if not math.isnan(f["gap_pct"]) and f["gap_pct"] > 6:
        return "⚠️ GAP RISK"
    return "VALID"


# --------------------------------------------------------------------------
# Score a single feature dict
# --------------------------------------------------------------------------
def score_features(f: dict, regime: str = "neutral",
                   news: Optional[dict] = None, apply_news: bool = True) -> dict:
    w = config.WEIGHTS
    subs = {
        "rvol": _rvol_sub(f["rvol"]),
        "momentum": _momentum_sub(f["price_change_pct"], f["atr_pct"]),
        "location": _location_sub(f["above_vwap"], f["cmp"], f["prev_high"]),
        "smart_money": _smart_money_sub(f["deliv_per"], f["turnover_cr"]),
        "rsi": _rsi_sub(f["rsi"]),
    }
    contrib = {k: round(subs[k] * w[k], 1) for k in subs}
    news_points = 0
    if apply_news and news:
        news_points = int(news.get("points", 0))
    contrib["news"] = news_points

    base = sum(contrib.values())
    mult = {"bull": config.REGIME_BULL, "bear": config.REGIME_BEAR}.get(regime, config.REGIME_NEUTRAL)
    final = _clamp(base * mult, 0, 100)

    desc = _descriptors(f)
    breakout = desc["breakout_status"] == "BREAKOUT READY"

    # hard liquidity gates
    gated, gate_reason = False, ""
    if f["cmp"] < config.GATE_MIN_PRICE:
        gated, gate_reason = True, f"price<₹{config.GATE_MIN_PRICE:.0f}"
    elif (not math.isnan(f["turnover_cr"])) and f["turnover_cr"] < config.GATE_MIN_TURNOVER_CR:
        gated, gate_reason = True, "illiquid"

    news_risk = bool(apply_news and news and news.get("red_flag"))
    if gated:
        signal = "AVOID"
    elif news_risk:                       # NEWS VETO: bad news blocks a signal
        signal = "AVOID"
    elif final >= config.SIGNAL_STRONG_BUY:
        signal = "STRONG BUY"
    elif final >= config.SIGNAL_WATCH:
        signal = "WATCH"
    else:
        signal = "AVOID"

    out = dict(f)
    out.update(desc)
    out.update({
        "score": round(final, 1),
        "base_score": round(base, 1),
        "regime_mult": mult,
        "contrib": contrib,
        "news_points": news_points,
        "news_label": (news or {}).get("label", "—"),
        "signal": signal,
        "gated": gated, "gate_reason": gate_reason,
        "news_risk": news_risk,
        "flag_terms": (news or {}).get("flag_terms", []) if news_risk else [],
        "fake_breakout": _fake_breakout(f, breakout),
    })
    return out


def to_dataframe(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df

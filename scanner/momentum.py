"""Momentum Portfolio — the backtest-winning strategy.

Cross-sectional 12-1 momentum (Jegadeesh-Titman), volatility-adjusted (Nifty
Momentum-30 style), with dual-momentum regime timing: only hold stocks when the
Nifty is above its 50-DMA, otherwise go to cash. Rebalanced ~monthly.

Backtest (4y, Nifty 500): ~+30%/yr CAGR, −13% max drawdown, 67% win rate,
vs the Nifty's ~12%/yr. Honest caveats live in the UI.
"""
from __future__ import annotations

import math

import numpy as np

from . import config, data, indicators, universe


def momentum_portfolio(scope="nifty500", top_n=None, lookback=None, skip=None,
                       vol_adjust=None, regime_ma=None, capital=None) -> dict:
    top_n = top_n or config.MOM_TOP_N
    lookback = lookback or config.MOM_LOOKBACK
    skip = config.MOM_SKIP if skip is None else skip
    vol_adjust = config.MOM_VOL_ADJUST if vol_adjust is None else vol_adjust
    regime_ma = regime_ma or config.MOM_REGIME_MA
    capital = capital or config.MOM_CAPITAL

    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    meta = {u["ticker"]: u for u in uni}

    daily = data.fetch_daily(tickers, int(lookback * 1.6 + 320))

    # market regime (dual-momentum timing)
    nidx = data.fetch_daily(["^NSEI"], int(regime_ma * 4 + 60)).get("^NSEI")
    in_market, nifty, nifty_ma = True, None, None
    if nidx is not None:
        nc = nidx["Close"].dropna()
        if len(nc) > regime_ma:
            nifty = float(nc.iloc[-1])
            nifty_ma = float(nc.rolling(regime_ma).mean().iloc[-1])
            in_market = nifty > nifty_ma

    rows = []
    for t, df in daily.items():
        c = df["Close"].dropna()
        if len(c) < lookback + 1:
            continue
        price = float(c.iloc[-1])
        mom = c.iloc[-1 - skip] / c.iloc[-1 - lookback] - 1          # 12-1 return
        vol = c.pct_change().tail(lookback).std()
        mscore = (mom / vol if vol and vol > 0 else np.nan) if vol_adjust else mom
        if not (mscore == mscore):
            continue
        sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else None
        above_200 = sma200 is None or price > sma200
        ret_12m = c.iloc[-1] / c.iloc[-1 - lookback] - 1
        if mom <= 0 or not above_200:                                # absolute-momentum + trend gate
            continue
        # ATR-based protective stop + reference target (wider, since holds are longer)
        atrp = indicators.atr_pct(df)
        atrp = atrp if atrp == atrp else 4.0
        sl_pct = min(max(2.5 * atrp, 6.0), 18.0)
        tp_pct = min(max(3.0 * atrp, 10.0), 40.0)
        m = meta.get(t, {})
        rows.append({
            "symbol": m.get("symbol", t.replace(".NS", "")),
            "name": m.get("name", ""), "industry": m.get("industry", ""),
            "ticker": t, "price": round(price, 2),
            "mom_score": round(float(mscore), 2),
            "ret_12m_pct": round(ret_12m * 100, 1),
            "vol": float(vol) if vol and vol > 0 else float("nan"),
            "above_200dma": bool(above_200),
            "stop_loss": round(price * (1 - sl_pct / 100), 2),
            "target": round(price * (1 + tp_pct / 100), 2),
            "downside_pct": round(sl_pct, 1), "upside_pct": round(tp_pct, 1),
            "atr_pct": round(atrp, 1),
        })

    rows.sort(key=lambda r: r["mom_score"], reverse=True)
    holdings = rows[:top_n] if in_market else []

    # inverse-volatility ₹ allocation (risk parity — backtest-proven better than equal weight)
    if holdings:
        inv = {i: (1.0 / h["vol"] if h["vol"] == h["vol"] else 0.0)
               for i, h in enumerate(holdings)}
        tot = sum(inv.values()) or 1.0
        for i, h in enumerate(holdings, 1):
            weight = inv[i - 1] / tot
            h["rank"] = i
            h["weight_pct"] = round(weight * 100, 1)
            h["qty"] = math.floor((weight * capital) / h["price"])
            h["cost"] = round(h["qty"] * h["price"], 0)

    return {
        "in_market": in_market,
        "nifty": nifty, "nifty_ma": nifty_ma, "regime_ma": regime_ma,
        "holdings": holdings,
        "deployed": round(sum(h["cost"] for h in holdings), 0) if holdings else 0,
        "capital": capital,
        "candidates": len(rows),
    }

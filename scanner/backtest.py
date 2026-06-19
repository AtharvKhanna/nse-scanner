"""Backtest the swing strategy on historical data (point-in-time, no look-ahead).

For each trading day it scores every stock using only data up to that day, buys the
top-scoring BUY signals, then simulates each position forward until its target,
stop-loss, or max-hold triggers. Reports win rate, returns, drawdown vs the Nifty.

Note: backtest uses the TECHNICAL swing score only (no news factor / red-flag veto —
historical news isn't available). That's an honest limitation, stated in the UI.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, data, universe


# --------------------------------------------------------------------------
# Vectorised indicators + swing score over a stock's whole history
# --------------------------------------------------------------------------
def _rsi_series(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_series(df, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr, atr / c * 100


def _cl(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def features(df: pd.DataFrame, idx_ret3m_pct: pd.Series, sl_mult=None,
             target_mult=None, sl_max=None, tgt_max=None) -> pd.DataFrame:
    """Return per-date score/stop/target/close/high/low/sma200 for one stock."""
    sl_mult = sl_mult or config.SWING_ATR_SL_MULT
    target_mult = target_mult or config.SWING_ATR_TARGET_MULT
    sl_max = sl_max or config.SWING_SL_MAX_PCT
    tgt_max = tgt_max or config.SWING_TARGET_MAX_PCT
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    rsi = _rsi_series(c)
    atr_val, atr_pct = _atr_series(df)
    ret1m = (c / c.shift(21) - 1) * 100
    ret3m = (c / c.shift(63) - 1) * 100
    rvol = v / v.rolling(20).mean().shift(1)
    high20 = h.rolling(20).max()
    high52 = h.rolling(252, min_periods=60).max()
    rs3m = ret3m - idx_ret3m_pct.reindex(c.index)

    # sub-scores (match scanner/swing.py)
    trend = ((c > ema20).astype(float) + (c > ema50).astype(float)
             + (ema20 > ema50).astype(float)) / 3
    mom = (_cl(ret3m / config.SWING_MOM_FULL)
           + _cl(ret1m / (config.SWING_MOM_FULL * 0.6))
           + _cl((rs3m + config.SWING_RS_FULL) / (2 * config.SWING_RS_FULL))) / 3
    brk = (_cl(1 - (high20 - c) / (high20 * 0.06))
           + _cl(1 - (high52 - c) / (high52 * 0.15))) / 2
    volm = _cl((rvol - 0.8) / (config.SWING_VOL_FULL - 0.8))
    rsisub = np.where(
        (rsi >= config.SWING_RSI_LOW) & (rsi <= config.SWING_RSI_HIGH), 1.0,
        np.where(rsi < config.SWING_RSI_LOW,
                 _cl((rsi - 35) / (config.SWING_RSI_LOW - 35)),
                 _cl(1 - (rsi - config.SWING_RSI_HIGH) /
                     (config.SWING_RSI_OVERBOUGHT - config.SWING_RSI_HIGH))))
    score = (trend * 25 + mom * 25 + brk * 20 + volm * 10 + pd.Series(rsisub, index=c.index) * 10)

    # ATR-based target/stop (match swing._plan, minus the resistance bump)
    atr_move = target_mult * atr_val / c * 100
    exp_pct = np.clip(np.maximum(atr_move, ret1m * 0.6),
                      config.SWING_TARGET_MIN_PCT, tgt_max)
    target = c * (1 + exp_pct / 100)
    sl_pct = np.clip(sl_mult * atr_pct, config.SWING_SL_MIN_PCT, sl_max)
    stop = c * (1 - sl_pct / 100)
    sma200 = c.rolling(200).mean()

    out = pd.DataFrame({"close": c, "high": h, "low": l, "score": score,
                        "stop": stop, "target": target, "sma200": sma200})
    return out


# --------------------------------------------------------------------------
# Event-driven simulation
# --------------------------------------------------------------------------
def run_backtest(scope="nifty500", years=2, capital=100000, max_positions=5,
                 max_hold_days=45, score_threshold=None, regime_filter=False,
                 stock_above_200dma=False, sl_mult=None, target_mult=None,
                 sl_max=None, tgt_max=None, trail_pct=None, end_offset_days=0,
                 progress: Optional[Callable[[float, str], None]] = None) -> dict:
    def report(p, m):
        if progress:
            progress(p, m)

    if score_threshold is None:
        score_threshold = config.SWING_BUY
    report(0.03, "Loading universe…")
    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    name = {u["ticker"]: u["symbol"] for u in uni}

    report(0.10, f"Downloading history for {len(tickers)} stocks…")
    # window + 200-DMA warmup + optional offset for robustness tests
    hist_days = int(years * 365 + end_offset_days + 320)
    hist = data.fetch_daily(tickers, hist_days)
    nidx = data.fetch_daily(["^NSEI"], hist_days).get("^NSEI")
    idx_close = nidx["Close"].dropna()
    idx_ret3m = (idx_close / idx_close.shift(63) - 1) * 100
    idx_ma = idx_close.rolling(50).mean()   # market-regime gate (Nifty 50-DMA)

    report(0.55, "Computing signals…")
    feats = {}
    for t, df in hist.items():
        try:
            feats[name[t]] = features(df, idx_ret3m, sl_mult, target_mult, sl_max, tgt_max)
        except Exception:
            continue
    if not feats:
        return {"error": "no data"}

    # align everything on a common date index, restricted to the backtest window
    score_mat = pd.DataFrame({s: f["score"] for s, f in feats.items()})
    close_mat = pd.DataFrame({s: f["close"] for s, f in feats.items()})
    high_mat = pd.DataFrame({s: f["high"] for s, f in feats.items()})
    low_mat = pd.DataFrame({s: f["low"] for s, f in feats.items()})
    stop_mat = pd.DataFrame({s: f["stop"] for s, f in feats.items()})
    tgt_mat = pd.DataFrame({s: f["target"] for s, f in feats.items()})
    sma_mat = pd.DataFrame({s: f["sma200"] for s, f in feats.items()})
    dates = score_mat.index.sort_values()
    end = dates[-1] - pd.Timedelta(days=int(end_offset_days))   # shift window back for robustness test
    start = end - pd.Timedelta(days=int(years * 365))
    dates = dates[(dates >= start) & (dates <= end)]

    report(0.7, "Simulating trades…")
    cash = float(capital)
    positions = []   # {stock, di, entry, qty, stop, target, entry_date}
    trades = []
    equity = []
    for di, date in enumerate(dates):
        # 1) exits
        for p in positions[:]:
            try:
                lo = low_mat.at[date, p["stock"]]
                hi = high_mat.at[date, p["stock"]]
                cl = close_mat.at[date, p["stock"]]
            except Exception:
                continue
            # trailing stop: raise the stop as the position makes new highs
            if trail_pct and hi == hi:
                p["peak"] = max(p.get("peak", p["entry"]), hi)
                p["stop"] = max(p["stop"], p["peak"] * (1 - trail_pct / 100))
            ex = reason = None
            if lo == lo and lo <= p["stop"]:
                ex, reason = p["stop"], "SL" if not trail_pct else "TRAIL"
            elif hi == hi and hi >= p["target"]:
                ex, reason = p["target"], "TP"
            elif di - p["di"] >= max_hold_days and cl == cl:
                ex, reason = cl, "TIME"
            if ex is not None:
                cash += p["qty"] * ex
                trades.append({"stock": p["stock"], "entry": round(p["entry"], 2),
                               "exit": round(ex, 2), "reason": reason,
                               "qty": p["qty"], "pnl": round((ex - p["entry"]) * p["qty"], 2),
                               "ret_pct": round((ex / p["entry"] - 1) * 100, 2),
                               "in": p["entry_date"].date().isoformat(),
                               "out": date.date().isoformat(),
                               "days": di - p["di"]})
                positions.remove(p)
        # 2) entries (optionally only when the market is in an uptrend)
        regime_ok = True
        if regime_filter:
            try:
                ic, im = idx_close.get(date), idx_ma.get(date)
                regime_ok = ic is not None and im == im and ic > im
            except Exception:
                regime_ok = True
        free = max_positions - len(positions)
        if free > 0 and cash > 0 and regime_ok:
            row = score_mat.loc[date].dropna()
            held = {p["stock"] for p in positions}
            cand = row[row >= score_threshold].drop(index=held, errors="ignore")
            for stock in cand.sort_values(ascending=False).index[:free]:
                price = close_mat.at[date, stock]
                stp = stop_mat.at[date, stock]
                tgt = tgt_mat.at[date, stock]
                if not (price == price and stp == stp and tgt == tgt) or price <= 0:
                    continue
                if stock_above_200dma:                      # stock-level trend filter
                    sma = sma_mat.at[date, stock]
                    if not (sma == sma) or price <= sma:
                        continue
                budget = cash / (max_positions - len(positions))
                qty = math.floor(budget / price)
                if qty < 1 or qty * price > cash:
                    continue
                cash -= qty * price
                positions.append({"stock": stock, "di": di, "entry": price, "qty": qty,
                                  "stop": stp, "target": tgt, "entry_date": date})
        # 3) mark-to-market equity
        mtm = 0.0
        for p in positions:
            try:
                px = close_mat.at[date, p["stock"]]
                mtm += p["qty"] * (px if px == px else p["entry"])
            except Exception:
                mtm += p["qty"] * p["entry"]
        equity.append((date, cash + mtm))

    report(0.95, "Computing metrics…")
    eq = pd.Series(dict(equity)).sort_index()
    bench = idx_close.reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * capital if len(bench) and bench.iloc[0] else bench
    metrics = _metrics(eq, trades, capital, idx_close.reindex(eq.index).dropna())
    report(1.0, "Done")
    return {"equity": eq, "benchmark": bench, "trades": pd.DataFrame(trades),
            "metrics": metrics}


def run_momentum_backtest(scope="nifty500", years=2, capital=100000, top_n=15,
                          lookback=252, skip=21, rebal_days=21, vol_adjust=True,
                          regime_filter=True, above_200dma=True, regime_ma=200,
                          end_offset_days=0,
                          progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Cross-sectional momentum (Jegadeesh-Titman / Nifty Momentum-30 style).

    Each rebalance, hold the top-N stocks by their (lookback - skip) return
    (optionally volatility-adjusted), equal weight; go to cash when the market is
    below its 200-DMA. Rebalanced every `rebal_days`.
    """
    def report(p, m):
        if progress:
            progress(p, m)

    report(0.05, "Loading universe…")
    uni = universe.load_universe(scope)
    tickers = [u["ticker"] for u in uni]
    name = {u["ticker"]: u["symbol"] for u in uni}

    report(0.1, "Downloading history…")
    hist_days = int(years * 365 + end_offset_days + 520)
    hist = data.fetch_daily(tickers, hist_days)
    nidx = data.fetch_daily(["^NSEI"], hist_days).get("^NSEI")
    idx_close = nidx["Close"].dropna()
    idx_ma = idx_close.rolling(regime_ma).mean()

    report(0.5, "Computing momentum…")
    close = pd.DataFrame({name[t]: df["Close"] for t, df in hist.items()})
    mom = close.shift(skip) / close.shift(lookback) - 1          # 12-1 momentum
    if vol_adjust:
        vol = close.pct_change().rolling(lookback).std()
        mom = mom / vol.replace(0, np.nan)
    sma200 = close.rolling(200).mean()

    dates = close.index.sort_values()
    end = dates[-1] - pd.Timedelta(days=int(end_offset_days))
    start = end - pd.Timedelta(days=int(years * 365))
    dates = dates[(dates >= start) & (dates <= end)]

    report(0.7, "Simulating rebalances…")
    equity = capital
    shares = {}
    curve = []
    rets = []          # per-rebalance basket returns for win-rate
    last_rebal_val = capital
    for di, date in enumerate(dates):
        # mark-to-market
        val = sum(s * close.at[date, st] for st, s in shares.items()
                  if close.at[date, st] == close.at[date, st])
        cash = equity - sum(s * 0 for s in shares.values())  # placeholder
        port = (val if shares else 0.0)
        # equity tracked as cash-when-flat + market value
        if not shares:
            port = equity
        else:
            port = val
        curve.append((date, port))

        if di % rebal_days != 0:
            continue
        # rebalance: realise basket return
        if shares:
            rets.append(port / last_rebal_val - 1)
            equity = port
        last_rebal_val = equity
        shares = {}
        # regime: only invest when market above its 200-DMA
        regime_ok = True
        if regime_filter:
            im = idx_ma.get(date)
            ic = idx_close.get(date)
            regime_ok = ic is not None and im == im and ic > im
        if not regime_ok:
            continue   # stay in cash
        row = mom.loc[date].dropna()
        if above_200dma:
            ok = close.loc[date] > sma200.loc[date]
            row = row[ok.reindex(row.index).fillna(False)]
        row = row[row > 0]                     # absolute momentum: positive only
        picks = row.sort_values(ascending=False).index[:top_n]
        if len(picks) == 0:
            continue
        alloc = equity / len(picks)
        for st in picks:
            px = close.at[date, st]
            if px == px and px > 0:
                shares[st] = alloc / px

    report(0.95, "Metrics…")
    eq = pd.Series(dict(curve)).sort_index()
    bench = idx_close.reindex(eq.index).ffill()
    bench = bench / bench.iloc[0] * capital if len(bench) and bench.iloc[0] else bench
    rets = pd.Series(rets)
    days = (eq.index[-1] - eq.index[0]).days or 1
    roll = eq.cummax()
    metrics = {
        "total_return_pct": round((eq.iloc[-1] / capital - 1) * 100, 1),
        "cagr_pct": round(((eq.iloc[-1] / capital) ** (365 / days) - 1) * 100, 1),
        "max_drawdown_pct": round(((eq - roll) / roll).min() * 100, 1),
        "num_trades": int((rets != 0).sum()),
        "win_rate_pct": round((rets > 0).mean() * 100, 1) if len(rets) else None,
        "avg_win_pct": round(rets[rets > 0].mean() * 100, 1) if (rets > 0).any() else None,
        "avg_loss_pct": round(rets[rets <= 0].mean() * 100, 1) if (rets <= 0).any() else None,
        "profit_factor": round(rets[rets > 0].sum() / -rets[rets <= 0].sum(), 2)
        if (rets <= 0).any() and rets[rets <= 0].sum() != 0 else None,
        "nifty_return_pct": round((bench.iloc[-1] / capital - 1) * 100, 1) if len(bench) else None,
        "final_value": round(eq.iloc[-1], 0),
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
    }
    report(1.0, "Done")
    return {"equity": eq, "benchmark": bench, "metrics": metrics}


def _metrics(eq, trades, capital, idx_close):
    if len(eq) < 2:
        return {}
    total_ret = (eq.iloc[-1] / capital - 1) * 100
    days = (eq.index[-1] - eq.index[0]).days or 1
    cagr = ((eq.iloc[-1] / capital) ** (365 / days) - 1) * 100
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min() * 100
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wins = tdf[tdf["pnl"] > 0] if n else tdf
    losses = tdf[tdf["pnl"] <= 0] if n else tdf
    win_rate = len(wins) / n * 100 if n else None
    gross_win = wins["pnl"].sum() if n else 0
    gross_loss = -losses["pnl"].sum() if n else 0
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    nifty_ret = (idx_close.iloc[-1] / idx_close.iloc[0] - 1) * 100 if len(idx_close) > 1 else None
    return {
        "total_return_pct": round(total_ret, 1),
        "cagr_pct": round(cagr, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "num_trades": n,
        "win_rate_pct": round(win_rate, 1) if win_rate is not None else None,
        "avg_win_pct": round(wins["ret_pct"].mean(), 1) if len(wins) else None,
        "avg_loss_pct": round(losses["ret_pct"].mean(), 1) if len(losses) else None,
        "profit_factor": round(pf, 2) if pf else None,
        "avg_hold_days": round(tdf["days"].mean(), 0) if n else None,
        "nifty_return_pct": round(nifty_ret, 1) if nifty_ret is not None else None,
        "final_value": round(eq.iloc[-1], 0),
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
    }

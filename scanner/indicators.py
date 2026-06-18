"""Technical indicators computed from price/volume history (pandas/numpy only).

Each function takes a pandas Series/DataFrame of *daily* OHLCV (oldest -> newest)
unless noted. Intraday VWAP is computed separately from 5-min bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI on the latest bar. Returns NaN if insufficient data."""
    close = close.dropna()
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    # all-gains edge case -> RSI 100
    if pd.isna(val) and avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0:
        return 100.0
    return float(val)


def ema(series: pd.Series, span: int) -> float:
    series = series.dropna()
    if len(series) < span:
        return float("nan")
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range as a % of last close. df needs High/Low/Close cols."""
    if len(df) < period + 1:
        return float("nan")
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    last_close = close.iloc[-1]
    if not last_close:
        return float("nan")
    return float(atr / last_close * 100)


def avg_volume(volume: pd.Series, days: int = 20) -> float:
    volume = volume.dropna()
    if len(volume) < 2:
        return float("nan")
    # exclude today's (last) bar from the average baseline
    base = volume.iloc[-(days + 1):-1] if len(volume) > days else volume.iloc[:-1]
    if len(base) == 0:
        return float("nan")
    return float(base.mean())


def pivot_vwap(high: float, low: float, cmp_: float) -> float:
    """The Excel's pivot proxy = (H+L+C)/3. Kept for reference/compat."""
    return (high + low + cmp_) / 3.0


def intraday_vwap(bars: pd.DataFrame) -> float:
    """Real VWAP from intraday bars (cols High/Low/Close/Volume). NaN if empty."""
    if bars is None or len(bars) == 0 or bars["Volume"].sum() == 0:
        return float("nan")
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    vol = bars["Volume"].fillna(0)
    denom = vol.sum()
    if denom == 0:
        return float("nan")
    return float((typical * vol).sum() / denom)


def pct_change(cmp_: float, prev_close: float) -> float:
    if not prev_close or pd.isna(prev_close):
        return float("nan")
    return (cmp_ - prev_close) / prev_close * 100.0


def gap_pct(open_: float, prev_close: float) -> float:
    if not prev_close or pd.isna(prev_close) or pd.isna(open_):
        return float("nan")
    return (open_ - prev_close) / prev_close * 100.0

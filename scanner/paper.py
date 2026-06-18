"""Paper trading: a virtual ₹10,000 portfolio driven by the swing signals.

Buys the top Nifty-500 swing BUY pick(s), then auto-closes a position when its
stop-loss or target is hit (checked against daily High/Low since entry). Tracks
realised + open P&L. State persists to a local JSON file.

Note: on Streamlit Community Cloud the filesystem is ephemeral, so reliable
long-term tracking is best done by running the app locally.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from typing import Optional

import pandas as pd

from . import config, data

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(HERE, "data", "paper_state.json")


def _today() -> str:
    return dt.date.today().isoformat()


def load_state() -> Optional[dict]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reset() -> None:
    if os.path.exists(STATE_PATH):
        os.remove(STATE_PATH)


def init_state(capital: float = config.PAPER_CAPITAL,
               positions: int = config.PAPER_POSITIONS) -> dict:
    state = {
        "start_date": _today(),
        "capital": float(capital),
        "cash": float(capital),
        "max_positions": int(positions),
        "positions": [],   # open: symbol, qty, entry, entry_date, stop, target, last
        "closed": [],      # done: + exit, exit_date, reason, pnl
    }
    save_state(state)
    return state


def _eligible(swing_df: pd.DataFrame) -> pd.DataFrame:
    need = {"recommendation", "price", "stop_loss", "target"}
    if swing_df is None or len(swing_df) == 0 or not need.issubset(swing_df.columns):
        return pd.DataFrame(columns=list(need))
    d = swing_df[swing_df["recommendation"].astype(str).str.contains("BUY")].copy()
    if "news_risk" in d.columns:
        d = d[~d["news_risk"].fillna(False)]
    d = d[d["price"] <= config.PAPER_MAX_PRICE]
    return d


def update(state: dict, swing_df: pd.DataFrame) -> dict:
    """Check exits (SL/TP) on open positions, then open new ones if capacity."""
    # ---- 1) exits ----
    tickers = [p["symbol"] + ".NS" for p in state["positions"]]
    hist = data.fetch_daily(tickers, 200) if tickers else {}
    still_open = []
    for p in state["positions"]:
        df = hist.get(p["symbol"] + ".NS")
        exit_price = reason = exit_date = None
        if df is not None and len(df):
            entry_d = dt.date.fromisoformat(p["entry_date"])
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, "date") else None
                if d is None or d <= entry_d:
                    continue
                low, high = float(row["Low"]), float(row["High"])
                if low <= p["stop"]:                       # stop-loss first (conservative)
                    exit_price, reason, exit_date = p["stop"], "🛑 SL", d.isoformat()
                    break
                if high >= p["target"]:
                    exit_price, reason, exit_date = p["target"], "🎯 TP", d.isoformat()
                    break
            p["last"] = round(float(df["Close"].iloc[-1]), 2)
        if exit_price is not None:
            pnl = (exit_price - p["entry"]) * p["qty"]
            state["cash"] += exit_price * p["qty"]
            state["closed"].append({**p, "exit": round(exit_price, 2),
                                    "exit_date": exit_date, "reason": reason,
                                    "pnl": round(pnl, 2)})
        else:
            still_open.append(p)
    state["positions"] = still_open

    # ---- 2) open new positions ----
    held = {p["symbol"] for p in state["positions"]}
    closed_today = {c["symbol"] for c in state["closed"] if c.get("exit_date") == _today()}
    for _, r in _eligible(swing_df).iterrows():
        if len(state["positions"]) >= state["max_positions"]:
            break
        sym = r["symbol"]
        if sym in held or sym in closed_today:
            continue
        price = float(r["price"])
        slots_left = state["max_positions"] - len(state["positions"])
        budget = state["cash"] if slots_left == 1 else state["cash"] / slots_left
        qty = math.floor(budget / price)
        if qty < 1 or qty * price > state["cash"]:
            continue
        state["cash"] -= qty * price
        state["positions"].append({
            "symbol": sym, "qty": qty, "entry": round(price, 2),
            "entry_date": _today(), "stop": round(float(r["stop_loss"]), 2),
            "target": round(float(r["target"]), 2), "last": round(price, 2),
        })
        held.add(sym)

    save_state(state)
    return state


def summary(state: dict) -> dict:
    open_val = sum(p["qty"] * p.get("last", p["entry"]) for p in state["positions"])
    total_val = state["cash"] + open_val
    realised = sum(c["pnl"] for c in state["closed"])
    unrealised = sum((p.get("last", p["entry"]) - p["entry"]) * p["qty"]
                     for p in state["positions"])
    wins = sum(1 for c in state["closed"] if c["pnl"] > 0)
    n = len(state["closed"])
    return {
        "total_value": round(total_val, 2),
        "cash": round(state["cash"], 2),
        "invested": round(open_val, 2),
        "realised": round(realised, 2),
        "unrealised": round(unrealised, 2),
        "total_return_pct": round((total_val / state["capital"] - 1) * 100, 2),
        "trades": n,
        "win_rate": round(wins / n * 100, 0) if n else None,
        "start_date": state["start_date"],
    }

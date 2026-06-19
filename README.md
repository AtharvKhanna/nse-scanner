# 📈 NSE Stock Scanner — Momentum, Swing, Long-Term & more

A website built on **live data** (Yahoo Finance + NSE) with a **News sentiment** factor.
Originated from `NSE_INTRADAY_SCANNER_ENHANCED.xlsx`, then extended with a research-driven,
**backtested** momentum strategy.

## 🚀 Momentum Portfolio (default — the backtest-winning strategy)
Cross-sectional **12-1 momentum** (Jegadeesh-Titman / Nifty Momentum-30 style), **volatility-
adjusted**, **inverse-volatility weighted**, holding the **top-15** Nifty-500 names and going to
**cash when the Nifty is below its 50-DMA** (dual-momentum timing). Rebalance ~monthly.

- **Backtest (4y, net of ~0.3% costs):** ~**27%/yr CAGR, −13% max drawdown** vs the Nifty's ~12%/yr.
- Chosen after testing residual momentum, low-vol tilts, sector caps and concentration — which did
  **not** improve it (rejected, not curve-fit). Tunable in `scanner/config.py` (`MOM_*`).
- **Backtest tab** lets you verify it (with cost + weighting + residual-momentum toggles).
- ⚠️ Backtest ≠ future; momentum has occasional crash years. For small capital, a **Nifty 200
  Momentum 30 ETF/index fund** is the cleanest way to run this. Not investment advice.

## 📈 Long-Term Investing mode (default)
For each Nifty 500 stock it tells you:
- **Current price**, a **verdict** (🟢 Strong Buy / Buy · 🟡 Accumulate on dips · ⚪ Hold · 🔴 Avoid),
- a **target price** (analyst consensus, sanity-checked), **upside %**,
- **≈ time to target** (months), a **stop-loss** + downside %, and **risk:reward**,
- the *why*: a score breakdown plus full fundamentals (P/E, PEG, ROE, debt, growth, 200-DMA…).

**Long-term score (0–100):** `Analyst 25 + Quality 25 + Valuation 20 + Trend 20 + News ±10`.
Verdict cut-offs: ≥72 Strong Buy · ≥58 Buy · ≥48 Accumulate · ≥38 Hold · else Avoid.

## 📅 Swing / Short-Term mode (≈15 days to 2 months)
A purely-technical medium-term model. For each stock: **current price**, verdict
(🟢 Buy / 🟡 Watch / 🔴 Avoid), an **ATR-based target** + upside %, **≈ days to target**
(capped ~2 months), a tight **stop-loss** + risk %, and **risk:reward** — plus 1-mo/3-mo
returns, relative strength vs Nifty, RSI and RVol.

**Swing score (0–100):** `Trend 25 + Momentum/RelStrength 25 + Breakout structure 20 +
Volume 10 + RSI band 10 + News ±10`. Verdict: ≥60 Buy · 45–59 Watch · else Avoid.
Targets/stops are ATR-multiples (no analyst targets — too far out for a swing).

## ⚡ Intraday Momentum mode
The original breakout scanner: Relative Volume, real VWAP, ATR-normalised momentum,
Delivery %, RSI band, market-regime multiplier, liquidity gates.
`RVol 25 + Momentum 20 + Location 20 + Smart Money 15 + RSI 10 + News ±10`, ≥75 → 🔥 Strong Buy.

All weights/thresholds live in [`scanner/config.py`](scanner/config.py).

## Setup
```bash
pip install -r requirements.txt
python -m scanner.universe      # one-time: build data/nifty500.csv & symbols.csv
streamlit run app.py            # opens http://localhost:8501
```

## Using it
1. Click **🔄 Refresh** in the morning to pull fresh prices, delivery % and news.
2. Work the **🔥 Intraday Watchlist** tab; confirm each setup on a live chart per the
   **📋 Daily SOP** tab.
3. Open **🔎 Stock Detail** to see the factor breakdown, price chart, and news.

## Notes
- Yahoo data is ~15-min delayed — fine for prep and the next-day watchlist. For true
  real-time ticks, a Zerodha Kite adapter can be dropped into `scanner/data.py` later.
- **This is a research/screening tool, not investment advice.** Always confirm before trading.

"""Central configuration: v2 factor weights, thresholds and gates.

All scoring knobs live here so the model can be tuned in one place.
The v2 model is a continuous 0-100 score = sum of factor contributions,
then multiplied by a market-regime factor. Hard "gates" remove stocks
that should never be suggested regardless of score.
"""

# --------------------------------------------------------------------------
# Factor weights (max points each factor can contribute to the base score)
# RVol 25 + Momentum 20 + Location 20 + SmartMoney 15 + RSI 10 + News 10 = 100
# --------------------------------------------------------------------------
WEIGHTS = {
    "rvol": 25,        # relative volume surge  (strongest intraday edge)
    "momentum": 20,    # ATR-normalised price change
    "location": 20,    # real VWAP position + breakout above prior-day high
    "smart_money": 15, # delivery % + turnover (real accumulation)
    "rsi": 10,         # RSI quality band (reward strength, punish exhaustion)
    "news": 10,        # news catalyst (signed: -10..+10)
}

# --------------------------------------------------------------------------
# Factor thresholds
# --------------------------------------------------------------------------
# Relative volume: scales 0..1 between these bounds (>=HIGH gets full marks)
RVOL_MIN = 1.0
RVOL_HIGH = 5.0

# Momentum: ATR-normalised move (price_change% / atr%). Full marks at >= this.
MOM_ATR_FULL = 1.5

# RSI quality band: peak reward in the sweet spot, penalty when overbought
RSI_SWEET_LOW = 58
RSI_SWEET_HIGH = 72
RSI_OVERBOUGHT = 80   # above this, exhaustion -> score decays toward 0

# Smart money
DELIVERY_STRONG = 55.0   # delivery % considered strong accumulation
TURNOVER_STRONG_CR = 25  # ₹ crore turnover considered institutional-grade

# Market regime multiplier (applied to base score)
REGIME_BULL = 1.10
REGIME_NEUTRAL = 1.00
REGIME_BEAR = 0.80

# --------------------------------------------------------------------------
# Hard gates (a stock failing any of these is marked AVOID/filtered)
# --------------------------------------------------------------------------
GATE_MIN_PRICE = 30.0          # ₹ - avoid penny stocks
GATE_MIN_TURNOVER_CR = 2.0     # ₹ crore - avoid illiquid names

# --------------------------------------------------------------------------
# Signal thresholds (on the final 0-100 score)
# --------------------------------------------------------------------------
SIGNAL_STRONG_BUY = 75
SIGNAL_WATCH = 45      # 45-74 -> WATCH (tomorrow), <45 -> AVOID

# Breakout / momentum status thresholds (for the descriptive columns)
BREAKOUT_RVOL = 2.0
BREAKOUT_PRICE_CHG = 3.0
INSTITUTIONAL_RVOL = 5.0

# ==========================================================================
# LONG-TERM INVESTING MODEL
# ==========================================================================
# Score = Analyst 25 + Quality 25 + Valuation 20 + Trend 20 + News ±10
LT_WEIGHTS = {
    "analyst": 25,    # consensus target upside + rating + coverage
    "quality": 25,    # ROE, margins, low debt, growth
    "valuation": 20,  # PEG / forward PE / P/B (cheaper = better)
    "trend": 20,      # above 200-DMA, golden cross, 12-mo return, 52w position
    "news": 10,       # news catalyst (signed)
}

# Analyst upside that earns full marks (e.g. +30% to mean target)
LT_UPSIDE_FULL = 0.30
# Quality thresholds (values at/above which the sub-factor is "good")
LT_ROE_GOOD = 0.15
LT_MARGIN_GOOD = 0.10
LT_GROWTH_GOOD = 0.12
LT_DE_GOOD = 50.0       # debt/equity at/below this is good; >150 is poor
LT_DE_POOR = 150.0
# Valuation thresholds
LT_PEG_CHEAP = 1.0
LT_PEG_EXPENSIVE = 2.5
LT_PB_CHEAP = 1.5
LT_PB_EXPENSIVE = 8.0

# Recommendation cut-offs (final 0-100 LT score)
LT_STRONG_BUY = 72
LT_BUY = 58
LT_ACCUMULATE = 48
LT_HOLD = 38            # below this -> AVOID

# Stop-loss / target sizing
LT_MAX_RISK = 0.15          # max stop distance below entry (15%)
LT_MIN_RISK = 0.07          # don't set a stop tighter than 7% for a long-term hold
LT_TARGET_MIN_MULT = 0.7    # sanity bounds for an analyst target vs price
LT_TARGET_MAX_MULT = 3.0
LT_SPEED_FLOOR = 15.0       # min assumed annual % move when projecting days-to-target
LT_DAYS_MIN = 90
LT_DAYS_MAX = 1095
LT_HISTORY_DAYS = 400       # ~1y+ of daily bars for 200-DMA and 12-mo return

# ==========================================================================
# SWING / SHORT-TERM MODEL  (holding ≈ 15 days to 2 months)
# ==========================================================================
# Score = Trend 25 + Momentum/RS 25 + Breakout 20 + Volume 10 + RSI 10 + News ±10
SWING_WEIGHTS = {
    "trend": 25,       # above 20 & 50 EMA, 20>50 (medium uptrend)
    "momentum": 25,    # 1-mo & 3-mo return + relative strength vs Nifty
    "breakout": 20,    # proximity to 20-day / 52-wk high (structure)
    "volume": 10,      # volume expansion confirming the move
    "rsi": 10,         # healthy momentum band (50-70), punish >75
    "news": 10,        # catalyst (signed)
}

SWING_MOM_FULL = 18.0      # 3-mo % return earning full momentum marks
SWING_RS_FULL = 8.0        # outperformance vs Nifty (3-mo %) for full RS credit
SWING_RSI_LOW = 50
SWING_RSI_HIGH = 70
SWING_RSI_OVERBOUGHT = 78
SWING_VOL_FULL = 1.5       # RVol that earns full volume-confirmation marks

# Target / stop sizing (ATR-based, tighter than long-term)
SWING_ATR_TARGET_MULT = 3.0   # target ≈ entry + N×ATR
SWING_ATR_SL_MULT = 1.6       # stop  ≈ entry − N×ATR
SWING_TARGET_MIN_PCT = 4.0
SWING_TARGET_MAX_PCT = 30.0
SWING_SL_MIN_PCT = 3.0
SWING_SL_MAX_PCT = 9.0
SWING_DAYS_MIN = 10
SWING_DAYS_MAX = 75           # ~2 months cap
SWING_HISTORY_DAYS = 200      # ~9 months daily bars (3-mo return + 50-EMA + structure)

# Recommendation cut-offs (final 0-100 swing score)
SWING_BUY = 60
SWING_WATCH = 45              # 45-59 -> WATCH (near setup), <45 -> AVOID

# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------
HISTORY_DAYS = 90        # daily bars pulled for RSI/EMA/ATR/AvgVol
AVG_VOLUME_DAYS = 20
CACHE_TTL_SECONDS = 900  # 15 min (intraday mode only)
DAILY_TTL = 86400        # 24h — swing & long-term refresh once per day
NEWS_CACHE_TTL_SECONDS = 3600 * 3
NEWS_MAX_HEADLINES = 6
NEWS_RECENCY_DAYS = 3    # headlines older than this are down-weighted

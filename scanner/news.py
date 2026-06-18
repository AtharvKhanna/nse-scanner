"""News factor: Google News RSS headlines + VADER sentiment (recency-weighted).

Returns a signed sentiment in [-1, 1] plus the headlines for display. Fetch only
for the shortlist (Strong Buy / Watch candidates) to keep the scan fast.
"""
from __future__ import annotations

import datetime as dt
import time
import urllib.parse
import urllib.request
from typing import Dict, List

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config

_ANALYZER = SentimentIntensityAnalyzer()

# Domain finance terms VADER doesn't know well -> nudge lexicon
_FINANCE_LEXICON = {
    "surge": 2.5, "soar": 3.0, "rally": 2.3, "jumps": 2.2, "jump": 2.0,
    "upgrade": 2.5, "outperform": 2.2, "beat": 2.0, "beats": 2.2, "record": 1.8,
    "bonus": 1.5, "buyback": 1.8, "wins": 2.0, "order": 1.2, "profit": 1.5,
    "plunge": -3.0, "crash": -3.2, "slump": -2.5, "tumble": -2.6, "downgrade": -2.6,
    "underperform": -2.2, "miss": -2.0, "misses": -2.2, "fraud": -3.5, "probe": -2.4,
    "raid": -2.5, "penalty": -2.2, "fine": -1.8, "default": -2.8, "loss": -1.8,
    "ban": -2.5, "resign": -1.8, "scam": -3.4, "lawsuit": -2.0, "downside": -1.5,
}
_ANALYZER.lexicon.update(_FINANCE_LEXICON)

_GNEWS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


def _clean_company(name: str) -> str:
    for suffix in [" Limited", " Ltd.", " Ltd", " Corporation", " Corp.", " Company"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def _query(symbol: str, name: str) -> str:
    base = _clean_company(name) if name else symbol
    return urllib.parse.quote(f'"{base}" stock')


def fetch_news(symbol: str, name: str = "",
               max_items: int = config.NEWS_MAX_HEADLINES) -> Dict:
    """Return {'score','label','points','headlines':[{title,link,published,sent}]}."""
    url = _GNEWS.format(q=_query(symbol, name))
    headlines: List[dict] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=15).read()
        feed = feedparser.parse(raw)
        now = dt.datetime.now(dt.timezone.utc)
        for entry in feed.entries[: max_items * 2]:
            title = entry.get("title", "")
            if not title:
                continue
            published = None
            age_days = 0.0
            if getattr(entry, "published_parsed", None):
                published = dt.datetime(*entry.published_parsed[:6],
                                        tzinfo=dt.timezone.utc)
                age_days = max((now - published).total_seconds() / 86400.0, 0.0)
            sent = _ANALYZER.polarity_scores(title)["compound"]
            headlines.append({
                "title": title,
                "link": entry.get("link", ""),
                "published": published.strftime("%d %b %H:%M") if published else "",
                "age_days": age_days,
                "sent": sent,
            })
            if len(headlines) >= max_items:
                break
    except Exception:
        pass

    score = _aggregate(headlines)
    return {
        "score": score,
        "label": _label(score),
        "points": round(score * config.WEIGHTS["news"]),
        "headlines": headlines,
    }


def _aggregate(headlines: List[dict]) -> float:
    """Recency-weighted average of headline compounds -> [-1, 1]."""
    if not headlines:
        return 0.0
    num = den = 0.0
    for h in headlines:
        # linear decay over NEWS_RECENCY_DAYS; older items keep a small floor weight
        w = max(1.0 - h["age_days"] / config.NEWS_RECENCY_DAYS, 0.25)
        num += h["sent"] * w
        den += w
    return num / den if den else 0.0


def _label(score: float) -> str:
    if score >= 0.15:
        return "📈 Positive"
    if score <= -0.15:
        return "📉 Negative"
    return "⚪ Neutral"


def fetch_news_batch(items: List[dict], pause: float = 0.0) -> Dict[str, Dict]:
    """items: [{'symbol','name'}]. Returns {symbol: news_dict}. Sequential + polite."""
    out = {}
    for it in items:
        out[it["symbol"]] = fetch_news(it["symbol"], it.get("name", ""))
        if pause:
            time.sleep(pause)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_news("OLAELEC", "Ola Electric Mobility Limited"),
                     indent=2, default=str)[:1500])

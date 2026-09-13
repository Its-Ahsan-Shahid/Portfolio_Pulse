"""
preprocessing.py
=================
Step 1 of Portfolio Pulse: pulls REAL news for a ticker from Alpha Vantage
and turns it into a clean pandas DataFrame, ready for embeddings (file 2)
and the bull/bear/moderator agents (file 3).

Kept deliberately simple:
  - Only ONE external API here (Alpha Vantage). No Finnhub, no spaCy,
    no manual CSV downloads.
  - "Relevance" = Alpha Vantage's own relevance score + a simple word match
    on the company name/ticker. No NER, no ML model needed for this step.
  - "Financial tone" = the Loughran-McDonald finance dictionary via the
    `pysentiment2` package (`pip install pysentiment2`) -- this has the
    dictionary BUILT IN, so there is no CSV to download or upload anywhere.
  - Alpha Vantage already gives you a sentiment score per article
    (`overall_sentiment_score`) -- we use that directly instead of running
    a second, heavier sentiment model.
  - Recency weighting per your plan: articles from the last `priority_days`
    (default 5) are tagged "priority_recent" (weight 2.0), the rest out to
    `days_back` (default 25) are tagged "context" (weight 1.0).

--------------------------------------------------------------------------
WHERE TO PUT YOUR API KEY
--------------------------------------------------------------------------
Scroll down to the block that says "PUT YOUR API KEY HERE" (right after the
imports, a few lines below). Two ways to use it:

  1. Quick testing (Colab or your own laptop): paste your key directly
     between the quotes there. Nothing else to configure.
  2. Before you push this to GitHub: delete whatever you pasted (leave it
     as "") and instead set it as an environment variable named
     ALPHAVANTAGE_API_KEY (e.g. `export ALPHAVANTAGE_API_KEY=...` in your
     terminal, or a Colab "secret"). The code checks the environment
     variable first automatically -- if it finds one, it uses that and
     ignores the blank string below. This is what keeps the real key out
     of your GitHub history.

Get a free key at: https://www.alphavantage.co/support/#api-key
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from getpass import getpass

import requests
import pandas as pd
import pysentiment2 as ps

# ============================================================================
# PUT YOUR API KEY HERE  (see the note at the top of this file)
# ============================================================================
ALPHAVANTAGE_API_KEY = ""   # <-- paste your key between the quotes for quick testing
# ============================================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
NEWS_CACHE_DIR = DATA_DIR / "av_news_cache"
NEWS_CACHE_DIR.mkdir(exist_ok=True)

_LM = ps.LM()  # Loughran-McDonald dictionary, bundled with pysentiment2 -- no CSV needed


def get_api_key() -> str:
    """
    Resolves the Alpha Vantage key: env var first (safe for GitHub), then
    the ALPHAVANTAGE_API_KEY variable pasted above (handy for quick local/
    Colab testing), then finally a hidden prompt as a last resort.
    """
    return (
        os.getenv("ALPHAVANTAGE_API_KEY")
        or ALPHAVANTAGE_API_KEY
        or getpass("Enter your Alpha Vantage API key: ")
    ).strip()


# ---------------------------------------------------------------------------
# Alpha Vantage's own sentiment buckets (for reference / your eval step later)
# More negative = more bearish, more positive = more bullish.
# ---------------------------------------------------------------------------
def av_sentiment_label(score: float) -> str:
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.35:
        return "Somewhat-Bullish"
    return "Bullish"


def _cache_path(ticker: str, days_back: int) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return NEWS_CACHE_DIR / f"{ticker.upper()}_{days_back}d_{today}.json"


def fetch_news(ticker: str, days_back: int = 25, limit: int = 200, use_cache: bool = True) -> list:
    """
    Calls Alpha Vantage's NEWS_SENTIMENT endpoint for real.

    Note: the free tier is 25 requests PER DAY (not per minute). This
    function caches the raw response to disk per (ticker, day), so you can
    re-run your pipeline many times on the same ticker in one day without
    spending another request.
    """
    cache_file = _cache_path(ticker, days_back)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    time_from = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%dT%H%M")
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "time_from": time_from,
        "sort": "LATEST",
        "limit": limit,
        "apikey": get_api_key(),
    }
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    # Alpha Vantage returns HTTP 200 even when rate-limited or the key is bad
    # -- the error shows up as a "Note"/"Information" key instead of an HTTP
    # error code, so check for that explicitly.
    if "Note" in payload or "Information" in payload:
        raise RuntimeError(f"Alpha Vantage returned no data (likely rate-limited or bad key): {payload}")

    articles = payload.get("feed", [])

    # De-duplicate: the same story is often syndicated to several outlets.
    seen, deduped = set(), []
    for art in articles:
        url = art.get("url")
        if url and url in seen:
            continue
        seen.add(url)
        deduped.append(art)

    if use_cache:
        cache_file.write_text(json.dumps(deduped))
    return deduped


def _is_relevant(article: dict, ticker: str, company_name: str) -> tuple:
    """Simple relevance check: Alpha Vantage's own score OR a plain name/ticker match."""
    av_score = 0.0
    for ts in article.get("ticker_sentiment", []):
        if ts.get("ticker", "").upper() == ticker.upper():
            try:
                av_score = float(ts.get("relevance_score", 0.0))
            except (TypeError, ValueError):
                av_score = 0.0

    text = f"{article.get('title','')} {article.get('summary','')}".lower()
    name_hit = bool(company_name) and company_name.lower() in text
    ticker_hit = bool(re.search(r"\b" + re.escape(ticker.lower()) + r"\b", text))

    is_relevant = av_score >= 0.15 or name_hit or ticker_hit
    return is_relevant, av_score


def build_news_dataframe(ticker: str, company_name: str = None, days_back: int = 25,
                          priority_days: int = 5, use_cache: bool = True) -> pd.DataFrame:
    """
    THE MAIN FUNCTION. Fetches real news for `ticker`, scores it, and
    returns one clean DataFrame:

        build_news_dataframe("NKE", company_name="Nike")

    Columns:
        title, summary, url, source, time_published, days_ago,
        recency_window ("priority_recent" | "context"), recency_weight (2.0 | 1.0),
        is_relevant, av_relevance_score,
        av_sentiment_score, av_sentiment_label   (Alpha Vantage's own sentiment),
        lm_positive, lm_negative, lm_polarity     (Loughran-McDonald financial tone)
    """
    articles = fetch_news(ticker, days_back=days_back, use_cache=use_cache)
    now = datetime.now(timezone.utc)
    rows = []

    for art in articles:
        ts = art.get("time_published")
        try:
            published = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) if ts else None
        except ValueError:
            published = None

        days_ago = (now - published).total_seconds() / 86400.0 if published else None
        if days_ago is not None and days_ago > days_back:
            continue  # safety net, time_from already filters most of this

        recency_window = "priority_recent" if (days_ago is not None and days_ago <= priority_days) else "context"
        recency_weight = 2.0 if recency_window == "priority_recent" else 1.0

        is_relevant, av_relevance = _is_relevant(art, ticker, company_name)

        title, summary = art.get("title", "") or "", art.get("summary", "") or ""
        full_text = f"{title}. {summary}"

        try:
            av_score = float(art.get("overall_sentiment_score", 0.0))
        except (TypeError, ValueError):
            av_score = 0.0

        lm_tokens = _LM.tokenize(full_text)
        lm_scores = _LM.get_score(lm_tokens)

        rows.append({
            "title": title,
            "summary": summary,
            "url": art.get("url"),
            "source": art.get("source"),
            "time_published": ts,
            "days_ago": round(days_ago, 2) if days_ago is not None else None,
            "recency_window": recency_window,
            "recency_weight": recency_weight,
            "is_relevant": is_relevant,
            "av_relevance_score": av_relevance,
            "av_sentiment_score": av_score,
            "av_sentiment_label": av_sentiment_label(av_score),
            "lm_positive": int(lm_scores["Positive"]),
            "lm_negative": int(lm_scores["Negative"]),
            "lm_polarity": round(float(lm_scores["Polarity"]), 3),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["recency_weight", "days_ago"], ascending=[False, True]).reset_index(drop=True)
    return df


# ============================================================================
# FRONTEND-FACING FUNCTIONS
# (give these two to your frontend developer -- description below)
# ============================================================================

def get_news_for_ui(ticker: str, company_name: str = None, days_back: int = 25,
                     priority_days: int = 5, only_relevant: bool = True) -> list:
    """
    Call this from your backend route. Returns a plain list of dicts
    (JSON-serializable) -- no pandas, no API key needed on the caller's side.

        get_news_for_ui("NKE", company_name="Nike")
        -> [{"title": "...", "av_sentiment_label": "Bullish", "recency_window": "priority_recent", ...}, ...]
    """
    df = build_news_dataframe(ticker, company_name=company_name, days_back=days_back, priority_days=priority_days)
    if only_relevant and not df.empty:
        df = df[df["is_relevant"]]
    return json.loads(df.to_json(orient="records"))


def get_news_summary_stats(ticker: str, company_name: str = None, days_back: int = 25,
                            priority_days: int = 5) -> dict:
    """
    Small aggregate for a dashboard tile -- counts and sentiment breakdown,
    no need to ship the whole article list just to show a header number.
    """
    df = build_news_dataframe(ticker, company_name=company_name, days_back=days_back, priority_days=priority_days)
    if df.empty:
        return {"total_articles": 0}
    relevant = df[df["is_relevant"]]
    return {
        "total_articles": int(len(df)),
        "relevant_articles": int(len(relevant)),
        "priority_recent_count": int((df["recency_window"] == "priority_recent").sum()),
        "context_count": int((df["recency_window"] == "context").sum()),
        "sentiment_breakdown": relevant["av_sentiment_label"].value_counts().to_dict(),
    }


# ============================================================================
# RUN IT AND SEE YOUR DATA
# ============================================================================

if __name__ == "__main__":
    TICKER = "NKE"
    COMPANY_NAME = "Nike"

    df = build_news_dataframe(TICKER, company_name=COMPANY_NAME)
    print(f"Fetched {len(df)} articles for {TICKER}.\n")
    print(df.head(10))

    df.to_csv("portfolio_pulse_news.csv", index=False)
    print("\nSaved full dataset to portfolio_pulse_news.csv")

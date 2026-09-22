"""
embeddings_store.py
====================
Step 2 of Portfolio Pulse: takes the DataFrame from preprocessing.py,
embeds each article, and stores it in a local vector database (Chroma) so
the agents in file 3 can retrieve "the most relevant news" instead of
reading every article every time.

Kept deliberately simple:
  - Chroma runs locally, no server/account/API key needed for the vector DB
    itself. It's saved to a folder on disk (`./chroma_db`) so it persists
    between runs.
  - Embedding model: `all-MiniLM-L6-v2` (small, free, downloads once from
    Hugging Face the first time you run this -- no API key needed for it
    either).
  - Recency weighting from preprocessing.py carries straight through as
    metadata, so retrieval can be biased toward the last 5 days without
    re-embedding anything.

Nothing in this file needs an API key.
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=PERSIST_DIR)


def build_vector_store(df: pd.DataFrame, ticker: str) -> "chromadb.Collection":
    """
    Embeds every row of `df` (the output of preprocessing.build_news_dataframe)
    and stores it in a Chroma collection named after the ticker. Re-running
    this for the same ticker replaces the old collection with fresh data.
    """
    collection_name = f"news_{ticker.lower()}"
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist yet -- fine
    collection = _client.create_collection(collection_name, embedding_function=_embedding_fn)

    if df.empty:
        return collection

    documents = (df["title"].fillna("") + ". " + df["summary"].fillna("")).tolist()
    ids = [str(i) for i in range(len(df))]
    df = df.reset_index(drop=True)
    metadatas = df[[
        "title", "url", "source", "days_ago", "recency_window", "recency_weight",
        "av_sentiment_label", "av_sentiment_score", "lm_polarity",
    ]].fillna("").to_dict(orient="records")
    for i, meta in enumerate(metadatas):
        meta["ticker"] = ticker.upper()
        meta["article_id"] = f"{ticker.upper()}_{i}"
        meta["publication_date"] = meta.get("days_ago")  # human-friendly date added below
    # replace days_ago-only dates with an actual calendar date string
    if "time_published" in df.columns:
        pub_dates = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
        for i, meta in enumerate(metadatas):
            meta["publication_date"] = pub_dates.iloc[i].strftime("%Y-%m-%d") if pd.notnull(pub_dates.iloc[i]) else "unknown"

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    return collection


def get_collection(ticker: str):
    """Loads an already-built collection for `ticker` without rebuilding it."""
    return _client.get_collection(f"news_{ticker.lower()}", embedding_function=_embedding_fn)


def retrieve_context(ticker: str, query: str = None, n_results: int = 10) -> list:
    """
    Returns the `n_results` most relevant articles for `query` (defaults to
    a generic "financial outlook and risk" query if you don't have a
    specific question). Each result is a dict with the article text,
    metadata (including recency_weight), and similarity distance.
    """
    collection = get_collection(ticker)
    query = query or f"{ticker} financial outlook, risks, and recent developments"
    results = collection.query(query_texts=[query], n_results=min(n_results, collection.count() or 1))

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({"text": doc, "metadata": meta, "distance": dist})

    # Prioritize recent news: sort by recency_weight (high first), then by
    # similarity distance (low = more relevant) as the tiebreaker.
    out.sort(key=lambda r: (-float(r["metadata"].get("recency_weight", 1.0)), r["distance"]))
    return out


def get_context_for_agents(ticker: str, query: str = None, n_results: int = 10) -> str:
    """
    THE FUNCTION THE AGENTS FILE CALLS. Returns one plain text block with
    the most relevant news, recent articles clearly marked, ready to drop
    straight into a prompt.
    """
    results = retrieve_context(ticker, query=query, n_results=n_results)
    if not results:
        return f"No recent news found for {ticker}."

    lines = []
    for r in results:
        meta = r["metadata"]
        tag = "[RECENT - last 5 days]" if meta.get("recency_window") == "priority_recent" else "[CONTEXT]"
        sentiment = meta.get("av_sentiment_label", "Neutral")
        source = meta.get("source", "Unknown source")
        pub_date = meta.get("publication_date", "unknown date")
        url = meta.get("url", "")
        lines.append(
            f"{tag} ({sentiment}) {r['text']}\n"
            f"    Source: {source} | Published: {pub_date} | URL: {url}"
        )
    return "\n\n".join(lines)


# ============================================================================
# FRONTEND-FACING FUNCTION
# ============================================================================

def get_relevant_news_for_ui(ticker: str, query: str = None, n_results: int = 10) -> list:
    """
    Call this from your backend route if the frontend wants a "most relevant
    articles" panel instead of "all articles" (which preprocessing.py's
    get_news_for_ui already provides). Returns plain JSON-serializable dicts.
    """
    return retrieve_context(ticker, query=query, n_results=n_results)


if __name__ == "__main__":
    from preprocessing import build_news_dataframe

    TICKER = "NKE"
    df = build_news_dataframe(TICKER, company_name="Nike")
    relevant_df = df[df["is_relevant"]] if not df.empty else df

    collection = build_vector_store(relevant_df, TICKER)
    print(f"Stored {collection.count()} articles in the vector DB for {TICKER}.\n")

    context = get_context_for_agents(TICKER)
    print("=== Retrieved context that would be handed to the agents ===")
    print(context)

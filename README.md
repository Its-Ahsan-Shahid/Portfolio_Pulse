# Portfolio Pulse -- Backend (simple version)

Four files, no frontend, no framework. Real Alpha Vantage news in, a bull
vs. bear vs. moderator verdict out.

```
preprocessing.py      -> fetches + scores real news for a ticker (pandas DataFrame)
embeddings_store.py   -> embeds it into a local vector DB (Chroma)
agents.py             -> bull / bear / moderator agents (Grok / xAI API)
summary.py            -> runs all three and gives you the final report
```

## Setup

```bash
pip install -r requirements.txt
```

## Where to put your API keys

Two keys needed, one per file that uses an external API:

- **Alpha Vantage** (free): get one at https://www.alphavantage.co/support/#api-key
  Open `preprocessing.py`, find the block near the top that says
  `PUT YOUR API KEY HERE`, and paste it between the quotes:
  ```python
  ALPHAVANTAGE_API_KEY = "your_key_here"
  ```
- **Grok / xAI**: get one at https://console.x.ai
  Open `agents.py`, find the same kind of block, and paste it there:
  ```python
  XAI_API_KEY = "your_key_here"
  ```

That's enough to run everything locally or in Colab today.

**Before you push this to GitHub**, delete what you pasted (put the quotes
back to `""`) and instead set the same two names as environment variables
on whatever machine actually runs the code (your laptop's `.env` file, a
Colab secret, your server's environment, etc.):
```
ALPHAVANTAGE_API_KEY=...
XAI_API_KEY=...
```
Both files check the environment variable first automatically, so nothing
else changes -- you just stop leaving a real key sitting in a file that
goes into git history.

## Running it

```bash
python summary.py
```
This runs the whole pipeline for NKE/Nike (edit the ticker at the bottom of
`summary.py`) and prints a bull case, a bear case, and a moderator verdict,
then saves `portfolio_pulse_report.json`.

You can also run any file on its own to see just that stage's output and
its own saved data file (`portfolio_pulse_news.csv` from `preprocessing.py`,
console output from `embeddings_store.py`, etc.).

## Functions to give your frontend developer

Tell them: call these, you don't need any API key on your end, and every
one of them returns a plain JSON-serializable Python `dict`/`list` (no
pandas, no custom objects).

| Function | File | What it returns |
|---|---|---|
| `generate_portfolio_pulse_report(ticker, company_name=None)` | `summary.py` | **The main one.** One dict with news stats + bull case + bear case + moderator verdict -- everything one "Analyze" button needs. |
| `get_news_for_ui(ticker, company_name=None)` | `preprocessing.py` | List of every relevant news article with its sentiment, if you want a raw news feed panel in the UI. |
| `get_news_summary_stats(ticker, company_name=None)` | `preprocessing.py` | Small dict of counts (total articles, how many are from the last 5 days, sentiment breakdown) -- good for a dashboard header tile. |
| `get_relevant_news_for_ui(ticker, query=None)` | `embeddings_store.py` | List of the most relevant articles for a specific question, if you add a search box later. |

In practice, your backend route for an "Analyze [TICKER]" button is just:
```python
from summary import generate_portfolio_pulse_report
report = generate_portfolio_pulse_report(ticker, company_name=company_name)
return jsonify(report)   # or however your framework returns JSON
```

## Notes

- Alpha Vantage's free tier is 25 requests/day. `preprocessing.py` caches
  each day's response to disk per ticker, so re-running the pipeline
  several times on the same ticker in one day costs one request, not many.
- The Loughran-McDonald financial dictionary is used via the `pysentiment2`
  package -- it's bundled inside the package, so there's no CSV to download
  or upload anywhere.
- Alpha Vantage's own sentiment bucket definitions (used in the output):
  `<= -0.35` Bearish, `-0.35..-0.15` Somewhat-Bearish, `-0.15..0.15` Neutral,
  `0.15..0.35` Somewhat-Bullish, `>= 0.35` Bullish.

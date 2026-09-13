"""
summary.py
==========
Step 4 of Portfolio Pulse: runs the whole pipeline end to end for a ticker
and produces one final summary dict/report:

    news --> vector store --> bull/bear/moderator debate --> summary

This is the single function your frontend/backend route should actually
call for the "analyze this stock" button.
"""

import json

from preprocessing import build_news_dataframe, get_news_summary_stats
from embeddings_store import build_vector_store, get_context_for_agents
from agents import run_debate


def generate_portfolio_pulse_report(ticker: str, company_name: str = None,
                                     days_back: int = 25, priority_days: int = 5,
                                     question: str = None) -> dict:
    """
    THE ONE FUNCTION TO CALL FOR THE WHOLE PIPELINE.

        generate_portfolio_pulse_report("NKE", company_name="Nike")

    Returns a plain, JSON-serializable dict:
        {
          "ticker": "NKE",
          "news_stats": {...},                 # counts + sentiment breakdown
          "bull_case": "...",
          "bear_case": "...",
          "moderator_verdict": "...",
        }
    """
    # 1. Real news + scoring
    df = build_news_dataframe(ticker, company_name=company_name,
                               days_back=days_back, priority_days=priority_days)
    relevant_df = df[df["is_relevant"]] if not df.empty else df
    news_stats = get_news_summary_stats(ticker, company_name=company_name,
                                         days_back=days_back, priority_days=priority_days)

    # 2. Embed + store, then retrieve the most relevant context
    build_vector_store(relevant_df, ticker)
    context = get_context_for_agents(ticker, query=question)

    # 3. Bull / bear / moderator debate
    debate = run_debate(ticker, context)

    return {
        "ticker": ticker.upper(),
        "news_stats": news_stats,
        "bull_case": debate["bull"],
        "bear_case": debate["bear"],
        "moderator_verdict": debate["moderator"],
    }


def print_report(report: dict) -> None:
    """Pretty-prints a report dict to the console/notebook."""
    print(f"\n{'='*60}\nPORTFOLIO PULSE REPORT: {report['ticker']}\n{'='*60}")
    print(f"\nNews stats: {json.dumps(report['news_stats'], indent=2)}")
    print(f"\n--- BULL CASE ---\n{report['bull_case']}")
    print(f"\n--- BEAR CASE ---\n{report['bear_case']}")
    print(f"\n--- MODERATOR VERDICT ---\n{report['moderator_verdict']}")


if __name__ == "__main__":
    report = generate_portfolio_pulse_report("NKE", company_name="Nike")
    print_report(report)

    with open("portfolio_pulse_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full report to portfolio_pulse_report.json")

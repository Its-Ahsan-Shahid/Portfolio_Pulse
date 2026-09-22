"""
agents.py
=========
Step 3 of Portfolio Pulse: three simple agents that read the news context
retrieved in embeddings_store.py and argue about it, in the style of
StockSense-AI's bull/bear debate -- kept deliberately plain (no framework
like LangGraph, just three prompt-based function calls):

    bull_agent(ticker, context)      -> makes the strongest bullish case
    bear_agent(ticker, context)      -> makes the strongest bearish case
    moderator_agent(ticker, ...)     -> weighs both and gives a final verdict

Each agent uses:
  - In-context learning: the prompt includes a short worked example of the
    exact output format we want, so the model imitates the structure.
  - Chain-of-thought: the model is told to reason step by step internally
    before answering, which measurably improves financial reasoning quality
    -- but we ask it to keep that reasoning brief and only show the final
    structured answer, so the output stays usable in a UI.

Uses xAI's Grok API, which is OpenAI-compatible (plain HTTPS + `requests`,
no extra SDK needed).

--------------------------------------------------------------------------
WHERE TO PUT YOUR API KEY
--------------------------------------------------------------------------
Scroll down to the block that says "PUT YOUR API KEY HERE" (right after the
imports). Same two options as preprocessing.py:
  1. Quick testing: paste your key between the quotes there.
  2. Before pushing to GitHub: leave it as "" and set an environment
     variable named XAI_API_KEY instead. The code checks the environment
     variable first automatically.

Get a key at: https://console.x.ai  (Grok / xAI API keys)
Current model names change occasionally -- check https://docs.x.ai if the
default model below ever returns a "model not found" error.
"""

import os
import requests
from getpass import getpass
!pip install -q groq gradio
from groq import Groq

# ============================================================================
# PUT YOUR API KEY HERE
# ============================================================================
GROQ_API_KEY = "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"   # <-- paste your Groq key (from console.groq.com) between the quotes
# ============================================================================

MODEL = "openai/gpt-oss-120b"


def get_api_key() -> str:
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return (
        os.getenv("GROQ_API_KEY")
        or GROQ_API_KEY
        or getpass("Enter your Groq API key: ")
    ).strip()


def call_grok(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    client = Groq(api_key=get_api_key())
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Shared instruction: think step by step, but only show the final answer in
# this exact structure. This is the in-context example every agent reuses.
# ---------------------------------------------------------------------------
_FORMAT_EXAMPLE = """
Example of the format to follow (for a different company, just for structure):

Case: Strong bullish case for ACME Corp based on the news.
Key Points:
- ACME beat earnings estimates by 12%, showing real demand strength.
- New product line launched last week, expanding total addressable market.
Confidence: High
"""


def bull_agent(ticker: str, context: str) -> str:
    """Argues the strongest BULLISH case it can, using only the given news context."""
    system_prompt = (
        "You are a bullish equity research analyst. Your job is to build the "
        "strongest reasonable bullish case for the stock using ONLY the news "
        "provided -- do not invent facts that aren't in the context. "
        "Think through the news step by step first, then give only the final "
        "answer in the exact format shown in the example. Keep it concise: "
        "3-5 key points, plain language a portfolio manager can skim in 10 seconds."
        + _FORMAT_EXAMPLE
    )
    user_prompt = f"Ticker: {ticker}\n\nRecent news context:\n{context}\n\nGive the bullish case."
    return call_grok(system_prompt, user_prompt)


def bear_agent(ticker: str, context: str) -> str:
    """Argues the strongest BEARISH case it can, using only the given news context."""
    system_prompt = (
        "You are a bearish/risk-focused equity research analyst. Your job is "
        "to build the strongest reasonable bearish case for the stock using "
        "ONLY the news provided -- do not invent facts that aren't in the "
        "context. Think through the news step by step first, then give only "
        "the final answer in the exact format shown in the example. Keep it "
        "concise: 3-5 key points, plain language a portfolio manager can "
        "skim in 10 seconds."
        + _FORMAT_EXAMPLE
    )
    user_prompt = f"Ticker: {ticker}\n\nRecent news context:\n{context}\n\nGive the bearish case."
    return call_grok(system_prompt, user_prompt)


def moderator_agent(ticker: str, bull_case: str, bear_case: str, context: str) -> str:
    """Weighs the bull and bear cases and gives a balanced final verdict."""
    system_prompt = (
        "You are a neutral portfolio risk moderator. You are given a bullish "
        "case and a bearish case for the same stock, both built from the "
        "same news. Weigh them against each other and the underlying news "
        "context, then give a short, balanced verdict -- do not just repeat "
        "both sides, actually judge which concerns matter more right now and "
        "why. Explicitly favor news tagged [RECENT - last 5 days] over "
        "[CONTEXT] when they conflict, since it's more current. Think step "
        "by step first, then answer in this format:\n\n"
        "Verdict: <one sentence: net bullish / net bearish / balanced, and why>\n"
        "Reasoning:\n- <point>\n- <point>\n"
        "Confidence: <Low/Medium/High>"
    )
    user_prompt = (
        f"Ticker: {ticker}\n\n"
        f"Bull case:\n{bull_case}\n\n"
        f"Bear case:\n{bear_case}\n\n"
        f"Underlying news context:\n{context}\n\n"
        "Give your verdict."
    )
    return call_grok(system_prompt, user_prompt)


# ============================================================================
# FRONTEND-FACING FUNCTION
# ============================================================================

def run_debate(ticker: str, context: str) -> dict:
    """
    THE FUNCTION TO CALL. Runs all three agents in order and returns a
    plain dict -- ready to send straight to a frontend as JSON.

        run_debate("NKE", context)
        -> {"bull": "...", "bear": "...", "moderator": "..."}
    """
    bull = bull_agent(ticker, context)
    bear = bear_agent(ticker, context)
    moderator = moderator_agent(ticker, bull, bear, context)
    return {"bull": bull, "bear": bear, "moderator": moderator}


if __name__ == "__main__":
    sample_context = (
        "[RECENT - last 5 days] (Somewhat-Bearish) Nike cuts annual revenue guidance "
        "amid weak China sales.\n"
        "[CONTEXT] (Bullish) Nike unveils new advertisement campaign featuring rising "
        "basketball stars."
    )
    result = run_debate("NKE", sample_context)
    print("=== BULL ===\n", result["bull"])
    print("\n=== BEAR ===\n", result["bear"])
    print("\n=== MODERATOR ===\n", result["moderator"])

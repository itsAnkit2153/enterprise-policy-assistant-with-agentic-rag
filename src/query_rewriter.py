"""
Query Rewriting
===============
Pre-processes the user's raw question before retrieval to improve recall:

1. Rewrites the query to be more specific and retrieval-friendly
2. Generates up to 3 alternative phrasings with different vocabulary
3. Returns all variants so retriever can search with each and merge results

Output schema:
{
  "original_query"  : str,
  "rewritten_query" : str,
  "alternatives"    : [str, str, str],
  "reasoning"       : str
}
"""

import json
from langchain_groq import ChatGroq
from src.config import GROQ_MODEL, GROQ_API_KEY


def query_rewriting_prompt() -> str:
    return """You are a Query Rewriting specialist for an Enterprise Policy Document search system.

Your job is to take a user's raw question and rewrite it to maximise retrieval recall from a vector database of policy PDFs.

Rewriting rules:
1. Expand abbreviations and acronyms (e.g. "PTO" → "paid time off", "WFH" → "work from home")
2. Replace informal language with formal policy terminology
3. Make implicit subjects explicit (e.g. "how many days?" → "how many days of annual leave is an employee entitled to?")
4. Remove filler words and conversational tone
5. Generate up to 3 alternative phrasings that use different vocabulary but mean the same thing
   — these help catch chunks that use synonyms or different section titles

Return ONLY valid JSON — no markdown fences, no preamble:

{
  "original_query"  : "<user's original question>",
  "rewritten_query" : "<improved, retrieval-optimised version>",
  "alternatives"    : ["<alt phrasing 1>", "<alt phrasing 2>", "<alt phrasing 3>"],
  "reasoning"       : "<one sentence explaining what you changed and why>"
}"""


def rewrite_query(user_question: str) -> dict:
    """
    Call the Groq LLM to rewrite the user's question for better retrieval.
    Returns a dict with: original_query, rewritten_query, alternatives, reasoning.
    Falls back gracefully if LLM call fails.
    """
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

    try:
        response = llm.invoke([
            {"role": "system", "content": query_rewriting_prompt()},
            {"role": "user",   "content": user_question},
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        result = json.loads(raw)

        # Normalise — guarantee all keys exist
        return {
            "original_query":  user_question,
            "rewritten_query": result.get("rewritten_query", user_question),
            "alternatives":    result.get("alternatives", [])[:3],
            "reasoning":       result.get("reasoning", ""),
        }

    except Exception as e:
        # Graceful fallback — pass original query through unchanged
        return {
            "original_query":  user_question,
            "rewritten_query": user_question,
            "alternatives":    [],
            "reasoning":       f"Query rewriting unavailable: {e}",
        }

import json
from langchain_core.tools import tool
from src.config import TOP_K
from src.retriever import retrieve_multi
from src.query_rewriter import rewrite_query

# Vector store injected at runtime
_vector_store = None


def set_vector_store(vs):
    """Inject the active vector store into the tools module."""
    global _vector_store
    _vector_store = vs


def _require_vs():
    if _vector_store is None:
        raise RuntimeError(
            "No vector store loaded. Please upload and index policy documents first."
        )
    return _vector_store


@tool
def query_rewriter(user_question: str) -> str:
    """
    Rewrite the user's question to improve retrieval from the policy vector store.

    Call this tool FIRST — before policy_retriever — on every user question.

    It expands abbreviations, formalises language, and generates alternative phrasings.
    Returns a JSON object with:
      - rewritten_query : the improved query to pass to policy_retriever
      - alternatives    : up to 3 alternative phrasings
      - reasoning       : what was changed and why

    Pass ALL queries (rewritten_query + alternatives) to policy_retriever.
    """
    result = rewrite_query(user_question)
    return json.dumps(result, ensure_ascii=False)


@tool
def policy_retriever(queries_json: str) -> str:
    """
    Search the uploaded policy documents using one or more queries.

    Input: a JSON string — either:
      - A single query string: "how many days annual leave?"
      - A JSON array of query strings: ["rewritten query", "alt 1", "alt 2"]
      - The full JSON output from query_rewriter (uses rewritten_query + alternatives)

    Results from all queries are merged and deduplicated by chunk_id.
    Returns a JSON list of matching chunks, each with:
      - content     : the policy excerpt text
      - source_file : the PDF filename
      - chunk_id    : unique ID (used for citation verification)
      - score       : relevance score 0–1
      - page        : page number in the source PDF

    Use the returned context — and ONLY that context — to compose your answer.
    Always call query_rewriter first, then pass its output here.
    """
    vs = _require_vs()

    # Parse input flexibly
    queries: list[str] = []

    raw = queries_json.strip()

    # Try JSON parse first
    try:
        parsed = json.loads(raw)

        if isinstance(parsed, str):
            queries = [parsed]

        elif isinstance(parsed, list):
            queries = [q for q in parsed if isinstance(q, str) and q.strip()]

        elif isinstance(parsed, dict):
            # Full query_rewriter output
            rw = parsed.get("rewritten_query", "")
            alts = parsed.get("alternatives", [])
            if rw:
                queries.append(rw)
            queries.extend([a for a in alts if isinstance(a, str) and a.strip()])

    except json.JSONDecodeError:
        # Plain string fallback
        queries = [raw]

    if not queries:
        return json.dumps({"status": "error", "message": "No valid queries provided.", "chunks": []})

    chunks = retrieve_multi(queries, vs, top_k=TOP_K)

    if not chunks:
        return json.dumps({
            "status":  "no_results",
            "message": "No relevant policy content found for this question.",
            "chunks":  [],
        })

    return json.dumps({
        "status":      "ok",
        "query_count": len(queries),
        "chunk_count": len(chunks),
        "chunks":      chunks,
    }, ensure_ascii=False)

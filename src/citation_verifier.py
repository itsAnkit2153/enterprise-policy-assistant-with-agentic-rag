"""
Citation Verification
=====================
After the agent returns sources, verify each cited chunk against ChromaDB:

1. Existence check   — does the chunk_id actually exist in the store?
2. Attribution check — does the stored chunk's source_file match what was cited?
3. Relevance check   — do key terms from the answer appear in the chunk content?

Each source is tagged with:
  "verified"  : True | False
  "status"    : "VERIFIED" | "WRONG_SOURCE" | "NOT_FOUND" | "PARTIAL"
  "detail"    : human-readable explanation
"""

import re
from src.vector_store import get_chunk_by_id


def _keyword_overlap(answer: str, content: str, threshold: float = 0.15) -> bool:
    """
    Return True if enough meaningful words from the answer appear in the chunk.
    Uses a simple token-overlap ratio.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "about", "from", "and", "or", "but",
        "not", "no", "so", "if", "as", "it", "its", "this", "that", "i",
        "you", "we", "they", "he", "she", "my", "your", "our", "their",
    }

    def tokens(text: str) -> set[str]:
        words = re.findall(r"[a-z]+", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 3}

    answer_tokens = tokens(answer)
    if not answer_tokens:
        return True  # can't check — don't penalise

    content_tokens = tokens(content)
    overlap = answer_tokens & content_tokens
    ratio = len(overlap) / len(answer_tokens)
    return ratio >= threshold


def verify_citations(
    sources: list[dict],
    answer: str,
    vector_store,
) -> list[dict]:
    """
    Verify each source dict (from agent output) against ChromaDB.

    Input source dict fields used:
      chunk_id, source_file, snippet (optional)

    Returns the same list with three extra fields per source:
      verified : bool
      status   : str
      detail   : str
    """
    verified_sources = []

    for src in sources:
        chunk_id  = src.get("chunk_id", "").strip()
        cited_file = src.get("source_file", "").strip()

        # ── 1. Existence check ────────────────────────────────────────────────
        if not chunk_id:
            verified_sources.append({
                **src,
                "verified": False,
                "status":   "NOT_FOUND",
                "detail":   "No chunk_id provided — cannot verify.",
            })
            continue

        stored = get_chunk_by_id(vector_store, chunk_id)

        if stored is None:
            verified_sources.append({
                **src,
                "verified": False,
                "status":   "NOT_FOUND",
                "detail":   f"chunk_id '{chunk_id}' does not exist in the index.",
            })
            continue

        # ── 2. Attribution check ──────────────────────────────────────────────
        stored_file = (
            stored["metadata"].get("source_file")
            or stored["metadata"].get("source_filename", "")
        ).strip()

        if cited_file and stored_file and cited_file != stored_file:
            verified_sources.append({
                **src,
                "verified": False,
                "status":   "WRONG_SOURCE",
                "detail":   f"Cited as '{cited_file}' but chunk belongs to '{stored_file}'.",
            })
            continue

        # ── 3. Relevance check ────────────────────────────────────────────────
        content = stored["content"]
        relevant = _keyword_overlap(answer, content)

        if relevant:
            verified_sources.append({
                **src,
                "verified": True,
                "status":   "VERIFIED",
                "detail":   "Chunk exists, source matches, and content supports the answer.",
            })
        else:
            verified_sources.append({
                **src,
                "verified": False,
                "status":   "PARTIAL",
                "detail":   "Chunk exists and source matches, but content overlap with the answer is low.",
            })

    return verified_sources

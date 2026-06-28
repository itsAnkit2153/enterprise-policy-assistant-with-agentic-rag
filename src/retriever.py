from src.config import TOP_K


def retrieve(query: str, vector_store, top_k: int = TOP_K) -> list[dict]:
    """
    Single-query cosine similarity search.
    Returns chunks matching prompt-3 output schema:
      content, source_file, chunk_id, score, document_label, page
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, distance in results:
        score = round(max(0.0, 1.0 - float(distance)), 4)
        chunks.append({
            "content":        doc.page_content,
            "source_file":    doc.metadata.get("source_file")
                              or doc.metadata.get("source_filename", "unknown"),
            "chunk_id":       doc.metadata.get("chunk_id", ""),
            "score":          score,
            "document_label": doc.metadata.get("document_label", "Unknown Document"),
            "page":           doc.metadata.get("page", 0),
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def retrieve_multi(queries: list[str], vector_store, top_k: int = TOP_K) -> list[dict]:
    """
    Run retrieve() for each query in the list, merge results, and deduplicate
    by chunk_id — keeping the highest score seen for each chunk.
    Returns a list sorted by score descending.
    """
    seen: dict[str, dict] = {}   # chunk_id → best chunk dict

    for query in queries:
        for chunk in retrieve(query, vector_store, top_k=top_k):
            cid = chunk["chunk_id"]
            if cid not in seen or chunk["score"] > seen[cid]["score"]:
                seen[cid] = chunk

    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k * 2]   # cap total results at 2× top_k

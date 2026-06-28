import time
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from src.config import VECTOR_STORE_PATH, COLLECTION_NAME


def _chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_vector_store(embedding_model) -> Chroma:
    """Load (or create) the persistent ChromaDB vector store."""
    client = _chroma_client()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTOR_STORE_PATH,
        client=client,
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_vector_store(embedding_model) -> Chroma:
    """Delete the existing collection and return a fresh store."""
    client = _chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_vector_store(embedding_model)


def get_indexed_filenames(vector_store: Chroma) -> set[str]:
    """
    Return the set of source_file values already present in the collection.
    Used for incremental indexing — skip files already indexed.
    """
    try:
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        filenames = set()
        for meta in results.get("metadatas", []):
            fn = meta.get("source_file") or meta.get("source_filename")
            if fn:
                filenames.add(fn)
        return filenames
    except Exception:
        return set()


def get_chunk_by_id(vector_store: Chroma, chunk_id: str) -> dict | None:
    """
    Fetch a single chunk by its chunk_id metadata field.
    Returns dict with content + metadata, or None if not found.
    Used by citation verifier.
    """
    try:
        collection = vector_store._collection
        results = collection.get(
            where={"chunk_id": chunk_id},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        if docs:
            return {"content": docs[0], "metadata": metas[0] if metas else {}}
        return None
    except Exception:
        return None


def index_chunks(chunks: list, vector_store: Chroma, batch_size: int = 25) -> int:
    """
    Add chunks to the vector store in batches.
    Returns the total number of chunks indexed.
    """
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [chunk.metadata["chunk_id"] for chunk in batch]
        vector_store.add_documents(documents=batch, ids=ids)
        if i + batch_size < total:
            time.sleep(1)
    return total


def get_total_chunks(vector_store: Chroma) -> int:
    """Return total number of chunks in the collection."""
    try:
        return vector_store._collection.count()
    except Exception:
        return 0

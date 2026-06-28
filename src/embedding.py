from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Vector Store ---
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "policy_documents"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 4

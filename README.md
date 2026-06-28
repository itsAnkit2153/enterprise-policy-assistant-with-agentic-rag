# Enterprise Policy Assistant — Agentic RAG

A Streamlit-powered AI agent that answers employee questions about company policies using **dynamic PDF upload**, **ChromaDB vector search**, a **LangGraph ReAct agent** backed by Groq LLM, and **citation verification**.

---

## Features

| Feature | Detail |
|---|---|
| Dynamic PDF upload | No hardcoded policy categories — any PDF works |
| Deduplication | Duplicate filenames are detected and skipped |
| Incremental indexing | Re-uploading skips already-indexed files |
| Agentic retrieval | LangGraph ReAct agent calls `policy_retriever` before every answer |
| Grounded answers | Strict: never answers from model memory |
| Citation verification | Every cited chunk is verified against ChromaDB (existence + attribution + relevance) |
| Source citations | Document name, page number, chunk ID, verbatim snippet |
| Structured output | `answer`, `sources`, `confidence`, `reasoning` |
| Chat history | Full session history with collapsible past questions |

---

## Project Structure

```
policy-assistant/
├── app.py                    # Streamlit UI entry point
├── requirements.txt
├── .env.example
└── src/
    ├── config.py             # Env vars + constants
    ├── loader.py             # PDF → LangChain Documents
    ├── chunking.py           # Token-aware recursive splitting (500/100)
    ├── metadata.py           # Chunk enrichment: source_file, document_type, upload_time, chunk_id
    ├── embedding.py          # HuggingFace embeddings (cached singleton)
    ├── vector_store.py       # ChromaDB: create / reset / index / query by ID
    ├── retriever.py          # Cosine similarity search → {content, source_file, chunk_id, score}
    ├── tools.py              # LangChain tool: policy_retriever
    ├── prompts.py            # System prompt (prompt-5 spec)
    ├── agent.py              # LangGraph ReAct agent builder + runner
    └── citation_verifier.py  # Existence + attribution + relevance checks per source
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # add your GROQ_API_KEY
streamlit run app.py
```

---

## Environment Variables

| Variable | Required | Default |
|---|---|---|
| `GROQ_API_KEY` | ✅ | — |
| `GROQ_MODEL` | ❌ | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | ❌ | — (only for gated HF models) |

---

## Citation Verification

After the agent returns sources, each citation is checked against ChromaDB:

1. **Existence** — does the `chunk_id` exist in the index?
2. **Attribution** — does the stored chunk's `source_file` match what was cited?
3. **Relevance** — do key answer terms appear in the chunk content?

Each source displays one of four statuses:
- ✅ **VERIFIED** — all three checks passed
- ⚠️ **PARTIAL** — chunk exists and source matches, but content overlap is low
- 🔀 **WRONG_SOURCE** — chunk exists but belongs to a different file
- ❌ **NOT_FOUND** — chunk_id does not exist in the index (hallucinated citation)

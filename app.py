import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import streamlit as st
from datetime import datetime, timezone

from src.loader import load_pdfs_from_uploads
from src.chunking import chunk_documents
from src.metadata import enrich_metadata
from src.embedding import get_embedding_model
from src.vector_store import (
    get_vector_store, reset_vector_store, index_chunks,
    get_indexed_filenames, get_total_chunks,
)
from src.agent import build_agent, run_agent
from src.query_rewriter import rewrite_query
from src.citation_verifier import verify_citations

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0f172a;
  border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

.sidebar-stat {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  margin: 0.4rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
}
.sidebar-stat .label { color: #94a3b8 !important; }
.sidebar-stat .value { color: #38bdf8 !important; font-weight: 700; font-size: 1.1rem; }

.file-pill {
  background: #1e3a5f;
  border: 1px solid #2563eb44;
  border-radius: 6px;
  padding: 0.3rem 0.6rem;
  margin: 0.2rem 0;
  font-size: 0.78rem;
  color: #93c5fd !important;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.file-pill-new { border-color: #22c55e88; color: #86efac !important; background: #14532d44; }
.file-pill-skip { border-color: #f59e0b88; color: #fcd34d !important; background: #78350f44; }

section[data-testid="stSidebar"] .stButton > button {
  background: #2563eb;
  color: #fff !important;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  width: 100%;
  padding: 0.65rem 1rem;
  font-size: 0.9rem;
  transition: background 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #1d4ed8; }
section[data-testid="stSidebar"] .stButton > button:disabled {
  background: #334155 !important;
  color: #64748b !important;
}

/* ── Main ── */
.main { background: #f8fafc; }
.block-container { padding: 1.75rem 2.5rem; max-width: 860px; }

/* ── Page header ── */
.pa-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  border-radius: 14px;
  padding: 1.75rem 2rem;
  margin-bottom: 1.5rem;
}
.pa-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.pa-header p  { margin: 0.3rem 0 0; color: #94a3b8; font-size: 0.9rem; }

/* ── Chat bubble (question) ── */
.chat-question {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 12px 12px 12px 2px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.95rem;
  color: #1e3a5f;
  font-weight: 500;
}

/* ── Answer card ── */
.answer-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.section-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
  margin: 1rem 0 0.4rem;
}
.answer-text { color: #1e293b; font-size: 0.95rem; line-height: 1.7; }

/* ── Badges ── */
.badge {
  display: inline-block;
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  margin-right: 0.35rem;
}
.b-high     { background: #dcfce7; color: #166534; }
.b-medium   { background: #fef9c3; color: #854d0e; }
.b-low      { background: #fee2e2; color: #991b1b; }

/* ── Source chip ── */
.source-chip {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.75rem 1rem;
  margin-bottom: 0.6rem;
  font-size: 0.83rem;
  color: #334155;
}
.source-chip .src-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
}
.source-chip strong { color: #0f172a; }
.src-meta { color: #64748b; font-size: 0.78rem; }
.src-snippet {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  color: #64748b;
  border-left: 3px solid #3b82f6;
  padding-left: 0.6rem;
  margin-top: 0.35rem;
  line-height: 1.5;
}

/* Citation verification badges */
.cv-verified { background: #dcfce7; color: #166534; }
.cv-partial  { background: #fef9c3; color: #854d0e; }
.cv-notfound { background: #fee2e2; color: #991b1b; }
.cv-wrong    { background: #ffe4e6; color: #9f1239; }
.cv-detail   { font-size: 0.72rem; color: #64748b; margin-top: 0.25rem; }

/* ── Reasoning box ── */
.reasoning-box {
  background: #f1f5f9;
  border-left: 4px solid #3b82f6;
  border-radius: 0 8px 8px 0;
  padding: 0.7rem 1rem;
  font-size: 0.85rem;
  color: #334155;
  line-height: 1.6;
}

/* ── Query form ── */
.stTextArea textarea {
  border-radius: 10px !important;
  border: 1.5px solid #cbd5e1 !important;
  font-size: 0.93rem !important;
}
.stTextArea textarea:focus {
  border-color: #3b82f6 !important;
  box-shadow: 0 0 0 3px #3b82f620 !important;
}

/* ── History expander ── */
.streamlit-expanderHeader {
  background: #f8fafc !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  font-size: 0.88rem !important;
  color: #334155 !important;
}

/* ── Empty state ── */
.empty-state {
  text-align: center;
  padding: 3rem 1rem;
  color: #94a3b8;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-state p { font-size: 0.95rem; }

#MainMenu, footer { visibility: hidden; }

/* -- Query Rewriting panel -- */
.qr-panel {
  background: #fafafa;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #8b5cf6;
  border-radius: 0 10px 10px 0;
  padding: 0.9rem 1.1rem;
  margin-bottom: 1rem;
  font-size: 0.85rem;
}
.qr-panel .qr-title {
  font-weight: 700;
  color: #6d28d9;
  font-size: 0.78rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.qr-row { margin-bottom: 0.3rem; color: #334155; }
.qr-row span.label { color: #94a3b8; font-size: 0.78rem; margin-right: 0.4rem; }
.qr-row span.value { color: #1e293b; font-weight: 500; }
.qr-alt {
  display: inline-block;
  background: #ede9fe;
  color: #5b21b6;
  border-radius: 5px;
  padding: 0.1rem 0.5rem;
  font-size: 0.76rem;
  margin: 0.15rem 0.2rem 0.15rem 0;
}
.qr-reason { color: #64748b; font-size: 0.78rem; margin-top: 0.4rem; font-style: italic; }

</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "agent": None,
    "vector_store": None,
    "indexed_files": [],
    "total_chunks": 0,
    "chat_history": [],
    "upload_buffer": {},   # filename → UploadedFile, deduplication store
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
def confidence_badge(conf: str) -> str:
    cls = {"HIGH": "b-high", "MEDIUM": "b-medium", "LOW": "b-low"}.get(conf.upper(), "b-low")
    return f'<span class="badge {cls}">⬤ {conf} confidence</span>'



def render_rewrite_panel(meta: dict):
    """Display the query rewriting metadata panel."""
    if not meta:
        return
    original  = meta.get("original_query", "")
    rewritten = meta.get("rewritten_query", "")
    alts      = meta.get("alternatives", [])
    reasoning = meta.get("reasoning", "")

    # Only show panel if the query was actually changed
    if rewritten == original and not alts:
        return

    alt_html = "".join(f'<span class="qr-alt">{a}</span>' for a in alts) if alts else "<em style='color:#94a3b8'>none</em>"

    st.markdown(f"""
    <div class="qr-panel">
      <div class="qr-title">&#x1F50D; Query Rewriting</div>
      <div class="qr-row"><span class="label">Original:</span><span class="value">{original}</span></div>
      <div class="qr-row"><span class="label">Rewritten:</span><span class="value">{rewritten}</span></div>
      <div class="qr-row"><span class="label">Alternatives:</span>{alt_html}</div>
      {"<div class='qr-reason'>" + reasoning + "</div>" if reasoning else ""}
    </div>
    """, unsafe_allow_html=True)

CV_ICONS = {
    "VERIFIED":     ("✅", "cv-verified"),
    "PARTIAL":      ("⚠️", "cv-partial"),
    "NOT_FOUND":    ("❌", "cv-notfound"),
    "WRONG_SOURCE": ("🔀", "cv-wrong"),
}


def render_source(src: dict):
    status  = src.get("status", "NOT_FOUND")
    icon, cv_cls = CV_ICONS.get(status, ("❓", "cv-notfound"))
    doc     = src.get("source_file", "Unknown")
    page    = src.get("page", "?")
    cid     = src.get("chunk_id", "")
    snippet = (src.get("snippet") or "")[:200]
    detail  = src.get("detail", "")

    st.markdown(f"""
    <div class="source-chip">
      <div class="src-header">
        <strong>📄 {doc}</strong>
        <span class="src-meta">Page {page}</span>
        <span class="src-meta">·</span>
        <span class="src-meta"><code>{cid}</code></span>
        <span class="badge {cv_cls}">{icon} {status.replace("_", " ")}</span>
      </div>
      {"<div class='src-snippet'>" + snippet + "</div>" if snippet else ""}
      {"<div class='cv-detail'>" + detail + "</div>" if detail else ""}
    </div>
    """, unsafe_allow_html=True)


def render_result(result: dict):
    conf    = result.get("confidence", "LOW")
    answer  = result.get("answer", "No answer generated.")
    sources = result.get("sources", [])
    reason  = result.get("reasoning", "")

    st.markdown(f"""
    <div class="answer-card">
      <div>{confidence_badge(conf)}</div>
      <div class="section-label">Answer</div>
      <div class="answer-text">{answer}</div>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        st.markdown('<div class="section-label">Sources &amp; Citation Verification</div>', unsafe_allow_html=True)
        for src in sources:
            render_source(src)

    if reason:
        st.markdown('<div class="section-label">Reasoning</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="reasoning-box">{reason}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Policy Assistant")
    st.markdown("---")

    # ── Stats ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sidebar-stat">
      <span class="label">Documents indexed</span>
      <span class="value">{len(st.session_state.indexed_files)}</span>
    </div>
    <div class="sidebar-stat">
      <span class="label">Total chunks</span>
      <span class="value">{st.session_state.total_chunks}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Upload Policy PDFs")

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supported: PDF only. Duplicates are skipped automatically.",
        label_visibility="collapsed",
    )

    # Deduplicate into buffer
    if uploaded:
        for f in uploaded:
            if not f.name.endswith(".pdf"):
                st.sidebar.warning(f"⚠️ Skipped '{f.name}' — only PDF files are supported.")
                continue
            st.session_state.upload_buffer[f.name] = f

    # Show what's in the buffer
    if st.session_state.upload_buffer:
        already_indexed = set(st.session_state.indexed_files)
        st.markdown(f"**{len(st.session_state.upload_buffer)} file(s) ready:**")
        for fname in st.session_state.upload_buffer:
            pill_cls = "file-pill-skip" if fname in already_indexed else "file-pill-new"
            icon     = "↩" if fname in already_indexed else "＋"
            st.markdown(
                f'<div class="file-pill {pill_cls}">{icon} {fname}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    replace_mode = st.checkbox(
        "Replace existing index",
        value=False,
        help="Tick to wipe the current index and rebuild from these files only.",
    )

    process_btn = st.button(
        "⚡ Process Documents",
        disabled=not st.session_state.upload_buffer,
    )

    # ── Indexing ──────────────────────────────────────────────────────────────
    if process_btn and st.session_state.upload_buffer:
        files_to_index = list(st.session_state.upload_buffer.values())

        with st.spinner("Loading & indexing documents…"):
            try:
                emb = get_embedding_model()
                vs  = reset_vector_store(emb) if replace_mode else get_vector_store(emb)

                # Incremental: skip already-indexed files unless replacing
                already = get_indexed_filenames(vs) if not replace_mode else set()
                new_files   = [f for f in files_to_index if f.name not in already]
                skipped     = [f.name for f in files_to_index if f.name in already]

                if skipped:
                    st.sidebar.info(f"Skipped (already indexed): {', '.join(skipped)}")

                if new_files:
                    upload_time = datetime.now(timezone.utc).isoformat()

                    docs   = load_pdfs_from_uploads(new_files)
                    chunks = chunk_documents(docs)
                    chunks = enrich_metadata(chunks, upload_time=upload_time)
                    count  = index_chunks(chunks, vs)

                    st.session_state.vector_store  = vs
                    st.session_state.agent         = build_agent(vs)
                    st.session_state.total_chunks  = get_total_chunks(vs)
                    st.session_state.indexed_files = list(
                        get_indexed_filenames(vs)
                    )
                    st.session_state.upload_buffer = {}
                    st.session_state.chat_history  = []

                    st.sidebar.success(
                        f"✅ Indexed {count} chunks from {len(new_files)} new file(s)."
                    )
                    st.rerun()
                else:
                    st.sidebar.warning("All selected files are already indexed.")

            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

    st.markdown("---")

    # ── Indexed file list ─────────────────────────────────────────────────────
    if st.session_state.indexed_files:
        st.markdown("### 📄 Indexed Documents")
        for fname in sorted(st.session_state.indexed_files):
            st.markdown(
                f'<div class="file-pill">📎 {fname}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("*No documents indexed yet.*")

    st.markdown("---")

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pa-header">
  <h1>📋 Enterprise Policy Assistant</h1>
  <p>Ask questions about your company policies — grounded answers with verified source citations.</p>
</div>
""", unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
if not st.session_state.agent:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">📂</div>
      <p>Upload your policy PDFs in the sidebar and click <strong>⚡ Process Documents</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_area(
            "Ask a policy question",
            placeholder=(
                "e.g. How many days of annual leave am I entitled to?\n"
                "Can I expense a business meal under $50 without a receipt?"
            ),
            height=90,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", use_container_width=True)

    if submitted and user_query.strip():
        with st.spinner("Retrieving policy context and generating answer…"):
            try:
                result, rewrite_meta = run_agent(st.session_state.agent, user_query.strip())

                # Citation verification
                raw_sources = result.get("sources", [])
                if raw_sources and st.session_state.vector_store:
                    result["sources"] = verify_citations(
                        raw_sources,
                        result.get("answer", ""),
                        st.session_state.vector_store,
                    )

                st.session_state.chat_history.insert(
                    0, {"question": user_query.strip(), "result": result, "rewrite_meta": rewrite_meta}
                )
            except Exception as e:
                st.error(f"Agent error: {e}")

    # ── Latest result ─────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[0]
        st.markdown(
            f'<div class="chat-question">🙋 {latest["question"]}</div>',
            unsafe_allow_html=True,
        )
        render_rewrite_panel(latest.get("rewrite_meta", {}))
        render_result(latest["result"])

        # ── History ───────────────────────────────────────────────────────────
        if len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown("#### Previous questions")
            for entry in st.session_state.chat_history[1:]:
                with st.expander(f"🙋 {entry['question']}"):
                    render_result(entry["result"])

"""
Streamlit UI — Knowledge Assistant

Features:
- Clean, professional design (no emoji clutter)
- Streaming responses via st.write_stream
- Document upload with summary generation
- Session metrics panel
- Session timeout
"""
import time
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import process_query_stream, ingest_document
from agents.generator import format_citations
from vectordb.milvus_client import get_milvus_client, reset_milvus_client
from vectordb.doc_registry import list_documents as list_registered_docs, clear_registry
from metrics import SessionMetrics
from utils import validate_api_key
from config import config

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS — minimal, professional, dark-friendly
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Clean header typography */
    .main-header {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        opacity: 0.6;
        margin-bottom: 1.5rem;
    }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .intent-retrieval { background: #1a3a4a; color: #5bb8d4; }
    .intent-summary { background: #2a3a1a; color: #8bc34a; }
    .intent-comparison { background: #3a2a1a; color: #ffb74d; }
    .intent-synthesis { background: #2a1a3a; color: #ce93d8; }
    .intent-direct { background: #1a2a3a; color: #90caf9; }
    .intent-clarify { background: #3a3a1a; color: #fff176; }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.6rem 0.75rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .metric-label {
        font-size: 0.7rem;
        opacity: 0.5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Doc card */
    .doc-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .doc-name {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }
    .doc-meta {
        font-size: 0.75rem;
        opacity: 0.5;
    }

    /* Security note on landing */
    .security-note {
        font-size: 0.8rem;
        opacity: 0.5;
        margin-top: 1rem;
        text-align: center;
        line-height: 1.6;
    }

    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Clean sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.5;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "messages": [],
        "documents_loaded": False,
        "api_key": "",
        "authenticated": False,
        "metrics": SessionMetrics(),
        "session_start": time.time(),
        "last_activity": time.time(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_session_timeout():
    if not st.session_state.authenticated:
        return False
    elapsed = (time.time() - st.session_state.last_activity) / 60
    if elapsed > config.session_timeout_minutes:
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.messages = []
        st.session_state.metrics = SessionMetrics()
        return True
    return False


def touch_activity():
    st.session_state.last_activity = time.time()


# ---------------------------------------------------------------------------
# Landing Page
# ---------------------------------------------------------------------------

def render_landing_page():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="main-header">Knowledge Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Document Q&A — powered by Gemini</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown("#### Enter your Gemini API Key")
        st.caption(
            "Your key stays in memory for this session only. "
            "Never stored on the server."
        )

        api_key_input = st.text_input(
            "Gemini API Key", type="password",
            placeholder="AIza...",
            help="Get your key at https://aistudio.google.com/apikey",
            label_visibility="collapsed",
        )

        if st.button("Start Session", use_container_width=True, type="primary"):
            if not api_key_input:
                st.error("Please enter your Gemini API key.")
            elif not validate_api_key(api_key_input):
                st.error("Invalid key format. Gemini keys start with 'AIza' and are 39 characters.")
            else:
                st.session_state.api_key = api_key_input
                st.session_state.authenticated = True
                st.session_state.session_start = time.time()
                st.session_state.last_activity = time.time()
                st.session_state.metrics = SessionMetrics()
                st.rerun()

        st.divider()

        st.markdown(
            '<div class="security-note">'
            'Your API key stays in browser session memory only.<br/>'
            'Not stored in any database, file, or log.<br/>'
            'Session ends when you close this tab.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.expander("How to get a Gemini API key"):
            st.markdown(
                "1. Go to [Google AI Studio](https://aistudio.google.com/apikey)\n"
                "2. Sign in with your Google account\n"
                "3. Click **Create API key**\n"
                "4. Copy and paste above"
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="main-header">Knowledge Assistant</div>', unsafe_allow_html=True)

        session_min = int((time.time() - st.session_state.session_start) / 60)
        st.caption(f"Session: {session_min}m")

        if st.button("Change API Key", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = ""
            st.session_state.messages = []
            st.session_state.metrics = SessionMetrics()
            st.rerun()

        st.divider()

        # Upload
        st.markdown("### Upload")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "xlsx", "pptx", "txt"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.max_upload_size_mb:
                st.error(f"File too large ({file_size_mb:.1f} MB). Max: {config.max_upload_size_mb} MB")
            else:
                with st.spinner("Processing..."):
                    temp_path = Path("data/uploads") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    try:
                        stats = ingest_document(
                            str(temp_path),
                            api_key=st.session_state.api_key,
                            metrics=st.session_state.metrics,
                        )
                        st.success(f"Indexed {stats['chunks_inserted']} chunks")
                        if stats.get("summary"):
                            st.info(stats['summary'][:200])
                        st.session_state.documents_loaded = True
                        touch_activity()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        # Documents
        st.markdown("### Documents")
        docs = list_registered_docs()
        if docs:
            for doc in docs:
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-name">{doc.filename}</div>'
                    f'<div style="font-size:0.8rem;opacity:0.7;margin-bottom:4px">{doc.summary[:120]}...</div>'
                    f'<div class="doc-meta">{doc.chunk_count} chunks · {doc.total_chars:,} chars</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents uploaded yet.")

        st.divider()

        # Metrics
        st.markdown("### Metrics")
        m = st.session_state.metrics.get_summary()

        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-card"><div class="metric-value">{m["queries"]}</div><div class="metric-label">Queries</div></div>'
            f'<div class="metric-card"><div class="metric-value">{m["docs_uploaded"]}</div><div class="metric-label">Docs</div></div>'
            f'<div class="metric-card"><div class="metric-value">{m["total_tokens"]:,}</div><div class="metric-label">Tokens</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.caption(f"LLM: {m['llm_input_tokens']:,} in / {m['llm_output_tokens']:,} out")
        st.caption(f"Embed: {m['embed_tokens']:,} tokens · {m['embed_calls']} calls")
        st.caption(f"Est. cost: ${m['est_cost_usd']:.4f}")

        st.divider()

        # Actions
        st.markdown("### Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Samples", use_container_width=True):
                sample_dir = Path("data/samples")
                if sample_dir.exists():
                    loaded = 0
                    for f in sample_dir.iterdir():
                        if f.suffix.lower() in [".pdf", ".docx", ".xlsx", ".pptx", ".txt"]:
                            try:
                                ingest_document(str(f), api_key=st.session_state.api_key, metrics=st.session_state.metrics)
                                loaded += 1
                            except Exception:
                                pass
                    if loaded > 0:
                        st.success(f"Loaded {loaded}")
                        st.session_state.documents_loaded = True
                        st.rerun()
        with col2:
            if st.button("Reset DB", type="secondary", use_container_width=True):
                try:
                    reset_milvus_client()
                    clear_registry()
                    st.session_state.messages = []
                    st.session_state.documents_loaded = False
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

INTENT_LABELS = {
    "retrieval": ("Retrieval", "intent-retrieval"),
    "summary": ("Summary", "intent-summary"),
    "comparison": ("Comparison", "intent-comparison"),
    "synthesis": ("Synthesis", "intent-synthesis"),
    "direct": ("Direct", "intent-direct"),
    "clarify": ("Clarify", "intent-clarify"),
}


def render_chat():
    st.markdown('<div class="main-header">Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your uploaded documents</div>', unsafe_allow_html=True)

    st.divider()

    # Empty state
    if not st.session_state.messages:
        st.markdown(
            "**Upload a document** in the sidebar, then try:\n"
            "- *\"Summarize this document\"*\n"
            "- *\"How does X relate to Y?\"*\n"
            "- *\"Compare sections A and B\"*"
        )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant":
                intent_val = msg.get("intent", "")
                if intent_val:
                    label, css_class = INTENT_LABELS.get(intent_val, (intent_val, "intent-retrieval"))
                    st.markdown(f'<span class="intent-badge {css_class}">{label}</span>', unsafe_allow_html=True)

                if msg.get("citations"):
                    with st.expander("Sources"):
                        for i, cit in enumerate(msg["citations"], 1):
                            parts = [cit["source_file"]]
                            if cit.get("page_number"):
                                parts.append(f"p.{cit['page_number']}")
                            if cit.get("section"):
                                parts.append(cit["section"])
                            st.caption(f"{i}. {' · '.join(parts)}")

    # Chat input — streaming
    if prompt := st.chat_input("Ask a question..."):
        touch_activity()
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                stream_ctx, text_stream = process_query_stream(
                    prompt,
                    api_key=st.session_state.api_key,
                    metrics=st.session_state.metrics,
                    chat_history=st.session_state.messages,
                )

                # Intent badge
                intent_val = stream_ctx.intent.value
                label, css_class = INTENT_LABELS.get(intent_val, (intent_val, "intent-retrieval"))
                st.markdown(f'<span class="intent-badge {css_class}">{label}</span>', unsafe_allow_html=True)

                if stream_ctx.sub_queries:
                    st.caption(f"Sub-queries: {', '.join(stream_ctx.sub_queries)}")

                # Stream the response
                full_text = st.write_stream(text_stream)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "citations": [
                        {
                            "source_file": c.source_file,
                            "page_number": c.page_number,
                            "section": c.section,
                            "excerpt": c.excerpt,
                        }
                        for c in stream_ctx.citations
                    ],
                    "intent": intent_val,
                })

                # Show sources after streaming completes
                if stream_ctx.citations:
                    with st.expander("Sources"):
                        for i, c in enumerate(stream_ctx.citations, 1):
                            parts = [c.source_file]
                            if c.page_number:
                                parts.append(f"p.{c.page_number}")
                            if c.section:
                                parts.append(c.section)
                            st.caption(f"{i}. {' · '.join(parts)}")

            except Exception as e:
                error_msg = str(e)
                if "api" in error_msg.lower() and "key" in error_msg.lower():
                    display_error = "API key error. Please check your key."
                else:
                    display_error = f"Error: {error_msg[:200]}"
                st.error(display_error)
                st.session_state.messages.append({
                    "role": "assistant", "content": display_error, "citations": [],
                })


def main():
    init_session_state()
    if check_session_timeout():
        st.warning("Session timed out. Please re-enter your API key.")
    if not st.session_state.authenticated:
        render_landing_page()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()

"""
Streamlit UI — Knowledge Assistant

Features:
- Landing page with Gemini API key input
- Document upload with summary generation
- Chat-based Q&A with 6 intent types
- Session timeout and metrics tracking
- Per-session API key (never stored server-side)
"""
import time
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import process_query, ingest_document
from vectordb.milvus_client import get_milvus_client, reset_milvus_client
from vectordb.doc_registry import list_documents as list_registered_docs, clear_registry
from metrics import SessionMetrics
from utils import validate_api_key
from config import config

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "metrics" not in st.session_state:
        st.session_state.metrics = SessionMetrics()
    if "session_start" not in st.session_state:
        st.session_state.session_start = time.time()
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()


def check_session_timeout():
    """Check if session has timed out. Returns True if expired."""
    if not st.session_state.authenticated:
        return False
    elapsed_minutes = (time.time() - st.session_state.last_activity) / 60
    if elapsed_minutes > config.session_timeout_minutes:
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.messages = []
        st.session_state.metrics = SessionMetrics()
        return True
    return False


def touch_activity():
    """Update last activity timestamp."""
    st.session_state.last_activity = time.time()


def render_landing_page():
    """Render the API key entry landing page."""
    st.markdown(
        """
        <style>
        .security-note {
            font-size: 0.85rem;
            color: #6c757d;
            margin-top: 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("## 🧠 Knowledge Assistant")
        st.markdown("*Document Q&A powered by Gemini*")

        st.divider()

        st.markdown("### Enter your Gemini API Key")
        st.caption(
            "Your API key is used only in-memory for this session. "
            "It is never stored on the server or logged."
        )

        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Get your key at https://aistudio.google.com/apikey",
            label_visibility="collapsed",
        )

        if st.button("Start Session", use_container_width=True, type="primary"):
            if not api_key_input:
                st.error("Please enter your Gemini API key.")
            elif not validate_api_key(api_key_input):
                st.error(
                    "Invalid API key format. Gemini keys start with 'AIza' "
                    "and are 39 characters long."
                )
            else:
                st.session_state.api_key = api_key_input
                st.session_state.authenticated = True
                st.session_state.session_start = time.time()
                st.session_state.last_activity = time.time()
                st.session_state.metrics = SessionMetrics()
                st.rerun()

        st.divider()

        st.markdown(
            """
            <div class="security-note">
            🔒 <strong>Security:</strong> Your API key stays in your browser session memory only.<br/>
            It is not stored in any database, file, or log on the server.<br/>
            The session ends when you close this tab.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("How to get a Gemini API key"):
            st.markdown(
                """
                1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
                2. Sign in with your Google account
                3. Click **Create API key**
                4. Copy the key and paste it above
                
                The free tier includes generous usage limits.
                """
            )


def render_sidebar():
    """Render the sidebar with documents, metrics, and actions."""
    with st.sidebar:
        st.header("🧠 Knowledge Assistant")
        st.caption("Document AI")

        # Session info
        session_minutes = int((time.time() - st.session_state.session_start) / 60)
        st.caption(f"⏱️ Session: {session_minutes}m")

        if st.button("🔑 Change API Key", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = ""
            st.session_state.messages = []
            st.session_state.metrics = SessionMetrics()
            st.rerun()

        st.divider()

        # Upload
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "xlsx", "pptx", "txt"],
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.max_upload_size_mb:
                st.error(
                    f"File too large ({file_size_mb:.1f} MB). "
                    f"Max: {config.max_upload_size_mb} MB"
                )
            else:
                with st.spinner("Processing & generating summary..."):
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
                        st.success(f"✅ Indexed {stats['chunks_inserted']} chunks")
                        if stats.get("summary"):
                            st.info(f"📝 **Summary:** {stats['summary'][:200]}...")
                        st.session_state.documents_loaded = True
                        touch_activity()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        # Registered documents
        st.subheader("📚 Documents")
        docs = list_registered_docs()
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc.filename}"):
                    st.write(f"**Summary:** {doc.summary}")
                    st.caption(f"Topics: {doc.topics}")
                    st.caption(f"Chunks: {doc.chunk_count} | Chars: {doc.total_chars:,}")
        else:
            st.info("No documents uploaded yet")

        st.divider()

        # Metrics panel
        st.subheader("📊 Session Metrics")
        m = st.session_state.metrics
        summary = m.get_summary()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", summary["queries"])
            st.metric("LLM Calls", summary["llm_calls"])
        with col2:
            st.metric("Docs", summary["docs_uploaded"])
            st.metric("Embed Calls", summary["embed_calls"])

        st.caption(f"🔤 LLM Input: {summary['llm_input_tokens']:,} tokens")
        st.caption(f"🔤 LLM Output: {summary['llm_output_tokens']:,} tokens")
        st.caption(f"🔤 Embed: {summary['embed_tokens']:,} tokens")
        st.caption(f"💰 Est. Cost: ${summary['est_cost_usd']:.4f}")

        st.divider()

        # Actions
        st.subheader("⚡ Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Load Samples", use_container_width=True):
                sample_dir = Path("data/samples")
                if sample_dir.exists():
                    loaded = 0
                    for f in sample_dir.iterdir():
                        if f.suffix.lower() in [".pdf", ".docx", ".xlsx", ".pptx", ".txt"]:
                            try:
                                ingest_document(
                                    str(f),
                                    api_key=st.session_state.api_key,
                                    metrics=st.session_state.metrics,
                                )
                                loaded += 1
                            except Exception:
                                pass
                    if loaded > 0:
                        st.success(f"Loaded {loaded}")
                        st.session_state.documents_loaded = True
                        st.rerun()
                    else:
                        st.warning("No samples")

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


def render_chat():
    """Render the main chat interface."""
    st.header("Knowledge Assistant")
    st.write("Ask questions about your uploaded documents.")

    # Intent badges
    intent_icons = {
        "retrieval": "🔍",
        "summary": "📝",
        "comparison": "⚖️",
        "synthesis": "🔗",
        "direct": "💡",
        "clarify": "❓",
    }

    st.divider()

    # Empty state
    if not st.session_state.messages:
        st.info(
            "**Getting Started:** Upload documents in the sidebar, "
            "then ask questions below.\n\n"
            "**Try:** \"Summarize this document\", \"How does X relate to Y?\", "
            "\"Compare sections A and B\""
        )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant":
                intent_val = msg.get("intent", "")
                if intent_val:
                    icon = intent_icons.get(intent_val, "")
                    st.caption(f"{icon} {intent_val.replace('_', ' ').title()}")

                if msg.get("citations"):
                    with st.expander("View Sources"):
                        for i, cit in enumerate(msg["citations"], 1):
                            meta = []
                            if cit.get("page_number"):
                                meta.append(f"Page {cit['page_number']}")
                            if cit.get("section"):
                                meta.append(cit["section"])
                            source_text = f"**{i}. {cit['source_file']}**"
                            if meta:
                                source_text += f" — {' · '.join(meta)}"
                            st.write(source_text)

    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        touch_activity()
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = process_query(
                        prompt,
                        api_key=st.session_state.api_key,
                        metrics=st.session_state.metrics,
                        chat_history=st.session_state.messages,
                    )

                    intent_val = result.intent.value
                    icon = intent_icons.get(intent_val, "")
                    st.caption(f"{icon} {intent_val.replace('_', ' ').title()}")

                    if result.sub_queries:
                        st.caption(f"Analyzed as: {', '.join(result.sub_queries)}")

                    st.write(result.response)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.response,
                            "citations": [
                                {
                                    "source_file": c.source_file,
                                    "page_number": c.page_number,
                                    "section": c.section,
                                    "excerpt": c.excerpt,
                                }
                                for c in result.citations
                            ],
                            "intent": intent_val,
                        }
                    )

                except Exception as e:
                    error_msg = str(e)
                    if "api" in error_msg.lower() and "key" in error_msg.lower():
                        display_error = (
                            "API key error. Please check your key is valid and try again."
                        )
                    else:
                        display_error = f"Error: {error_msg[:200]}"

                    st.error(display_error)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": display_error,
                            "citations": [],
                        }
                    )


def main():
    init_session_state()

    # Check session timeout
    if check_session_timeout():
        st.warning("⏰ Session timed out. Please re-enter your API key.")

    if not st.session_state.authenticated:
        render_landing_page()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()

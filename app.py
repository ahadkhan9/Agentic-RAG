"""
Streamlit UI — Knowledge Assistant

Features:
- Landing page with Gemini API key input
- Document upload and ingestion
- Chat-based Q&A with citations
- Per-session API key (never stored server-side)
"""
import streamlit as st
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import process_query, ingest_document
from vectordb.milvus_client import get_milvus_client, reset_milvus_client
from utils import validate_api_key
from config import config

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="⚙️",
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


def render_landing_page():
    """Render the API key entry landing page."""
    st.markdown(
        """
        <style>
        .landing-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4rem 1rem;
        }
        .landing-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .landing-subtitle {
            font-size: 1.1rem;
            color: #888;
            margin-bottom: 2rem;
        }
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
        st.markdown("## ⚙️ Knowledge Assistant")
        st.markdown("*Manufacturing Document Q&A powered by Gemini*")

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
                
                The free tier includes generous usage limits for testing.
                """
            )


def render_sidebar():
    """Render the sidebar with document management."""
    with st.sidebar:
        st.header("⚙️ Knowledge Assistant")
        st.caption("Manufacturing AI")

        # Logout button
        if st.button("🔑 Change API Key", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = ""
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "xlsx", "pptx", "txt"],
        )

        if uploaded_file:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > config.max_upload_size_mb:
                st.error(
                    f"File too large ({file_size_mb:.1f} MB). "
                    f"Max: {config.max_upload_size_mb} MB"
                )
            else:
                with st.spinner("Processing..."):
                    temp_path = Path("data/uploads") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    try:
                        stats = ingest_document(
                            str(temp_path), api_key=st.session_state.api_key
                        )
                        st.success(f"Indexed {stats['chunks_inserted']} chunks")
                        st.session_state.documents_loaded = True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        # Database status
        st.subheader("Database Status")
        try:
            client = get_milvus_client(api_key=st.session_state.api_key)
            stats = client.get_collection_stats()
            if stats.get("exists"):
                count = stats.get("num_entities", 0)
                st.metric(label="Chunks Indexed", value=f"{count:,}")
            else:
                st.info("No documents indexed yet")
        except Exception:
            st.warning("Database unavailable")

        st.divider()

        # Actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Load Samples", use_container_width=True):
                sample_dir = Path("data/samples")
                if sample_dir.exists():
                    loaded = 0
                    for f in sample_dir.iterdir():
                        if f.suffix.lower() in [
                            ".pdf", ".docx", ".xlsx", ".pptx", ".txt",
                        ]:
                            try:
                                ingest_document(
                                    str(f), api_key=st.session_state.api_key
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
                    st.session_state.messages = []
                    st.session_state.documents_loaded = False
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


def render_chat():
    """Render the main chat interface."""
    st.header("Manufacturing Knowledge Assistant")
    st.write("Ask questions about your manufacturing documents.")

    st.divider()

    # Empty state
    if not st.session_state.messages:
        st.info(
            "**Getting Started:** Upload documents in the sidebar, "
            "then ask questions below."
        )

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant" and msg.get("citations"):
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
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = process_query(
                        prompt, api_key=st.session_state.api_key
                    )

                    # Intent badge
                    intent = result.intent.value.replace("_", " ").title()
                    st.caption(f"Query type: {intent}")

                    # Sub-queries
                    if result.sub_queries:
                        st.caption(
                            f"Analyzed as: {', '.join(result.sub_queries)}"
                        )

                    # Response
                    st.write(result.response)

                    # Save
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
                            "intent": result.intent.value,
                        }
                    )

                except Exception as e:
                    error_msg = str(e)
                    # Don't leak API key details in error messages
                    if "api" in error_msg.lower() and "key" in error_msg.lower():
                        display_error = (
                            "API key error. Please check your key is valid "
                            "and try again."
                        )
                    else:
                        display_error = "Could not process your question."

                    st.error(display_error)
                    st.caption(f"Error: {error_msg[:200]}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": display_error,
                            "citations": [],
                        }
                    )


def main():
    init_session_state()

    # Show landing page or main app
    if not st.session_state.authenticated:
        render_landing_page()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()

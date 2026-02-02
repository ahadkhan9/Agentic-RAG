"""
Streamlit UI - Manufacturing Knowledge Assistant
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from agents.orchestrator import process_query, ingest_document
from vectordb.milvus_client import get_milvus_client, reset_milvus_client

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Knowledge Assistant")
        st.caption("Manufacturing AI")

        st.divider()

        # Upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "xlsx", "pptx", "txt"]
        )

        if uploaded_file:
            with st.spinner("Processing..."):
                temp_path = Path("data/uploads") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                try:
                    stats = ingest_document(str(temp_path))
                    st.success(f"Indexed {stats['chunks_inserted']} chunks")
                    st.session_state.documents_loaded = True
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        # Database
        st.subheader("Database Status")
        try:
            client = get_milvus_client()
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
                        if f.suffix.lower() in ['.pdf', '.docx', '.xlsx', '.pptx', '.txt']:
                            try:
                                ingest_document(str(f))
                                loaded += 1
                            except:
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
    st.header("Manufacturing Knowledge Assistant")
    st.write("Ask questions about your manufacturing documents.")

    st.divider()

    # Empty state
    if not st.session_state.messages:
        st.info("**Getting Started:** Upload documents in the sidebar, then ask questions below.")

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
                    result = process_query(prompt)

                    # Intent badge
                    intent = result.intent.value.replace("_", " ").title()
                    st.caption(f"Query type: {intent}")

                    # Sub-queries
                    if result.sub_queries:
                        st.caption(f"Analyzed as: {', '.join(result.sub_queries)}")

                    # Response
                    st.write(result.response)

                    # Save
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.response,
                        "citations": [
                            {
                                "source_file": c.source_file,
                                "page_number": c.page_number,
                                "section": c.section,
                                "excerpt": c.excerpt
                            }
                            for c in result.citations
                        ],
                        "intent": result.intent.value
                    })

                except Exception as e:
                    st.error("Could not process your question.")
                    st.caption(f"Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Could not process your question.",
                        "citations": []
                    })


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

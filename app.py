"""
app.py - Main Streamlit UI for Obsidian Vault Knowledge Assistant.

Integrates rag_engine.py and llm_service.py into a responsive, single-page chat application
with file uploaders, sample vault loading, persistent state, and chunk citation previews.
"""

import os
import glob
import streamlit as st
from rag_engine import RAGEngine
from llm_service import LLMService, get_groq_api_key

# Page configuration
st.set_page_config(
    page_title="Obsidian Vault Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for visual excellence
st.markdown("""
<style>
    /* Main layout tuning */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Card style for sources */
    .source-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    
    .source-header {
        font-weight: 600;
        color: #64B5F6;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    
    .source-preview {
        font-size: 0.88rem;
        color: #CFD8DC;
        font-style: italic;
        line-height: 1.4;
    }

    .badge-similarity {
        background-color: #2E7D32;
        color: white;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        float: right;
    }

    /* Sidebar improvements */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sidebar-subtitle {
        font-size: 0.85rem;
        color: #90A4AE;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine() -> RAGEngine:
    return RAGEngine()


@st.cache_resource
def get_llm_service() -> LLMService:
    return LLMService()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vault_loaded" not in st.session_state:
        st.session_state.vault_loaded = False
    if "loaded_notes_count" not in st.session_state:
        st.session_state.loaded_notes_count = 0
    if "loaded_chunks_count" not in st.session_state:
        st.session_state.loaded_chunks_count = 0
    if "loaded_filenames" not in st.session_state:
        st.session_state.loaded_filenames = []


def load_sample_vault():
    """Loads bundled sample_vault/*.md files into ChromaDB."""
    rag_engine = get_rag_engine()
    sample_files = glob.glob("sample_vault/*.md")
    
    if not sample_files:
        st.error("No sample Markdown files found in `sample_vault/` directory.")
        return

    files_dict = {}
    for filepath in sample_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            files_dict[filename] = f.read()

    progress_bar = st.sidebar.progress(0, text="Indexing sample vault notes...")
    
    def update_progress(percent):
        progress_bar.progress(percent, text=f"Indexing sample vault notes... ({percent}%)")

    total_notes, total_chunks = rag_engine.index_vault(files_dict, progress_callback=update_progress)
    progress_bar.empty()

    st.session_state.vault_loaded = True
    st.session_state.loaded_notes_count = total_notes
    st.session_state.loaded_chunks_count = total_chunks
    st.session_state.loaded_filenames = sorted(list(files_dict.keys()))
    
    st.sidebar.success(f"Successfully loaded sample vault! ({total_notes} notes, {total_chunks} chunks)")


def process_custom_uploads(uploaded_files):
    """Processes uploaded Markdown files into ChromaDB."""
    rag_engine = get_rag_engine()
    files_dict = {}

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.endswith(".md"):
            st.sidebar.error(f"Skipped '{uploaded_file.name}': Only `.md` Markdown files are supported.")
            continue
        
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        files_dict[uploaded_file.name] = content

    if not files_dict:
        st.sidebar.warning("No valid `.md` Markdown files selected.")
        return

    progress_bar = st.sidebar.progress(0, text="Embedding & indexing uploaded notes...")
    
    def update_progress(percent):
        progress_bar.progress(percent, text=f"Indexing notes... ({percent}%)")

    total_notes, total_chunks = rag_engine.index_vault(files_dict, progress_callback=update_progress)
    progress_bar.empty()

    st.session_state.vault_loaded = True
    st.session_state.loaded_notes_count = total_notes
    st.session_state.loaded_chunks_count = total_chunks
    st.session_state.loaded_filenames = sorted(list(files_dict.keys()))

    st.sidebar.success(f"Successfully indexed {total_notes} notes ({total_chunks} chunks)!")


def clear_vault():
    """Resets the vector database and clears chat session history."""
    rag_engine = get_rag_engine()
    rag_engine.clear_vault()
    
    st.session_state.messages = []
    st.session_state.vault_loaded = False
    st.session_state.loaded_notes_count = 0
    st.session_state.loaded_chunks_count = 0
    st.session_state.loaded_filenames = []
    st.sidebar.info("Vault database and chat history cleared.")


def render_sources_section(retrieved_chunks):
    """Renders formatted sources below assistant responses."""
    if not retrieved_chunks:
        return

    st.markdown("---")
    st.markdown("#### 📚 Sources & Citations")

    # Group by unique source filename for clean display
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "Unknown Note")
        heading = chunk.get("heading", "")
        heading_str = f" > {heading}" if heading else ""
        similarity = chunk.get("score", 0.0)
        full_text = chunk.get("text", "")
        preview_text = full_text[:150].replace("\n", " ") + ("..." if len(full_text) > 150 else "")

        with st.container():
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="badge-similarity">Similarity: {similarity:.2f}</span>
                    <div class="source-header">📄 [{idx}] {source}{heading_str}</div>
                    <div class="source-preview">"{preview_text}"</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            with st.expander(f"🔍 View full excerpt from `{source}`", expanded=False):
                st.markdown(f"**Section:** {heading or 'Main body'}")
                st.markdown(f"**Tags:** `{chunk.get('tags', 'none')}`")
                st.code(full_text, language="markdown")


def main():
    init_session_state()

    # --- SIDEBAR UI ---
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 Knowledge Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Obsidian Vault RAG System</div>', unsafe_allow_html=True)
        st.markdown("---")

        # API Key warning if missing
        groq_key = get_groq_api_key()
        if not groq_key:
            st.warning("⚠️ `GROQ_API_KEY` not found in `.env` or `st.secrets`.")
        else:
            st.caption("🟢 Groq API Connected (`llama-3.1-8b-instant`)")

        st.subheader("📁 Vault Management")
        
        # Load Sample Vault Button
        if st.button("🚀 Load Sample Vault", use_container_width=True, help="Load bundled Generative AI demo notes"):
            load_sample_vault()

        st.markdown("**OR Upload Custom Vault (.md)**")
        uploaded_files = st.file_uploader(
            "Upload Markdown (.md) notes",
            type=["md"],
            accept_multiple_files=True,
            help="Select one or multiple Markdown files from your Obsidian vault"
        )
        
        if uploaded_files and st.button("📥 Index Uploaded Notes", use_container_width=True):
            process_custom_uploads(uploaded_files)

        st.markdown("---")

        # Vault statistics & status
        if st.session_state.vault_loaded and st.session_state.loaded_notes_count > 0:
            st.subheader("📊 Indexed Notes")
            st.caption(f"**Total Notes:** {st.session_state.loaded_notes_count} | **Total Chunks:** {st.session_state.loaded_chunks_count}")

            with st.expander(f"📋 Loaded Files ({st.session_state.loaded_notes_count})", expanded=False):
                for fname in st.session_state.loaded_filenames:
                    st.markdown(f"- `{fname}`")

            if st.button("🗑️ Clear Vault", use_container_width=True, type="secondary"):
                clear_vault()
                st.rerun()
        else:
            st.info("No vault loaded yet. Load the sample vault or upload `.md` files to begin.")

    # --- MAIN CONTENT AREA ---
    st.title("🧠 Obsidian Vault Knowledge Assistant")
    st.markdown("Ask natural language questions about your Obsidian notes and get factual answers with source citations.")

    # Banner warning if no vault loaded
    if not st.session_state.vault_loaded or st.session_state.loaded_notes_count == 0:
        st.info("👈 **Get Started**: Click **'🚀 Load Sample Vault'** in the sidebar to test immediately, or upload your own `.md` notes.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources_section(message["sources"])

    # Chat Input
    query = st.chat_input("Ask a question about your Obsidian notes...")

    if query:
        if not st.session_state.vault_loaded or st.session_state.loaded_notes_count == 0:
            st.warning("Please load the sample vault or upload Markdown notes in the sidebar before asking questions.")
            return

        # Render user message
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Process query with spinner
        with st.chat_message("assistant"):
            with st.spinner("Searching notes & generating answer..."):
                rag_engine = get_rag_engine()
                llm_service = get_llm_service()

                # Step 1: Retrieve relevant chunks
                retrieved_chunks = rag_engine.query_vault(query, top_k=4)

                # Step 2: Generate LLM answer grounded in context
                answer, cited_sources = llm_service.generate_answer(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    chat_history=st.session_state.messages
                )

                st.markdown(answer)

                if retrieved_chunks:
                    render_sources_section(retrieved_chunks)

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_chunks
        })


if __name__ == "__main__":
    main()

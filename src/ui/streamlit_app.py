"""Streamlit chat interface for the RAG chatbot.
Answers queries from pre-ingested documents in ChromaDB using Ollama.
Supports streaming responses and chat history persistence."""

import hashlib
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.config import settings
from src.document.processor import process_document
from src.document.registry import DocumentRegistry
from src.document.schemas import DocumentStatus
from src.embeddings.embedder import Embedder
from src.embeddings.store import ChromaStore
from src.llm.ollama_llm import OllamaLLM
from src.pdf.chunker import chunk_pages
from src.pdf.extractor import extract_pages
from src.pdf.ocr import extract_pages_with_fallback
from src.rag.chain import RAGChain
from src.utils.helpers import is_arabic_text

HISTORY_DIR = Path("data/chat_history")
FEEDBACK_DIR = Path("data/feedback")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "current_history_path" not in st.session_state:
        st.session_state.current_history_path = None
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {}
    if "query_document_id" not in st.session_state:
        st.session_state.query_document_id = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()
    if "_cleanup_done" not in st.session_state:
        st.session_state._cleanup_done = False


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> Embedder:
    return Embedder(settings.embeddings)


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_store() -> ChromaStore:
    return ChromaStore(settings.vector_store, load_embedder())


@st.cache_resource(show_spinner="Loading RAG pipeline (Ollama)...")
def load_rag_chain() -> RAGChain:
    if "rag_chain" in st.session_state:
        return st.session_state.rag_chain
    store = load_store()
    st.session_state.total_chunks = store.count_chunks()
    llm = OllamaLLM(**settings.llm.ollama.model_dump())
    chain = RAGChain(store=store, llm=llm)
    chain.rebuild_index()
    st.session_state.rag_chain = chain
    return chain


def render_text(text: str) -> str:
    if is_arabic_text(text):
        return f'<div dir="rtl" style="text-align: right; font-size: 1.1rem;">{text}</div>'
    return text


def _preview_text(text: str, max_chars: int = 120) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.6:
        return truncated[:last_space] + "..."
    return truncated.rstrip() + "..."


# ── Chat history helpers ──


def _list_sessions() -> list[Path]:
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("*.json"), reverse=True)


def _session_label(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data:
            first = data[0]
            q = first.get("content", "?")[:50]
            return f"{path.stem} — {q}"
    except Exception:
        pass
    return path.stem


def save_messages():
    if not st.session_state.messages:
        return
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = st.session_state.current_history_path
    if path is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = HISTORY_DIR / f"{ts}.json"
        st.session_state.current_history_path = path
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)


def load_messages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def new_conversation():
    save_messages()
    st.session_state.messages = []
    st.session_state.current_history_path = None
    st.session_state.feedback_given = {}
    st.rerun()


def delete_current_conversation():
    """Delete the currently selected conversation JSON file and reset chat state."""
    path = st.session_state.current_history_path
    if path is not None and path.exists():
        path.unlink()
    st.session_state.messages = []
    st.session_state.current_history_path = None
    st.session_state.feedback_given = {}
    st.rerun()


def clear_all_conversations():
    """Delete all conversation JSON files in the history directory."""
    if not HISTORY_DIR.exists():
        return
    for f in HISTORY_DIR.glob("*.json"):
        f.unlink()
    st.session_state.messages = []
    st.session_state.current_history_path = None
    st.session_state.feedback_given = {}
    st.rerun()


def _save_feedback(msg_idx: int, answer: str, feedback: str):
    """Write feedback to JSONL file."""
    msg = st.session_state.messages[msg_idx]
    user_msg = ""
    for m in reversed(st.session_state.messages[:msg_idx]):
        if m["role"] == "user":
            user_msg = m["content"]
            break
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    path = FEEDBACK_DIR / "feedback.jsonl"
    conv_id = (
        st.session_state.current_history_path.stem
        if st.session_state.current_history_path
        else "unsaved"
    )
    record = {
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conv_id,
        "question": user_msg,
        "answer": answer,
        "source_documents": msg.get("source_documents", {}),
        "feedback": feedback,
        "model": settings.llm.ollama.model,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Document management ──


DOCUMENTS_DB = Path("data/processed/documents.db")


@st.cache_resource
def get_registry() -> DocumentRegistry:
    return DocumentRegistry(DOCUMENTS_DB)


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def status_emoji(status: DocumentStatus) -> str:
    mapping = {
        DocumentStatus.UPLOADED: "🕐",
        DocumentStatus.PROCESSING: "⏳",
        DocumentStatus.INDEXED: "✅",
        DocumentStatus.FAILED: "❌",
    }
    return mapping.get(status, "❓")


def confirm_and_delete_document(doc_id: str):
    key = f"delete_confirm_{doc_id}"
    if st.session_state.get(key):
        registry = get_registry()
        store = load_store()
        doc = registry.get(doc_id)
        if doc:
            store.delete_by_document(doc_id)
            registry.delete(doc_id)
            chain = load_rag_chain()
            chain.rebuild_index()
            st.session_state.total_chunks = store.count_chunks()
            st.session_state.pop(key, None)
            st.success(f"Deleted `{doc.filename}`")
            st.rerun()
    else:
        st.session_state[key] = True
        st.rerun()


def reindex_document(doc_id: str):
    registry = get_registry()
    doc = registry.get(doc_id)
    if doc is None:
        st.error("Document not found in registry.")
        return
    raw_dir = Path(settings.vector_store.persist_directory).parent.parent / "raw"
    pdf_path = (raw_dir / doc.filename).resolve()
    if not pdf_path.exists():
        st.error(f"Source PDF `{doc.filename}` not found on disk.")
        return
    store = load_store()
    store.delete_by_document(doc_id)
    chain = load_rag_chain()
    chain.rebuild_index()
    registry.update_status(doc_id, DocumentStatus.UPLOADED)
    thread = threading.Thread(
        target=process_document,
        args=(doc_id, pdf_path, doc.filename, registry),
        daemon=True,
    )
    thread.start()
    st.success(f"Re-indexing `{doc.filename}` in the background...")
    st.rerun()


# ── Main ──


def main():
    st.set_page_config(
        page_title=settings.ui.page_title,
        page_icon=settings.ui.page_icon,
        layout=settings.ui.layout,
    )

    st.title(f"{settings.ui.page_icon} {settings.ui.page_title}")
    st.markdown("Ask questions about your pre-loaded PDF documents.")

    init_session_state()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Status")

        try:
            chain = load_rag_chain()
            embedder = load_embedder()
            store = load_store()
            total = st.session_state.total_chunks
            st.success(f"Ready — {total} document chunks loaded")
            st.metric("Chunks in database", total)
        except Exception as e:
            st.error(f"Failed to load RAG pipeline: {e}")
            st.info(
                "1. Place PDFs in `data/raw/`\n"
                "2. Run `python scripts/ingest.py`\n"
                "3. Restart this app"
            )
            st.stop()

        st.divider()

        # ── Chat history selector ──
        st.header("Conversations")
        col1, col2 = st.columns([3, 1])
        with col1:
            sessions = _list_sessions()
            labels = [_session_label(s) for s in sessions]
            if sessions:
                selected_label = st.selectbox(
                    "Load previous",
                    ["(current)"] + labels,
                    label_visibility="collapsed",
                )
                if selected_label != "(current)":
                    idx = labels.index(selected_label)
                    try:
                        st.session_state.messages = load_messages(sessions[idx])
                        st.session_state.current_history_path = sessions[idx]
                        st.session_state.feedback_given = {}
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Failed to load: {ex}")
        with col2:
            st.button("New", on_click=new_conversation)

        if st.session_state.current_history_path:
            if st.button("Delete current", type="secondary", use_container_width=True):
                delete_current_conversation()

        # Clear all conversations with confirmation
        clear_key = "_confirm_clear_all"
        if st.session_state.get(clear_key):
            st.warning("This will permanently delete all saved conversations.")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Yes, clear all", type="primary", use_container_width=True):
                    clear_all_conversations()
            with col_b:
                if st.button("Cancel", use_container_width=True):
                    st.session_state[clear_key] = False
                    st.rerun()
        else:
            if st.button("Clear all conversations", use_container_width=True):
                st.session_state[clear_key] = True
                st.rerun()

        st.caption(
            "Chat history is saved locally under `data/chat_history/` and is ignored by Git."
        )

        st.divider()

        # ── PDF upload ──
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Add a PDF to the knowledge base",
            type="pdf",
            label_visibility="collapsed",
        )

        # ── Session cleanup (demo mode) ──
        if settings.ui.clear_documents_when_no_upload:
            if uploaded_file is None:
                if not st.session_state.get("_cleanup_done"):
                    store.clear()
                    get_registry().clear()
                    if "rag_chain" in st.session_state:
                        st.session_state.rag_chain.rebuild_index()
                    st.session_state.total_chunks = 0
                    st.session_state.query_document_id = None
                    st.session_state.uploaded_files = set()
                    st.session_state._cleanup_done = True
                    st.rerun()
            else:
                st.session_state._cleanup_done = False

        if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_files:
            raw_dir = Path(settings.vector_store.persist_directory).parent.parent / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            save_path = raw_dir / uploaded_file.name

            # Calculate SHA256 hash for duplicate detection
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            # Check registry for existing document with same hash
            registry = get_registry()
            existing = registry.get_by_hash(file_hash)
            if existing and existing.status in (
                DocumentStatus.INDEXED,
                DocumentStatus.PROCESSING,
            ):
                st.info(
                    f"`{uploaded_file.name}` already ingested (ID: {existing.document_id[:8]}...)"
                )
                st.session_state.uploaded_files.add(uploaded_file.name)
            else:
                # Save the file
                save_path.write_bytes(file_bytes)

                # Create registry entry
                doc = registry.create(
                    filename=uploaded_file.name,
                    file_size_bytes=len(file_bytes),
                    file_hash=file_hash,
                )
                registry.update_status(doc.document_id, DocumentStatus.PROCESSING)

                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    try:
                        pages = extract_pages_with_fallback(
                            save_path,
                            text_extract_fn=extract_pages,
                            lang=settings.pdf.ocr_language,
                            ocr_threshold=settings.pdf.ocr_threshold_chars,
                        )
                        ocr_note = ""
                        if pages and pages[0].text.startswith("[OCR]"):
                            ocr_note = " (via OCR)"
                        chunks = chunk_pages(
                            pages,
                            settings.pdf,
                            source_file=save_path.name,
                            embedder=embedder,
                            pdf_path=save_path,
                        )
                        # Tag chunks with document_id before storing
                        now_iso = datetime.now(UTC).isoformat()
                        embedding_model = settings.embeddings.model_name
                        for c in chunks:
                            c.document_id = doc.document_id
                            c.original_text = c.text
                            c.processed_text = c.text
                            c.embedding_model = embedding_model
                            c.created_at = now_iso

                        store.add_chunks(chunks)
                        st.session_state.total_chunks = store.count_chunks()
                        if "rag_chain" in st.session_state:
                            st.session_state.rag_chain.rebuild_index()

                        registry.update_status(
                            doc.document_id,
                            DocumentStatus.INDEXED,
                            num_pages=len(pages),
                            num_chunks=len(chunks),
                        )

                        text_count = len([c for c in chunks if c.chunk_type == "text"])
                        table_count = len([c for c in chunks if c.chunk_type == "table"])
                        chart_count = len([c for c in chunks if c.chunk_type == "chart"])
                        parts = [f"{text_count} text"]
                        if table_count:
                            parts.append(f"{table_count} tables")
                        if chart_count:
                            parts.append(f"{chart_count} charts")

                        # Mark success (rendered after spinner exits)
                        st.session_state._ingest_success = (
                            f"Ingested {len(chunks)} chunks{ocr_note} from `{uploaded_file.name}` "
                            f"({', '.join(parts)})"
                        )
                        st.session_state._ingest_filename = uploaded_file.name
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")
                        registry.update_status(
                            doc.document_id, DocumentStatus.FAILED, error_message=str(e)
                        )
                        save_path.unlink(missing_ok=True)

                # Outside spinner — show success and trigger re-render
                if st.session_state.get("_ingest_success"):
                    st.success(st.session_state._ingest_success)
                    st.session_state.uploaded_files.add(st.session_state._ingest_filename)
                    del st.session_state._ingest_success
                    del st.session_state._ingest_filename
                    st.rerun()

        st.divider()

        # ── Document management ──
        st.header("Documents")
        try:
            registry = get_registry()
            docs = registry.list()
            if not docs:
                st.caption("No documents uploaded yet.")
            else:
                for doc in docs:
                    label = (
                        f"{status_emoji(doc.status)} {doc.filename}"
                        f" ({doc.num_chunks or '?'} chunks, "
                        f"{format_file_size(doc.file_size_bytes)})"
                    )
                    with st.container(border=True):
                        cols = st.columns([3, 1, 1])
                        with cols[0]:
                            st.caption(label)
                        with cols[1]:
                            if doc.status in (DocumentStatus.INDEXED, DocumentStatus.FAILED):
                                if st.button(
                                    "🔄", key=f"reindex_{doc.document_id}", help="Re-index"
                                ):
                                    reindex_document(doc.document_id)
                        with cols[2]:
                            if doc.status in (
                                DocumentStatus.INDEXED,
                                DocumentStatus.FAILED,
                                DocumentStatus.UPLOADED,
                            ):
                                confirm_key = f"delete_confirm_{doc.document_id}"
                                if st.session_state.get(confirm_key):
                                    if st.button(
                                        "🗑️",
                                        key=f"delete_do_{doc.document_id}",
                                        help="Confirm delete",
                                    ):
                                        confirm_and_delete_document(doc.document_id)
                                else:
                                    if st.button(
                                        "✕", key=f"delete_req_{doc.document_id}", help="Delete"
                                    ):
                                        confirm_and_delete_document(doc.document_id)
                        if doc.error_message:
                            with cols[0]:
                                st.caption(f"Error: {doc.error_message[:100]}")
        except Exception as e:
            st.error(f"Document registry error: {e}")

        # ── Document selector for chat ──
        st.divider()
        st.header("Query Scope")
        try:
            registry = get_registry()
            doc_list = registry.list()
            all_label = "All documents"
            labels = [all_label] + [
                f"{d.filename} ({d.document_id})"
                for d in doc_list
                if d.status == DocumentStatus.INDEXED
            ]
            current = st.session_state.query_document_id
            current_label = all_label
            if current:
                for d in doc_list:
                    if d.document_id == current:
                        current_label = f"{d.filename} ({d.document_id})"
                        break
            idx = labels.index(current_label) if current_label in labels else 0
            selected = st.selectbox(
                "Search in:",
                labels,
                index=idx,
                label_visibility="collapsed",
            )
            if selected == all_label:
                st.session_state.query_document_id = None
            else:
                st.session_state.query_document_id = selected.split("(")[-1].rstrip(")")
        except Exception:
            pass

        st.divider()
        st.caption(
            f"**Model:** {settings.llm.ollama.model}\n\n"
            f"**Embeddings:** {settings.embeddings.model_name.split('/')[-1]}\n\n"
            f"**Top-K:** {settings.retrieval.top_k}"
        )

    # ── Render chat messages with feedback buttons ──
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(render_text(msg["content"]), unsafe_allow_html=True)
            mc = msg.get("citations", [])
            dc = msg.get("debug_chunks", [])
            if mc or dc:
                with st.expander("Sources", expanded=False):
                    if mc:
                        st.caption("**Citations used in answer:**")
                        for cit in mc:
                            pages_str = ", ".join(str(p) for p in cit.get("pages", []))
                            tag = " ✅" if cit.get("verified") else " ⚠️ unverified"
                            st.caption(f"  page(s) {pages_str}{tag}")
                    if dc:
                        if mc:
                            st.divider()
                        st.caption("**Retrieved context:**")
                        for i, c in enumerate(dc, 1):
                            st.caption(
                                f"[{i}] `{c['source']}` p{c['page']} (score={c['score']:.2f})"
                            )
                            st.text(c["preview"])
            if msg["role"] == "assistant" and msg.get("source_documents") is not None:
                fb = st.session_state.feedback_given.get(idx)
                if fb:
                    st.caption(f"{'👍' if fb == 'up' else '👎'} Feedback recorded")
                else:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("👍", key=f"fb_up_{idx}"):
                            _save_feedback(idx, msg["content"], "up")
                            st.session_state.feedback_given[idx] = "up"
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"fb_down_{idx}"):
                            _save_feedback(idx, msg["content"], "down")
                            st.session_state.feedback_given[idx] = "down"
                            st.rerun()

    # ── Chat input ──
    last_content = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
    placeholder = (
        "اسأل سؤالاً عن مستنداتك..."
        if is_arabic_text(last_content)
        else "Ask a question about your documents..."
    )

    if prompt := st.chat_input(placeholder):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(render_text(prompt), unsafe_allow_html=True)

        # Streaming assistant response
        with st.chat_message("assistant"):
            placeholder_el = st.empty()

            # Pass previous exchanges as history (exclude current question)
            prev_messages = st.session_state.messages[:-1]

            selected_doc_id = st.session_state.query_document_id

            def token_generator():
                for event in chain.query_stream(
                    prompt, history=prev_messages, document_id=selected_doc_id
                ):
                    if event["type"] == "token":
                        yield event["content"]
                    elif event["type"] == "done":
                        # Store final result for rendering after stream
                        st.session_state._stream_result = event

            # Write the streaming tokens
            placeholder_el.write_stream(token_generator)
            full_answer = st.session_state._stream_result["answer"]
            source_docs = st.session_state._stream_result["source_documents"]
            raw_chunks = st.session_state._stream_result["chunks"]
            citations = st.session_state._stream_result.get("citations", [])
            del st.session_state._stream_result

            # Re-render with RTL support
            placeholder_el.markdown(render_text(full_answer), unsafe_allow_html=True)

            debug_chunks = []
            if raw_chunks and source_docs:
                with st.expander("Sources", expanded=False):
                    if citations:
                        st.caption("**Citations used in answer:**")
                        for cit in citations:
                            pages_str = ", ".join(str(p) for p in cit.get("pages", []))
                            tag = " ✅" if cit.get("verified") else " ⚠️ unverified"
                            st.caption(f"  page(s) {pages_str}{tag}")
                        st.divider()
                    st.caption("**Retrieved context:**")
                    debug_chunks = [
                        {
                            "source": c.chunk.source_file,
                            "page": c.chunk.page_num,
                            "score": round(c.score, 3),
                            "preview": _preview_text(c.chunk.text, 120),
                        }
                        for c in raw_chunks
                    ]
                    for i, c in enumerate(debug_chunks, 1):
                        st.caption(f"[{i}] `{c['source']}` p{c['page']} (score={c['score']:.2f})")
                        st.text(c["preview"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "source_documents": source_docs,
                    "debug_chunks": debug_chunks,
                    "citations": citations,
                }
            )
            save_messages()

    # ── Welcome message ──
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! I can answer questions based on your loaded PDF documents. "
                "I support both **Arabic** and **English**. Try asking something!"
            )
            st.info(
                f"I have **{total} document chunks** loaded. Ask me anything about their content!"
            )


if __name__ == "__main__":
    main()

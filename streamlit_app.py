# streamlit_app.py

from memory_extractor import extract_memory
from database import init_db, save_message, get_chat_history, clear_chat
from multi_doc_db import (
    init_multi_doc_db,
    save_multi_doc_message,
    get_multi_doc_history,
    clear_multi_doc_chat
)

init_multi_doc_db()
init_db()

from memory_db import init_memory_db
init_memory_db()

import os
import json
import streamlit as st


st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG AI Assistant")
st.caption("Document Question Answering using RAG + Gemini + FAISS")


# ---------------------------
# Session Variables
# ---------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# UI-only chat history
if "multi_doc_ui_history" not in st.session_state:
    st.session_state.multi_doc_ui_history = None

if "single_doc_ui_history" not in st.session_state:
    st.session_state.single_doc_ui_history = None


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("💬 Chat Sessions")

if st.sidebar.button("➕ New Chat"):
    chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat = chat_id

for chat_id in st.session_state.chat_sessions.keys():
    if st.sidebar.button(chat_id):
        st.session_state.current_chat = chat_id

if st.session_state.current_chat:
    st.sidebar.success(f"Current Chat: {st.session_state.current_chat}")

st.sidebar.divider()
st.sidebar.subheader("⚙ Controls")

if st.sidebar.button("Clear Document Chat"):
    if st.session_state.current_chat:
        st.session_state.single_doc_ui_history = []
        st.rerun()


# ---------------------------
# Load Multi-Doc QA System
# ---------------------------
@st.cache_resource
def load_qa_system():
    from retrieval_pipeline import ask_question
    return ask_question


ask_question = load_qa_system()


# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs([
    "📚 Multi Document QA",
    "📄 Single Document Chat"
])


# =====================================================
# MULTI DOCUMENT RAG
# =====================================================
with tab1:

    st.subheader("Knowledge Base Chat")

    multi_chat_id = "multi_doc_chat"

    # Load DB history only once
    if st.session_state.multi_doc_ui_history is None:
        st.session_state.multi_doc_ui_history = get_multi_doc_history(multi_chat_id)

    history = st.session_state.multi_doc_ui_history

    chat_container = st.container()

    with chat_container:
        for role, message in history:
            with st.chat_message(role):
                st.write(message)

    formatted_history = "\n".join(
        f"{role}: {message}" for role, message in history
    )

    query = st.chat_input("Ask a question about your documents")

    if query:

        save_multi_doc_message(multi_chat_id, "user", query)
        extract_memory(query)

        st.session_state.multi_doc_ui_history.append(("user", query))

        with chat_container:
            with st.chat_message("user"):
                st.write(query)

        with st.spinner("🔎 Searching knowledge base..."):
            answer, docs = ask_question(query, formatted_history)

        with chat_container:
            with st.chat_message("assistant"):
                st.write(answer)

                if docs:
                    with st.expander("Sources"):
                        for doc in docs:
                            st.write(
                                os.path.basename(
                                    doc.metadata.get("source", "Unknown")
                                )
                            )

        sources = ", ".join(
            os.path.basename(doc.metadata.get("source", "Unknown"))
            for doc in docs
        )

        save_multi_doc_message(
            multi_chat_id,
            "assistant",
            answer,
            sources
        )

        st.session_state.multi_doc_ui_history.append(("assistant", answer))

        st.rerun()

    if st.button("Clear Multi Document Chat"):
        st.session_state.multi_doc_ui_history = []
        st.rerun()


# =====================================================
# SINGLE DOCUMENT CHAT
# =====================================================
with tab2:

    from single_doc_chat import (
        load_single_document,
        split_document,
        create_vectorstore,
        ask_single_doc
    )

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"]
    )

    if st.session_state.current_chat is None:
        chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[chat_id] = []
        st.session_state.current_chat = chat_id

    chat_id = st.session_state.current_chat

    if uploaded_file:

        filename = uploaded_file.name
        st.success(f"Document loaded: {filename}")

        if st.session_state.uploaded_files.get(chat_id) != filename:

            with st.spinner("Processing document..."):

                docs = load_single_document(uploaded_file)
                chunks = split_document(docs)

                st.session_state.vectorstores[chat_id] = create_vectorstore(chunks)

            st.session_state.uploaded_files[chat_id] = filename
            st.success("Document processed successfully!")

    if chat_id in st.session_state.vectorstores:

        if st.session_state.single_doc_ui_history is None:
            st.session_state.single_doc_ui_history = get_chat_history(chat_id)

        history = st.session_state.single_doc_ui_history

        for role, message in history:
            with st.chat_message(role):
                st.write(message)

        user_question = st.chat_input("Ask about this document")

        if user_question:

            with st.chat_message("user"):
                st.write(user_question)

            save_message(chat_id, "user", user_question)
            st.session_state.single_doc_ui_history.append(("user", user_question))

            with st.spinner("🤖 Generating answer..."):

                vectorstore = st.session_state.vectorstores[chat_id]

                answer = ask_single_doc(
                    vectorstore,
                    user_question
                )

            with st.chat_message("assistant"):
                st.write(answer)

            save_message(chat_id, "assistant", answer)
            st.session_state.single_doc_ui_history.append(("assistant", answer))


# ---------------------------
# Export Conversation
# ---------------------------
st.divider()

col1, col2 = st.columns(2)

with col1:

    if st.button("Export Chat History"):

        if st.session_state.current_chat:

            history = get_chat_history(
                st.session_state.current_chat
            )

            os.makedirs("logs", exist_ok=True)

            file_path = f"logs/{st.session_state.current_chat}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)

            st.success(f"Chat exported to {file_path}")


with col2:
    st.caption("Built with LangChain • FAISS • Gemini • Streamlit")
# streamlit_app.py

from database import init_db, save_message, get_chat_history, clear_chat

init_db()

import os
import json
import streamlit as st

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 RAG Question Answering System")


# ---------------------------
# Chat Session Manager
# ---------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# store vectorstores per chat
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}

# track uploaded files per chat
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}


# ---------------------------
# Sidebar Chat History
# ---------------------------
st.sidebar.title("💬 Chat History")

if st.sidebar.button("➕ New Chat"):
    chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat = chat_id

for chat_id in st.session_state.chat_sessions.keys():
    if st.sidebar.button(chat_id):
        st.session_state.current_chat = chat_id

if st.session_state.current_chat:
    st.sidebar.markdown(f"**Current Chat:** {st.session_state.current_chat}")


# ---------------------------
# Cache QA System
# ---------------------------
@st.cache_resource
def load_qa_system():
    from retrieval_pipeline import ask_question
    return ask_question


ask_question = load_qa_system()


# ---------------------------
# Multi-Document RAG UI
# ---------------------------
query = st.text_input("Ask a question")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        answer, docs = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    if docs:
        st.subheader("Sources")
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            st.write(f"- {source}")


# ---------------------------
# Single Document Chat
# ---------------------------
from single_doc_chat import (
    load_single_document,
    split_document,
    create_vectorstore,
    ask_single_doc
)

st.divider()
st.header("📄 Chat With Single Document")


# ---------------------------
# Upload Document
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx"]
)


# ---------------------------
# Ensure Chat Exists
# ---------------------------
if st.session_state.current_chat is None:
    chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat = chat_id

chat_id = st.session_state.current_chat


# ---------------------------
# Process Uploaded File
# ---------------------------
if uploaded_file:

    filename = uploaded_file.name

    # rebuild vectorstore if new document uploaded
    if st.session_state.uploaded_files.get(chat_id) != filename:

        with st.spinner("Processing document..."):

            docs = load_single_document(uploaded_file)
            chunks = split_document(docs)

            st.session_state.vectorstores[chat_id] = create_vectorstore(chunks)

        st.session_state.uploaded_files[chat_id] = filename

        st.success("Document processed!")


    # ---------------------------
    # Display Chat History (SQLite)
    # ---------------------------
    history = get_chat_history(chat_id)

    for role, message in history:
        with st.chat_message(role):
            st.write(message)


    # ---------------------------
    # Chat Input
    # ---------------------------
    user_question = st.chat_input("Ask about this document")

    if user_question:

        # Show user message
        with st.chat_message("user"):
            st.write(user_question)

        # Save to SQLite
        save_message(chat_id, "user", user_question)

        with st.spinner("Thinking..."):

            vectorstore = st.session_state.vectorstores[chat_id]

            answer = ask_single_doc(
                vectorstore,
                user_question
            )

        # Show assistant response
        with st.chat_message("assistant"):
            st.write(answer)

        # Save assistant response
        save_message(chat_id, "assistant", answer)


# ---------------------------
# Buttons Section
# ---------------------------
col1, col2 = st.columns(2)

# Clear Chat
with col1:
    if st.button("Clear Chat"):
        clear_chat(chat_id)
        st.rerun()

# Save Conversation
with col2:
    if st.button("Save Conversation"):

        os.makedirs("logs", exist_ok=True)

        history = get_chat_history(chat_id)

        file_path = f"logs/{chat_id}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        st.success(f"Conversation saved to {file_path}")
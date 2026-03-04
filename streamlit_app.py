# streamlit_app.py

import os
import streamlit as st
import json
st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 RAG Question Answering System")


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


# Chat history storage
if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []

# store vector db
if "single_vectorstore" not in st.session_state:
    st.session_state.single_vectorstore = None


uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx"]
)


# ---------------------------
# Process Uploaded File
# ---------------------------
if uploaded_file:

    if st.session_state.single_vectorstore is None:

        with st.spinner("Processing document..."):

            docs = load_single_document(uploaded_file)
            chunks = split_document(docs)

            st.session_state.single_vectorstore = create_vectorstore(chunks)

        st.success("Document processed!")

        # reset chat history for new document
        st.session_state.doc_chat_history = []


    # ---------------------------
    # Show Chat History
    # ---------------------------
    for message in st.session_state.doc_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # ---------------------------
    # Chat Input
    # ---------------------------
    user_question = st.chat_input("Ask about this document")

    if user_question:

        # Show user message
        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.doc_chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Get answer
        with st.spinner("Thinking..."):

            answer = ask_single_doc(
                st.session_state.single_vectorstore,
                user_question
            )

        # Show assistant message
        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.doc_chat_history.append({
            "role": "assistant",
            "content": answer
        })


    # ---------------------------
    # Clear Chat Button
    # ---------------------------
    if st.button("Clear Chat"):
        st.session_state.doc_chat_history = []

    if st.button("Save Conversation"):
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state.doc_chat_history, f, indent=4)

    st.success("Conversation saved successfully!")
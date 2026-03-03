# streamlit_app.py

import os
import streamlit as st
from retrieval_pipeline import ask_question

st.set_page_config(page_title="RAG System", layout="wide")

st.title("📚 RAG Question Answering System")

query = st.text_input("Ask a question")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        answer, docs = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    if docs:
        st.subheader("Sources")
        for doc in docs:
            st.write(os.path.basename(doc.metadata.get("source", "")))
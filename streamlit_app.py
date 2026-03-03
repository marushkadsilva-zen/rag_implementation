# streamlit_app.py

# streamlit_app.py

import os
import streamlit as st

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
# UI
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



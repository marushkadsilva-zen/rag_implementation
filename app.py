import streamlit as st
from retrieval_pipeline import ask_question

st.title("RAG System")

query = st.text_input("Ask a question")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        answer, docs = ask_question(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for doc in docs:
        st.write(doc.metadata.get("source"))
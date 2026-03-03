# retrieval_pipeline.py

import os
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 🔥 Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔥 Load vector store once
vectorstore = FAISS.load_local(
    "db/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 🔥 Global retriever (NO recreation per request)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

# 🔥 Better LLM pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0
)

llm = HuggingFacePipeline(pipeline=pipe)

# 🔥 Stronger anti-hallucination prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question ONLY using the provided context.
If the answer is not explicitly written in the context, say:
"I don't know based on the provided documents."

Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

chain = prompt | llm | StrOutputParser()


def ask_question(question: str):
    docs = retriever.invoke(question)

    if not docs:
        return "I don't know based on the provided documents.", []

    context = "\n\n".join([doc.page_content for doc in docs])

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer.strip(), docs
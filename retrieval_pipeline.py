import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from memory_db import get_memory

load_dotenv()

# ✅ Use LangChain-compatible embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Load vector store once
vectorstore = FAISS.load_local(
    "db/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------------------
# Optimized Retriever
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
# ✅ Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=800,
    max_retries=2
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

You have access to:

1. Previous conversation history
2. Retrieved document context

Use BOTH to answer the question.

If the answer exists in the conversation history,
use that information.

If the answer exists in the document context,
use that.

If neither contains the answer say:
"I don't know."

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["history", "context", "question"]
)
chain = prompt | model | StrOutputParser()

def ask_question(question: str, history: str = ""):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Retrieve stored memory
    user_name = get_memory("user_name")

    memory_context = ""

    if user_name:
        memory_context = f"User name is {user_name}."

    answer = chain.invoke({
        "history": history,
        "context": context + "\n" + memory_context,
        "question": question
    })

    return answer.strip(), docs
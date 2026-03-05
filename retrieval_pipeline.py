import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

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

# ✅ Strong anti-hallucination prompt
# prompt = PromptTemplate(
#     template="""
# You are a helpful assistant.

# Use the provided context to answer the question.
# You may paraphrase and summarize the context.

# If the context contains partial relevant information,
# use it to construct the best possible answer.

# Only say:
# "I don't know based on the provided documents."
# if the context is completely unrelated.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """,
#     input_variables=["context", "question"]
# )
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


# def ask_question(question: str):
#     docs = retriever.invoke(question)

#     if not docs:
#         return "I don't know .", []

#     context = "\n\n".join(doc.page_content for doc in docs)
#     print(context)
#     answer = chain.invoke({
#         "context": context,
#         "question": question
#     })

#     return answer.strip(), docs
def ask_question(question: str, history=None):

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    if history is None:
        history = ""

    answer = chain.invoke({
        "history": history,
        "context": context,
        "question": question
    })

    return answer.strip(), docs
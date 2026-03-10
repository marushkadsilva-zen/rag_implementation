import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# LOAD VECTOR STORE
# --------------------------------------------------

print("Loading FAISS vector database...")

vectorstore = FAISS.load_local(
    "db/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vector DB loaded!")


# --------------------------------------------------
# RETRIEVER
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20
    }
)


# --------------------------------------------------
# GEMINI MODEL
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_tokens=1500
)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant answering questions from research documents.

Use ONLY the provided context to answer.

The context may contain:
- TEXT paragraphs
- TABLE INFORMATION

If a table appears:
1. Explain what the table represents
2. Describe the columns
3. Interpret the rows and values

If the answer is not present in the context say:
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


# --------------------------------------------------
# BUILD CONTEXT FROM RETRIEVED DOCS
# --------------------------------------------------

def build_context(docs):

    context_parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        content_type = doc.metadata.get("content_type", "text")

        if content_type == "table":
            prefix = "TABLE INFORMATION"
        else:
            prefix = "TEXT"

        block = f"""
DOCUMENT {i}

SOURCE: {os.path.basename(source)}
PAGE: {page}
TYPE: {prefix}

CONTENT:
{doc.page_content}
"""

        context_parts.append(block)

    return "\n\n=============================\n\n".join(context_parts)


# --------------------------------------------------
# MAIN QUESTION FUNCTION
# --------------------------------------------------

def ask_question(question: str, history: str = ""):

    print("\nSearching vector database...")

    docs = retriever.invoke(question)

    print(f"Retrieved {len(docs)} chunks")

    context = build_context(docs)

    answer = chain.invoke({
        "history": history,
        "context": context,
        "question": question
    })

    return answer.strip(), docs
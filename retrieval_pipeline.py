from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------------
# EMBEDDINGS
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# --------------------------------
# LOAD QDRANT CLOUD DATABASE
# --------------------------------

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings
)

# --------------------------------
# RETRIEVER
# --------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k":5}
)

# --------------------------------
# LLM
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# --------------------------------
# PROMPT
# --------------------------------

prompt = PromptTemplate(
    template="""
Use the context to answer the question.

If the answer is not in the context say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context","question"]
)

chain = prompt | model | StrOutputParser()

# --------------------------------
# ASK QUESTION
# --------------------------------

def ask_question(question, history=""):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs
# single_doc_chat.py

import tempfile
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------
# Load Document
# ---------------------------
def load_single_document(file):

    suffix = file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)

    elif suffix == "txt":
        loader = TextLoader(tmp_path, autodetect_encoding=True)

    elif suffix == "docx":
        loader = Docx2txtLoader(tmp_path)

    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    # Debug: print extracted document text
    for doc in documents:
        print("\n===== DOCUMENT CONTENT PREVIEW =====")
        print(doc.page_content[:500])
        print("====================================\n")

    return documents


# ---------------------------
# Split Document
# ---------------------------
def split_document(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# ---------------------------
# Create Vector Store
# ---------------------------
def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ---------------------------
# Build QA Chain
# ---------------------------
def build_chain():

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer the question using ONLY the provided document context.

If the answer is not in the document say:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    chain = prompt | model | StrOutputParser()

    return chain


# ---------------------------
# Ask Question
# ---------------------------
def ask_single_doc(vectorstore, question):

    retriever = vectorstore.as_retriever(search_kwargs={"k":5})

    docs = retriever.invoke(question)

    # Debug retrieved chunks
    print("\n===== RETRIEVED CHUNKS =====")
    for doc in docs:
        print(doc.page_content[:300])
        print("-----------------------------")

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = build_chain()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer
import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    loaders = [
        DirectoryLoader(
            docs_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        ),
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_path, glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(docs_path, glob="**/*.csv", loader_cls=CSVLoader),
        DirectoryLoader(docs_path, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(docs_path, glob="**/*.html", loader_cls=UnstructuredHTMLLoader),
    ]

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading files: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents):
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_faiss_vector_store(chunks):
    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs("db", exist_ok=True)
    vectorstore.save_local("db/faiss_index")

    print("✅ FAISS index created successfully!")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_faiss_vector_store(chunks)


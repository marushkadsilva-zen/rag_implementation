import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents):
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def create_faiss_vector_store(chunks):
    print("Creating embeddings using Sentence Transformers...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    vectorstore.save_local("db/faiss_index")

    print("FAISS vector store created and saved successfully!")


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    create_faiss_vector_store(chunks)


if __name__ == "__main__":
    main()
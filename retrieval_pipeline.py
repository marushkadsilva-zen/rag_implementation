from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore


def main():
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = input("Enter your question: ")

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()

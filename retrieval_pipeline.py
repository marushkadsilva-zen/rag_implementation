from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_faiss_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "db/faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def main():
    vectorstore = load_faiss_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = input("Enter your question: ")

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()
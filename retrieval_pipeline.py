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
    print("Loading FAISS vector store...")
    vectorstore = load_faiss_vector_store()

    query = input("\nEnter your question: ")

    print("\nRetrieving results...\n")

    # ✅ Use MMR (diverse retrieval)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10
        }
    )

    docs = retriever.invoke(query)

    # ✅ Get similarity scores separately
    scored_docs = vectorstore.similarity_search_with_score(query, k=3)

    for i, doc in enumerate(docs):
        print("=" * 70)
        print(f"Result {i+1}")
        print("-" * 70)

        # Find matching score
        score = None
        for d, s in scored_docs:
            if d.page_content == doc.page_content:
                score = s
                break

        if score is not None:
            print(f"Similarity Score: {score:.4f}")
        else:
            print("Similarity Score: Not available")

        # Show source file
        source = doc.metadata.get("source", "Unknown")
        print(f"Source File: {source}")

        print("\nContent Preview:\n")
        print(doc.page_content[:500])

        print("=" * 70)


if __name__ == "__main__":
    main()
import os

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# --------------------------------------------------
# TABLE DETECTION
# --------------------------------------------------

def is_markdown_table(text):

    if "|" in text and "---" in text:
        return True

    return False


# --------------------------------------------------
# TABLE -> SEMANTIC TEXT
# --------------------------------------------------

def markdown_table_to_text(table):

    lines = table.split("\n")

    headers = [h.strip() for h in lines[0].split("|") if h.strip()]

    rows = lines[2:]

    output = []

    output.append("Table with columns: " + ", ".join(headers))

    for r in rows:

        cells = [c.strip() for c in r.split("|") if c.strip()]

        if len(cells) != len(headers):
            continue

        row_text = ", ".join(
            f"{headers[i]} = {cells[i]}" for i in range(len(headers))
        )

        output.append(row_text)

    return "\n".join(output)


# --------------------------------------------------
# EXTRACT PDF
# --------------------------------------------------

def extract_pdf(file_path):

    print(f"\nProcessing PDF: {file_path}")

    converter = DocumentConverter()

    result = converter.convert(file_path)

    markdown = result.document.export_to_markdown()

    blocks = markdown.split("\n\n")

    documents = []

    for i, block in enumerate(blocks):

        text = block.strip()

        if not text:
            continue

        # TABLE
        if is_markdown_table(text):

            table_text = markdown_table_to_text(text)

            documents.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": file_path,
                        "page": i,
                        "content_type": "table"
                    }
                )
            )

        # TEXT
        else:

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": i,
                        "content_type": "text"
                    }
                )
            )

    print(f"Extracted {len(documents)} blocks")

    return documents


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

# def load_documents(folder="docs"):

#     documents = []

#     for file in os.listdir(folder):

#         path = os.path.join(folder, file)

#         if file.endswith(".pdf"):

#             docs = extract_pdf(path)

#             documents.extend(docs)

#     print(f"\nTotal extracted elements: {len(documents)}")

#     return documents


def load_documents(folder="docs"):

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        # -----------------------
        # PDF
        # -----------------------
        if file.endswith(".pdf"):

            docs = extract_pdf(path)

            documents.extend(docs)

        # -----------------------
        # TXT
        # -----------------------
        elif file.endswith(".txt"):

            print(f"\nProcessing TXT: {path}")

            loader = TextLoader(path, encoding="utf-8")

            docs = loader.load()

            for d in docs:

                d.metadata["source"] = path
                d.metadata["content_type"] = "text"

            documents.extend(docs)

    print(f"\nTotal extracted elements: {len(documents)}")

    return documents
# --------------------------------------------------
# SPLIT TEXT ONLY
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:

        if doc.metadata["content_type"] == "table":

            chunks.append(doc)

        else:

            splits = splitter.split_documents([doc])

            chunks.extend(splits)

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks
# --------------------------------------------------
#  PRINT CHUNKS (DEBUGGING)
# --------------------------------------------------

def print_chunks(chunks, limit=10):

    print("\nPreview of chunks:\n")

    for i, chunk in enumerate(chunks[:limit]):

        print(f"\nChunk {i+1}")

        print("Metadata:", chunk.metadata)

        print("Content:\n", chunk.page_content[:500])

        print("--------------------------------------------------")
# --------------------------------------------------
# CREATE VECTOR STORE
# --------------------------------------------------

def create_vectorstore(chunks):

    print("\nCreating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs("db", exist_ok=True)

    vectorstore.save_local("db/faiss_index")

    print("\nFAISS index created successfully!")

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    docs = load_documents("docs")

    print("\nSample document preview:\n")

    for d in docs[:3]:

        print(d.metadata)

        print(d.page_content[:300])

        print("---------------")

    # SPLIT DOCUMENTS
    chunks = split_documents(docs)

    # PRINT CHUNKS
    print_chunks(chunks, limit=10)

    # CREATE VECTOR DATABASE
    create_vectorstore(chunks)
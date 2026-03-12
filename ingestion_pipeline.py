import os
import cv2
import pytesseract
import numpy as np

from pdf2image import convert_from_path
from PIL import Image

from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from langchain_community.document_loaders import TextLoader

load_dotenv()

# Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --------------------------------------------------
# TABLE DETECTION
# --------------------------------------------------

def is_markdown_table(text):
    return "|" in text and "---" in text


# --------------------------------------------------
# TABLE → KEY VALUE TEXT
# --------------------------------------------------

def markdown_table_to_kv(table):

    lines = table.split("\n")

    headers = [h.strip() for h in lines[0].split("|") if h.strip()]

    rows = lines[2:]

    output = []

    for i, r in enumerate(rows):

        cells = [c.strip() for c in r.split("|") if c.strip()]

        if len(cells) != len(headers):
            continue

        output.append(f"Row {i+1}:")

        for j in range(len(headers)):
            output.append(f"{headers[j]}: {cells[j]}")

        output.append("")

    return "\n".join(output)


# --------------------------------------------------
# GRAPH EXTRACTION (NEW)
# --------------------------------------------------

def extract_graph_data(pdf_path):

    print(f"Extracting graphs from {pdf_path}")

    images = convert_from_path(
    pdf_path,
    poppler_path=r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"
)

    graph_documents = []

    for page_num, img in enumerate(images):

        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        text = pytesseract.image_to_string(thresh)

        keywords = [
            "yield",
            "inflation",
            "rate",
            "%",
            "spread",
            "index"
        ]

        if any(k in text.lower() for k in keywords):

            graph_documents.append(
                Document(
                    page_content=f"Graph extracted from page {page_num}:\n{text}",
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": "graph"
                    }
                )
            )

    print(f"Graphs extracted: {len(graph_documents)}")

    return graph_documents


# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------

def extract_pdf(file_path):

    converter = DocumentConverter()

    result = converter.convert(file_path)

    markdown = result.document.export_to_markdown()

    blocks = markdown.split("\n\n")

    documents = []

    for i, block in enumerate(blocks):

        text = block.strip()

        if not text:
            continue

        if is_markdown_table(text):

            table_text = markdown_table_to_kv(text)

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

    return documents


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents(folder="docs"):

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        # ---------------- PDF ----------------
        if file.endswith(".pdf"):

            print(f"Processing PDF: {file}")

            docs = extract_pdf(path)
            documents.extend(docs)

            # Extract graph data
            graphs = extract_graph_data(path)
            documents.extend(graphs)

        # ---------------- TXT ----------------
        elif file.endswith(".txt"):

            print(f"Processing TXT: {file}")

            loader = TextLoader(path, encoding="utf-8")

            txt_docs = loader.load()

            for d in txt_docs:

                documents.append(
                    Document(
                        page_content=d.page_content,
                        metadata={
                            "source": path,
                            "content_type": "text"
                        }
                    )
                )

    print(f"\nTotal documents loaded: {len(documents)}")

    return documents


# --------------------------------------------------
# SPLIT DOCUMENTS
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:

        if doc.metadata["content_type"] in ["table", "graph"]:

            chunks.append(doc)

        else:

            splits = splitter.split_documents([doc])

            chunks.extend(splits)

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# CREATE QDRANT CLOUD VECTOR STORE
# --------------------------------------------------

def create_qdrant_db(chunks):

    print("\nCreating Qdrant Cloud database...\n")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    try:
        client.delete_collection("rag_collection")
        print("Old collection deleted")
    except:
        pass

    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embeddings
    )

    print("\nUploading documents in batches...\n")

    batch_size = 8

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i+batch_size]

        vectorstore.add_documents(batch)

        print(f"Uploaded batch {i//batch_size + 1}")

    print("\nDocuments uploaded to Qdrant Cloud successfully!")

    return vectorstore


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    docs = load_documents("docs")

    chunks = split_documents(docs)

    vectorstore = create_qdrant_db(chunks)

    print("\nDocuments indexed in Qdrant Cloud")
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "db"

def ingest_pdf(pdf_path):

    print("Loading PDF:", pdf_path)

    # 1. Delete old DB completely
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # 2. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print("Chunks created:", len(chunks))

    # 4. Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Create fresh DB
    print("Creating vector DB...")

    Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    print("Done indexing.")
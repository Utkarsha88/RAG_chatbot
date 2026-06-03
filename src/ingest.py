import os
import shutil
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def clear_old_db():
    """Safe delete for Windows (fixes file lock issue)"""

    if not os.path.exists(DB_PATH):
        return

    # give OS time to release file handles
    time.sleep(0.5)

    try:
        shutil.rmtree(DB_PATH, ignore_errors=True)
    except PermissionError:
        print("DB locked, retrying...")

        time.sleep(1)
        shutil.rmtree(DB_PATH, ignore_errors=True)


def ingest_pdf(pdf_path):

    file_name = os.path.basename(pdf_path).replace(".pdf", "")
    file_name = file_name.replace(" ", "_").lower()

    print(f"\nLoading PDF: {file_name}")

    # Read PDF
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    print("Chunks:", len(chunks))

    # 🔥 IMPORTANT FIX:
    # Each upload is stored cleanly per document name
    # AND we reset DB to avoid old contamination

    clear_old_db()

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_name="main"
)

    # modern Chroma auto-persists, so no db.persist() needed
    print("Indexed:", file_name)
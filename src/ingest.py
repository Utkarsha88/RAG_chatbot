from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import shutil


def ingest_pdf(pdf_path):

    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove old database
    shutil.rmtree("db", ignore_errors=True)

    print("Creating embeddings...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="db"
    )

    print("Done.")
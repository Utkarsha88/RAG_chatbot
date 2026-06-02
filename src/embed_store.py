from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --------------------
# 1. Load PDF
# --------------------
pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

# --------------------
# 2. Chunk text
# --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# --------------------
# 3. Embeddings model
# --------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------
# 4. Create vector DB
# --------------------
db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="db"
)

db.persist()

print("Vector DB created and saved!")
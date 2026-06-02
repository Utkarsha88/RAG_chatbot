from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# 1. Load embedding model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 2. Load vector DB
# -------------------------
db = Chroma(
    persist_directory="db",
    embedding_function=embedding_model
)

# -------------------------
# 3. Ask a question
# -------------------------
query = "What is machine learning?"

results = db.similarity_search(query, k=3)

# -------------------------
# 4. Show results
# -------------------------
print("\nQUERY:", query)
print("\n--- TOP MATCHES ---\n")

for i, doc in enumerate(results):
    print(f"Result {i+1}:\n")
    print(doc.page_content)
    print("\n-------------------\n")
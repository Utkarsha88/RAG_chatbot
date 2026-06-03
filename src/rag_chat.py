from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -------------------------
# Embedding Model
# -------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Load DB
# -------------------------
db = Chroma(
    persist_directory="db",
    embedding_function=embedding_model
)

# -------------------------
# LLM
# -------------------------
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)

# -------------------------
# Detect summary
# -------------------------
def is_summary(q):
    q = q.lower()
    return any(x in q for x in [
        "summary", "summarize", "overview",
        "entire document", "whole document"
    ])

# -------------------------
# Retrieve context
# -------------------------
def get_context(question):
    if is_summary(question):
        docs = db.get()["documents"]

        # better spread
        step = max(1, len(docs) // 25)
        selected = docs[::step][:25]

        return "\n\n".join(selected)

    docs = db.similarity_search(question, k=6)
    return "\n\n".join([d.page_content for d in docs])

# -------------------------
# MAIN ASK FUNCTION
# -------------------------
def ask(question):

    print("\n[DEBUG] Question:", question)
    print("[DEBUG] Retrieving documents...")

    context = get_context(question)

    print("[DEBUG] Context length:", len(context))

    if is_summary(question):

        prompt = f"""
You are a document analyst.

Write a structured summary:

Summary:
Main Idea:
Key Points:
Conclusion:

Context:
{context}
"""

    else:

        prompt = f"""
You are a strict QA assistant.

RULES:
- Use ONLY context
- If missing say "Not found"

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt).content
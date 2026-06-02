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
# Chroma DB
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
# Detect Summary Query
# -------------------------
def is_summary_query(question):
    q = question.lower()

    keywords = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "entire document",
        "whole document",
        "main idea",
        "full report"
    ]

    return any(k in q for k in keywords)


# -------------------------
# Get Summary Context
# -------------------------
def get_summary_context():
    all_docs = db.get()["documents"]

    if not all_docs:
        return ""

    # balanced coverage (not only first chunks)
    step = max(1, len(all_docs) // 20)
    selected = all_docs[::step]

    return "\n\n".join(selected)


# -------------------------
# Get QA Context
# -------------------------
def get_qa_context(question):
    docs = db.similarity_search(question, k=8)

    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------
# MAIN ASK FUNCTION
# -------------------------
def ask(question):

    print("\n[DEBUG] Retrieving documents...")

    if is_summary_query(question):

        context = get_summary_context()

        prompt = f"""
You are an expert document analyst.

Write a clear and structured summary.

FORMAT:
Summary:
Main Idea:
Key Points:
Conclusion:

Context:
{context}
"""

    else:

        context = get_qa_context(question)

        prompt = f"""
You are a strict document QA assistant.

RULES:
- Use ONLY the given context
- If answer not present, say "Not found in document"

Context:
{context}

Question:
{question}

Answer:
"""

    print("[DEBUG] Context length:", len(context))

    response = llm.invoke(prompt)

    return response.content
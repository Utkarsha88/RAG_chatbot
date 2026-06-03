from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_db():
    return Chroma(
        persist_directory="db",
        embedding_function=embedding_model,
        collection_name="main"
    )

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)

# -------------------------
# GET CONTEXT (ALL DOCS)
# -------------------------
def get_context(question):

    db = get_db()
    docs = db.similarity_search(question, k=8)

    return "\n\n".join(d.page_content for d in docs)

# -------------------------
# ASK FUNCTION
# -------------------------
def ask(question):

    print("\n[DEBUG] Question:", question)

    context = get_context(question)
    if not context.strip():
        return "No document context found. Please upload and index a PDF first."

    raw_query = question.strip()
    if len(raw_query.split()) <= 3 and not raw_query.endswith("?"):
        prompt_question = (
            f"Explain how the document describes or uses the term: {raw_query}. "
            "If the term is not discussed in the context, reply exactly: Not found in document."
        )
    else:
        prompt_question = raw_query

    prompt = f"""
You are a helpful document assistant.

Use ONLY the context below to answer the question. Do not add information that is not in the context.
If the answer is not contained in the context, respond exactly: Not found in document.

Context:
{context}

Question:
{prompt_question}

Answer:
"""

    res = llm.invoke(prompt)

    return res.content.strip()
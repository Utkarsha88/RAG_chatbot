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
# GET CONTEXT (DOCS + TEXT)
# -------------------------
def get_context_docs(question, k=8):

    db = get_db()
    docs = db.similarity_search(question, k=k)

    return docs


def get_context_text(docs):
    return "\n\n".join(d.page_content for d in docs)

# -------------------------
# ASK FUNCTION
# -------------------------
def ask(question):

    print("\n[DEBUG] Question:", question)

    raw_query = (question or "").strip()

    # fetch top documents (more for summaries)
    docs = get_context_docs(raw_query, k=12)
    context_text = get_context_text(docs)

    if not context_text.strip():
        return "No document context found. Please upload and index a PDF first."

    # Debug: print brief previews of top matches
    print("[DEBUG] Top matches previews:")
    for i, d in enumerate(docs[:3]):
        preview = d.page_content.replace("\n", " ")[:200]
        print(f"  {i+1}. {preview}...")

    # If the user asked for a summary, synthesize a document-level summary
    q_lower = raw_query.lower()
    summary_triggers = ["summarize", "summary", "summarise", "summery", "summarize the entire document", "summarize entire document"]
    if any(t in q_lower for t in summary_triggers):
        prompt = f"""
You are a concise document summarizer. Use ONLY the context below to produce a clear, structured summary of the entire document.
Cover: objective, dataset/features, methods, key results/findings, chemicals or sensors mentioned, and main conclusions. Keep the summary under 300 words.
If the information is not present in the context, say exactly: Not found in document.

Context:
{context_text}

Produce the summary:
"""
        res = llm.invoke(prompt)
        return res.content.strip()

    # For very short keyword queries, ask the model to explain how the term is used
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
{context_text}

Question:
{prompt_question}

Answer:
"""

    res = llm.invoke(prompt)

    return res.content.strip()
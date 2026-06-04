# 🤖 RAG Chatbot — PDF Q&A with Local LLM + Vector DB

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDFs and ask questions about them in natural language — powered by a local LLM (Ollama / Llama 3.2) and semantic search via ChromaDB. No OpenAI API or internet connection required.

---

## ✨ Features

- 📂 Upload and index PDF documents through a clean Gradio UI
- 🧠 Ask questions in natural language — summaries, details, comparisons
- 🔍 Semantic search using ChromaDB + HuggingFace MiniLM embeddings
- 🤖 Fully local LLM inference via Ollama (Llama 3.2 3B)
- 📝 Automatic text chunking (~1000 tokens) for precise retrieval
- 📊 Supports both Q&A mode and full document summarization
- ⚡ Fast local inference — no API keys, no usage costs
- 🔄 Re-index new PDFs at any time

---

## 🏗️ Architecture

```
PDF → Text Extraction → Chunking → Embeddings → ChromaDB
                                                      ↓
              User Query → Embedding → Retrieval → Context
                                                      ↓
                                             Ollama LLM → Answer
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── app.py                  # Gradio UI — upload, chat, clear
├── requirements.txt        # Python dependencies
├── README.md
│
├── data/
│   └── sample.pdf          # Sample PDF (replace with your own)
│
├── db/                     # ChromaDB vector store (auto-generated)
│   └── chroma.sqlite3
│
└── src/
    ├── ingest.py           # PDF ingestion, chunking & indexing
    ├── rag_chat.py         # RAG pipeline — retrieval + LLM response
    ├── retrieve.py         # Standalone retrieval testing
    ├── chunk_text.py       # Chunk inspection & testing
    └── load_pdf.py         # PDF loader utility
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio |
| LLM | Ollama (Llama 3.2 3B) |
| Embeddings | `sentence-transformers` (MiniLM) |
| Vector DB | ChromaDB |
| PDF Parsing | PyPDF |
| Orchestration | LangChain Community |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and pull the model

Download Ollama from 👉 [https://ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2:3b
```

Start the Ollama server (keep this running in the background):

```bash
ollama serve
```

---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:7860
```

---

## 📌 How to Use

**Step 1 — Upload a PDF**
1. Click **Upload PDF** and select your file
2. Click **Index PDF** to process and store it in the vector database

**Step 2 — Ask Questions**

| Example Query | What It Does |
|---|---|
| `Summarize the document` | Full document summary with key points |
| `What is this report about?` | High-level overview |
| `Explain section 2` | Deep-dives into a specific section |
| `Who is the author?` | Extracts metadata-level details |

---

## 🧠 How RAG Works (Step by Step)

1. **PDF Loading** — PyPDF reads and extracts raw text
2. **Chunking** — Text is split into ~1000-token overlapping chunks
3. **Embedding** — Each chunk is converted to a vector via MiniLM
4. **Indexing** — Vectors are stored in ChromaDB with metadata
5. **Query Embedding** — User's question is converted to a vector
6. **Retrieval** — Top-K most similar chunks are fetched
7. **Prompting** — Retrieved context is injected into the LLM prompt
8. **Generation** — Ollama generates a grounded, context-aware answer

---

## 📦 Key Components

### `ingest.py` — PDF Ingestion Pipeline
- Loads PDF pages with PyPDF
- Splits text into overlapping chunks
- Generates MiniLM embeddings
- Persists everything to ChromaDB

### `rag_chat.py` — RAG Engine
- Converts user query to embedding
- Retrieves semantically relevant chunks
- Detects summarization intent automatically
- Sends context + query to Ollama for response generation

### `app.py` — Gradio Interface
- PDF upload and indexing button
- Conversational chat interface with history
- Clear chat option

---

## ⚠️ Known Issues & Fixes

**1. Gradio chatbot format error**

Use the `tuples` type and return history correctly:
```python
gr.Chatbot(type="tuples")
# ...
history.append((user_message, bot_response))
```

**2. Old PDF content still appearing after re-upload**

ChromaDB persists previous embeddings. Clear the database before re-indexing:
```python
import shutil
shutil.rmtree("db", ignore_errors=True)
```

**3. Ollama not responding**

Make sure the server is running in a separate terminal:
```bash
ollama serve
```

---

## 🔥 Roadmap

- [ ] 📚 Multi-PDF chat support
- [ ] 📎 Source citations with chunk references per answer
- [ ] 🧾 Sidebar showing retrieved context chunks
- [ ] 💬 Streaming responses for faster feedback
- [ ] 🌐 Cloud deployment (Render / Hugging Face Spaces)
- [ ] 🔐 Authentication layer for private deployments

---

## 🧪 Example Output

**Query:** `Summarize the entire document`

```
📄 Summary
The document covers [topic] with a focus on [main theme].

🎯 Main Idea
[Core argument or purpose of the document]

🔑 Key Points
• Point 1
• Point 2
• Point 3

✅ Conclusion
[Closing takeaway extracted from the document]
```

**Query:** `What chemical is used in the system?`

```
Based on the document, the chemical used is [X], described in section [Y] as...
[Contextual explanation extracted from the relevant PDF chunk]
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

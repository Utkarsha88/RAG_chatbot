import gradio as gr
from src.rag_chat import ask
from src.ingest import ingest_pdf


# -------------------------
# Upload PDF
# -------------------------
def upload_pdf(pdf):
    if pdf is None:
        return "Please upload a PDF."

    pdf_path = getattr(pdf, "name", pdf)
    ingest_pdf(pdf_path)
    return "PDF indexed successfully."


# -------------------------
# Chat function (TUPLE FORMAT FIXED)
# -------------------------
def chat_fn(message, history):

    print("\n[DEBUG] Question:", message)

    answer = ask(message)

    if history is None:
        history = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        index_btn = gr.Button("Index PDF")

    status = gr.Textbox(label="Status")

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(placeholder="Ask something...")
    clear = gr.Button("Clear Chat")

    # Upload
    index_btn.click(upload_pdf, pdf_upload, status)

    # Chat
    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])

    # Clear
    clear.click(lambda: [], None, chatbot)

demo.launch()
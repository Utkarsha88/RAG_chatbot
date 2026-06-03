import gradio as gr
from src.rag_chat import ask
from src.ingest import ingest_pdf

# -------------------------
# Upload PDF
# -------------------------
def upload_pdf(file):

    if file is None:
        return "Upload a PDF first"

    ingest_pdf(file.name)

    return "PDF indexed successfully"

# -------------------------
# Chat function (FIXED FORMAT)
# -------------------------
def chat_fn(message, history):

    answer = ask(message)

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return "", history

# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    with gr.Row():
        pdf_upload = gr.File(file_types=[".pdf"])
        index_btn = gr.Button("Index PDF")

    status = gr.Textbox()

    chatbot = gr.Chatbot(type="messages", height=500)

    msg = gr.Textbox(placeholder="Ask something...")
    clear = gr.Button("Clear")

    index_btn.click(upload_pdf, pdf_upload, status)

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])

    clear.click(lambda: [], None, chatbot)

demo.launch()
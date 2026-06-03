import gradio as gr

from src.rag_chat import ask
from src.ingest import ingest_pdf


# -------------------------
# PDF Upload
# -------------------------
def upload_pdf(pdf):

    if pdf is None:
        return "Please upload a PDF."

    ingest_pdf(pdf.name)

    return "PDF indexed successfully."


# -------------------------
# Chat Function
# -------------------------
def chat_fn(message, history):

    answer = ask(message)

    history.append(
        (message, answer)
    )

    return "", history


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    with gr.Row():

        pdf_upload = gr.File(
            label="Upload PDF",
            file_types=[".pdf"]
        )

        index_btn = gr.Button("Index PDF")

    status = gr.Textbox(
        label="Status"
    )

    chatbot = gr.Chatbot(
        height=500
    )

    msg = gr.Textbox(
        placeholder="Ask something..."
    )

    clear = gr.Button("Clear Chat")

    index_btn.click(
        upload_pdf,
        inputs=pdf_upload,
        outputs=status
    )

    msg.submit(
        chat_fn,
        [msg, chatbot],
        [msg, chatbot]
    )

    clear.click(
        lambda: [],
        outputs=chatbot
    )


# -------------------------
# Run
# -------------------------
demo.launch()
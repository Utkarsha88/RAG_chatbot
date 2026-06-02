import gradio as gr
from src.rag_chat import ask

# -------------------------
# Chat function
# -------------------------
def chat_fn(message, history):
    print("\n[DEBUG] User question:", message)

    answer = ask(message)

    # IMPORTANT: old Gradio format
    history.append((message, answer))

    return history


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot (Document QA System)")

    chatbot = gr.Chatbot()   # ❌ no "type=" here

    msg = gr.Textbox(placeholder="Ask anything from document...")

    clear = gr.Button("Clear")

    # send message
    msg.submit(chat_fn, [msg, chatbot], chatbot)

    # clear chat
    clear.click(lambda: [], None, chatbot)

# -------------------------
# Run
# -------------------------
demo.launch()
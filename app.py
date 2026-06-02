import gradio as gr
from src.rag_chat import ask

# -------------------------
# Chat function (MESSAGES FORMAT)
# -------------------------
def chat_fn(message, history):

    print("\n[DEBUG] User question:", message)

    answer = ask(message)

    # convert history into correct format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 RAG Chatbot")

    # IMPORTANT: NO type argument
    chatbot = gr.Chatbot()

    msg = gr.Textbox(placeholder="Ask something...")

    clear = gr.Button("Clear")

    msg.submit(chat_fn, [msg, chatbot], chatbot)

    clear.click(lambda: [], None, chatbot)


# -------------------------
# Run
# -------------------------
demo.launch()
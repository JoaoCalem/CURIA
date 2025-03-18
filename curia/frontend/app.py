"""
A Gradio-based chatbot interface for interacting with a RAG system.
"""

import gradio as gr

from curia.rag import VectorStore


def chatbot(message, _):
    """
    Takes a user message and yields the streaming response from the RAG system.

    Args:
        message (str): The user's input message.
        history (list): The chat history (unused in this function).

    Yields:
        str: The streaming response from the RAG system.
    """
    streaming_response = vectors.query(message)
    message = ""
    for token in streaming_response.response_gen:
        message += token
        yield message


vectors = VectorStore(restart_database=False, streaming=True)

examples = ["What happens in case about Parfums Marcel Rochas, in detail?"]

iface = gr.ChatInterface(
    fn=chatbot,
    title="CURIA Chatbot",
    examples=examples,
)
iface.launch()

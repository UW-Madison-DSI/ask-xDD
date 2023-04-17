import gradio as gr
from .models import t5


def answer(context: str, question: str, **kwargs) -> str:
    """Prompt-based text summary using LLM."""

    TEMPLATE = f"Based on the following context answer this question: {question} Context: {text}"
    return t5(TEMPLATE.format(question=question, article=text), **kwargs)


with open("data/demo_text.txt", "r") as f:
    text = f.read()

example = [text, "Which is the best extractive reader?"]


def app(context, question):
    """Clean output version of `answer`."""
    y = answer(context, question)
    y = y.replace("<pad> context:", "")
    return y.replace("</s>", "")


gr.Interface(
    fn=app,
    inputs=[
        gr.Textbox(lines=10, label="context"),
        gr.Textbox(lines=2, label="question"),
    ],
    outputs=gr.Textbox(label="Answer"),
    examples=[example],
).launch()

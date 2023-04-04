import gradio as gr
from backend import prompt_sum

with open("data/demo_text.txt", "r") as f:
    text = f.read()

example = [text, "Which is the best extractive reader?"]


def app(text, question):
    """Clean output version of prompt_sum."""
    y = prompt_sum(text, question)
    y = y.replace("<pad> context:", "")
    return y.replace("</s>", "")


gr.Interface(
    fn=app,
    inputs=[gr.Textbox(lines=10, label="text"), gr.Textbox(lines=2, label="question")],
    outputs=gr.Textbox(label="Answer"),
    examples=[example],
).launch()

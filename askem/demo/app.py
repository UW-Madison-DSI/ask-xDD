from dataclasses import dataclass
from enum import Enum
from typing import Optional

import react
import streamlit as st

# Initialize states
st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data
def get_questions():
    with open("questions.txt", "r") as f:
        return f.read().splitlines()


if "questions" not in st.session_state:
    st.session_state.questions = get_questions()


# Convinience functions


@dataclass
class Message:
    role: str
    content: str
    container: str
    avatar: Optional[str] = None


class Container(Enum):
    MARKDOWN = "markdown"
    EXPANDER = "expander"


def render(message: Message) -> None:
    """Render message in chat."""

    with st.chat_message(message.role, avatar=message.avatar):
        if message.container == Container.EXPANDER:
            with st.expander(message.content[:80] + "..."):
                st.markdown(message.content)
        else:
            st.markdown(message.content)


def chat_log(role: str, content: str, container: Container = None, avatar: str = None):
    message = Message(role, content, container, avatar)
    st.session_state.messages.append(message)
    render(message)


# App logic
st.title("ASKEM: COVID-19 QA demo")

# Re-render chat history
for message in st.session_state.messages:
    render(message)


def main(question: str):
    """Main loop of the demo app."""
    chat_log(role="user", content=question)

    answer = {}
    react_iterator = react.get_iterator(question)

    # Iterate ReAct chain
    while not "output" in answer:
        with st.spinner("Generating..."):
            answer = next(react_iterator)

        if "intermediate_step" in answer:
            action_logs = answer["intermediate_step"][0][0].log.split("\n")
            for action_log in action_logs:
                chat_log(role="assistant", content=action_log)

            action_returns = answer["intermediate_step"][0][1].split("\n\n")
            for action_return in action_returns:
                chat_log(
                    role="assistant",
                    content=action_return,
                    container=Container.EXPANDER,
                    avatar="📜",
                )

    final_answer = answer["output"]
    chat_log(role="assistant", content=final_answer)


if question := st.chat_input("Ask a question about COVID-19", key="question"):
    main(question)

# Preset questions
with st.sidebar:
    preset_question = st.selectbox(
        "Select example questions", st.session_state.questions
    )
    run_from_preset = st.button("Run")

if run_from_preset:
    main(preset_question)

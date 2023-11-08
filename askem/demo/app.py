import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from citation import to_apa
from connector import query_react


def append_citation(document: dict) -> None:
    """Append citation to document."""

    try:
        document["citation"] = to_apa(document["paper_id"], in_text=True)
    except Exception:
        document["citation"] = document["paper_id"]


# Initialize states
st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "settings" not in st.session_state:
    st.session_state.settings = {}


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
    title: str = None
    link: str = None


def chat_log(
    role: str,
    content: str,
    container: str = None,
    avatar: str = None,
    title: str = None,
    link: str = None,
):
    message = Message(role, content, container, avatar, title, link)
    st.session_state.messages.append(message)
    render(message)


def render(message: Message) -> None:
    """Render message in chat."""

    with st.chat_message(message.role, avatar=message.avatar):
        if message.container == "expander":
            if message.title:
                title = message.title
            else:
                title = message.content[:80] + "..."

            with st.expander(title):
                st.markdown(message.content)
                if message.link:
                    st.markdown(f"Source: {message.link}")
        else:
            st.markdown(message.content)


# App logic
st.title("ASKEM: COVID-19 QA demo")

# Re-render chat history
for message in st.session_state.messages:
    render(message)


def main(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int,
    retriever_endpoint: str = None,
) -> dict:
    """Main loop of the demo app."""
    chat_log(role="user", content=question)

    if not retriever_endpoint:
        retriever_endpoint = os.getenv("RETRIEVER_URL")

    # Call the API

    with st.spinner(
        "Running... It may take 30 seconds or longer if you choose GPT-4. "
    ):
        final_answer = query_react(
            question=question,
            top_k=top_k,
            model_name=model_name,
            screening_top_k=screening_top_k,
            retriever_endpoint=retriever_endpoint,
        )
        for doc in final_answer["used_docs"]:
            append_citation(doc)
            chat_log(
                role="assistant",
                content=doc["text"],
                container="expander",
                avatar="📄",
                title=doc["citation"],
                link=f"https://xdd.wisc.edu/api/v2/articles/?docid={doc['paper_id']}",
            )
        chat_log(role="assistant", content=final_answer["answer"])


if question := st.chat_input("Ask a question about COVID-19", key="question"):
    main(question, **st.session_state.settings)

# Preset questions
with st.sidebar:
    st.subheader("Ask preset questions")
    preset_question = st.selectbox("Select a question", st.session_state.questions)
    run_from_preset = st.button("Run")

    st.subheader("Advanced settings")
    st.markdown(
        "You can customize the QA system, all of these settings are available in the [API route](http://cosmos0001.chtc.wisc.edu:4502/docs) as well."
    )

    st.session_state["settings"]["model_name"] = st.radio(
        "model", ["gpt-4", "gpt-3.5-turbo-16k"]
    )
    st.session_state["settings"]["top_k"] = st.number_input("retriever top-k", value=5)
    st.session_state["settings"]["screening_top_k"] = st.number_input(
        "screening phase top-k", value=100
    )


if run_from_preset:
    main(question=preset_question, **st.session_state.settings)

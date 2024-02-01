from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st
from citation import to_apa
from connector import query_react


@dataclass
class Topic:
    name: str  # backend name, must match with xdd dataset
    label: str  # frontend label
    preset_questions_path: str | Path

    def __post_init__(self) -> None:
        with open(self.preset_questions_path, "r") as f:
            self.preset_questions = f.read().splitlines()


@dataclass
class AppSettings:
    title: str
    topics: list[Topic]
    model_names: list[str]


@dataclass
class Message:
    role: str
    content: str
    container: str
    avatar: str | None = None
    title: str = None
    link: str = None


def append_citation(document: dict) -> None:
    """Append citation to document."""

    try:
        document["citation"] = to_apa(document["paper_id"], in_text=True)
    except Exception:
        document["citation"] = document["paper_id"]


def append_title(document: dict) -> None:
    """Append citation to document."""

    try:
        xdd_response = requests.get(
            f"https://xdd.wisc.edu/api/v2/articles/?docid={document['paper_id']}"
        )
        xdd_response.raise_for_status()
        xdd_data = xdd_response.json()
        document["title"] = xdd_data["success"]["data"][0]["title"]
    except Exception:
        document["title"] = ""


def fix_string(string: str) -> str:
    """Fix string for markdown."""
    return string.encode("utf-16", "surrogatepass").decode("utf-16")


def render(message: Message) -> None:
    """Render message in chat."""

    with st.chat_message(message.role, avatar=message.avatar):
        if message.container == "expander":
            if message.title:
                title = message.title
            else:
                title = message.content[:80] + "..."

            with st.expander(title):
                try:
                    st.markdown(message.content)
                except UnicodeEncodeError:
                    st.markdown(fix_string(message.content))
                if message.link:
                    st.markdown(f"Source: {message.link}")
        else:
            st.markdown(message.content)


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


def render_doc(doc: dict) -> None:
    """Render document in chat."""

    append_citation(doc)
    append_title(doc)
    chat_log(
        role="assistant",
        content=doc["text_content"],
        container="expander",
        avatar="📄",
        title=f"{doc['title']} [{doc['citation']}]",
        link=f"https://xdd.wisc.edu/api/v2/articles/?docid={doc['paper_id']}",
    )


def render_chunk(chunk: dict, verbose: bool) -> None:
    """Decode different types of chunk."""

    if "thoughts" in chunk and verbose:
        chat_log(role="assistant", content=chunk["thoughts"])

    if "used_docs" in chunk:
        for doc in chunk["used_docs"]:
            render_doc(doc)

    if "answer" in chunk:
        chat_log(role="assistant", content=chunk["answer"])


async def search(
    question: str,
    topic: str,
    top_k: int,
    model_name: str,
    screening_top_k: int,
    verbose: bool,
) -> dict:
    """Main loop of the demo app."""
    chat_log(role="user", content=question)

    # Call the API
    with st.spinner("Running... It may take 30 seconds or longer if you choose GPT-4."):
        async for chunk in query_react(
            question=question,
            topic=topic,
            top_k=top_k,
            model_name=model_name,
            screening_top_k=screening_top_k,
        ):
            with st.spinner():
                render_chunk(chunk, verbose=verbose)

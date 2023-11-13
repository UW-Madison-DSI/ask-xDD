from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st
from citation import to_apa
from connector import query_react


@dataclass
class AppSettings:
    title: str
    topics: list[str]
    preset_questions_paths: list[Path, str]
    model_names: list[str]

    @st.cache_data
    def preset_questions(self) -> dict[str, list[str]]:
        """Load preset questions from file."""

        preset_questions = {}
        for topic, path in zip(self.topics, self.preset_questions_paths):
            with open(path, "r") as f:
                preset_questions[topic] = f.read().splitlines()
        return preset_questions


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
        print(xdd_data)
        document["title"] = xdd_data["success"]["data"][0]["title"]
    except Exception:
        document["title"] = ""


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
            if verbose:
                with st.spinner():
                    chat_log(role="assistant", content=chunk)

            last_chunk = chunk

    if not verbose:
        chat_log(role="assistant", content=last_chunk)

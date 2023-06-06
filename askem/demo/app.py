import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
import streamlit as st
from dotenv import load_dotenv

import askem.retriever
import askem.summarizer

load_dotenv()
st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")
st.title("ASKEM: COVID-19 QA demo")


def ask_generator(question: str, context: str) -> dict:
    """Send request to generator REST API service."""

    response = requests.post(
        os.getenv("GENERATOR_URL"),
        headers={"Content-Type": "application/json"},
        json={"paragraph": context, "question": question},
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()


@st.cache_resource
def get_retriever_client():
    return askem.retriever.get_client()


RETRIEVER_CLIENT = get_retriever_client()

question = st.text_input("Ask your question.")

# Sidebar (for low-level system settings)
with st.sidebar:
    st.subheader("Retriever settings")
    top_k = st.slider("Top-k: How many paragraphs to retrieve?", 1, 10, 5)
    distance = st.slider(
        "Distance: How similar should the paragraphs be?", 0.0, 1.0, 0.7, 0.1
    )
    topic = st.selectbox("Topic filter", ["covid"])
    preprocessor_id = st.selectbox("Preprocessor filter", ["haystack_v0.0.2"])

    st.subheader("Generator settings")
    answer_score_threshold = st.slider("Answer score threshold", 0.0, 1.0, 0.0, 0.01)

    st.subheader("Summarizer settings")
    skip_generator = st.checkbox("Skip generator", value=False)


# When pressing submit button, execute the QA pipeline
if st.button("Submit"):
    # Retriever
    st.header("Retriever")
    with st.spinner("Getting relevant passages..."):
        paragraphs = askem.retriever.get_documents(
            RETRIEVER_CLIENT,
            question=question,
            top_k=top_k,
            distance=distance,
        )

    _ = [st.info(p.text) for p in paragraphs]

    # Generator
    st.header("Generator")
    if skip_generator:
        # Bypass generator workflow
        answers = [p.text for p in paragraphs]
    else:
        with st.spinner("Answering..."):
            contexts = [p.text for p in paragraphs]
            ask = partial(ask_generator, question)

            # Parallelize (TODO: Need to configure FastAPI for maximizing speed)
            with ThreadPoolExecutor() as executor:
                answers = list(executor.map(ask, contexts))

            print("Pre-screened answers:")
            print(answers)

            print("Answers, after filtering by threshold:")
            answers = [a for a in answers if a["score"] >= answer_score_threshold]
            st.json(answers)

            # text only answers
            answers = [a["answer"] for a in answers]
            st.success(answers)

    # Summarizer
    st.header("Summarizer")
    with st.spinner("Summarizing..."):
        simple_answer = askem.summarizer.summarize(question, contexts=answers)
        st.info(simple_answer)

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
import streamlit as st
from dotenv import load_dotenv

import askem.retriever
import askem.summarizer
from askem.demo.auth import st_check_password

load_dotenv()
st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")


# Resources
def ask_generator(question: str, context: str) -> dict:
    """Send request to generator REST API service."""

    response = requests.post(
        os.getenv("GENERATOR_URL"),
        headers={"Content-Type": "application/json"},
        json={"context": context, "question": question},
    )

    if response.status_code != 200:
        raise Exception(response.text)

    return response.json()


@st.cache_resource
def get_retriever_client():
    return askem.retriever.get_client()


RETRIEVER_CLIENT = get_retriever_client()


if st_check_password():
    st.title("ASKEM: COVID-19 QA demo")

    question = st.text_input("Ask your question.")

    # Sidebar (for low-level system settings)
    with st.sidebar:
        st.subheader("Retriever settings")
        top_k = st.slider("Top-k: How many documents to retrieve:", 1, 10, 5)
        distance = st.slider(
            "Distance: Maximum acceptable cosine distance:", 0.0, 1.0, 0.7, 0.1
        )
        topic = st.selectbox("Topic filter", ["covid"])
        preprocessor_id = st.selectbox(
            "Preprocessor filter", ["haystack_v0.0.1", "haystack_v0.0.2"]
        )

        doc_type = st.selectbox("Document type", ["paragraph", "table", "figure"])

        st.subheader("Generator settings")
        answer_score_threshold = st.slider(
            "Answer score threshold", 0.0, 1.0, 0.0, 0.01
        )

        st.subheader("Summarizer settings")
        skip_generator = st.checkbox("Skip generator", value=False)

    # When pressing submit button, execute the QA pipeline
    if st.button("Submit"):
        # Retriever
        st.header("Retriever")
        with st.spinner("Getting relevant passages..."):
            documents = askem.retriever.get_documents(
                RETRIEVER_CLIENT,
                question=question,
                top_k=top_k,
                distance=distance,
                topic=topic,
                doc_type=doc_type,
                preprocessor_id=preprocessor_id,
            )

        _ = [st.info(d.text) for d in documents]  # show documents in frontend

        # Generator
        st.header("Generator")

        if skip_generator:
            # Bypass generator workflow
            answers = [d.text for d in documents]
        else:
            with st.spinner("Answering..."):
                contexts = [d.text for d in documents]
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

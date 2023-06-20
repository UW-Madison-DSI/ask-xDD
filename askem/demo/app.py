import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import requests
import streamlit as st

import askem.retriever
import askem.summarizer
from askem.demo.auth import st_check_password
from askem.demo.citation import to_apa
from askem.demo.style import to_html

logging.basicConfig(level=logging.DEBUG)
st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")


# Resources
def ask_generator(question: str, context: str) -> dict:
    """Send request to generator REST API service."""

    response = requests.post(
        os.getenv("GENERATOR_URL"),
        headers={"Content-Type": "application/json"},
        # json={"paragraph": context, "question": question},
        json={
            "context": context,
            "question": question,
        },  # TODO: update to match with generator when deployed
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Generator Response: {response.json()}")
    return response.json()


def append_citation(document: askem.retriever.Document) -> None:
    """Append citation to document."""

    try:
        document.citation = to_apa(document.paper_id, in_text=True)
    except Exception:
        document.citation = document.paper_id


def append_answer(question: str, document: askem.retriever.Document) -> None:
    """Append generator answer to document."""

    document.answer = ask_generator(question, document.text)

    # Remove junk answer
    if document.answer["answer"] == ".":
        document.answer = None

    logging.info(f"Answer: {document.answer}")


@st.cache_resource
def get_retriever_client():
    return askem.retriever.get_client()


RETRIEVER_CLIENT = get_retriever_client()

# Password protection
if os.getenv("DEBUG") == "1" or st_check_password():
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
        preprocessor_id = st.selectbox("Preprocessor filter", ["haystack_v0.0.2"])

        doc_type = st.selectbox("Document type", ["paragraph", "table", "figure"])

        st.subheader("Summarizer settings")
        skip_generator = st.checkbox("Skip generator", value=False)

    # When pressing submit button, execute the QA pipeline
    # TODO: refactorize into a module

    if st.button("Submit"):
        # Retriever
        st.header("Answer")
        with st.spinner("Finding relevant documents..."):
            documents = askem.retriever.get_documents(
                RETRIEVER_CLIENT,
                question=question,
                top_k=top_k,
                distance=distance,
                topic=topic,
                doc_type=doc_type,
                preprocessor_id=preprocessor_id,
            )

            # Append citation to document
            with ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(append_citation, documents)

        # Generator
        if not skip_generator:
            with st.spinner("Generating intermediate answers..."):
                logging.info(f"Asking generator: {question}")
                ask = partial(append_answer, question)

                # Write answer to document
                with ThreadPoolExecutor(max_workers=8) as executor:
                    logging.debug(
                        f"Documents before executor: {[doc.__dict__ for doc in documents]}"
                    )
                    executor.map(ask, documents)

                logging.debug(f"Documents after executor: {documents}")

            documents = [doc for doc in documents if doc.answer is not None]
            logging.info(f"Documents after generator: {documents}")

        # Summarizer
        if not documents:
            st.warning(
                "Unable to locate any relevant papers that address your question."
            )
            st.stop()

        with st.spinner("Summarizing..."):
            if skip_generator:
                simple_answer = askem.summarizer.summarize(
                    question, contexts=[document.text for document in documents]
                )
            else:
                simple_answer = askem.summarizer.summarize(
                    question,
                    contexts=[document.answer["answer"] for document in documents],
                )

        # Output to UI
        st.info(simple_answer)

        # References
        st.subheader("References")
        for document in documents:
            with st.expander(document.citation):
                html = to_html(
                    doc_type=document.doc_type,
                    text=document.text,
                    generator_answer=document.answer,
                    cosmos_object_id=document.cosmos_object_id,
                )
                st.markdown(html, unsafe_allow_html=True)

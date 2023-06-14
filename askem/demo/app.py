import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import requests
import streamlit as st

import askem.retriever
import askem.summarizer
from askem.demo.auth import st_check_password
from askem.demo.citation import to_apa


logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")


# Resources
def ask_generator(question: str, context: str) -> dict:
    """Send request to generator REST API service."""

    response = requests.post(
        os.getenv("GENERATOR_URL"),
        headers={"Content-Type": "application/json"},
        json={"paragraph": context, "question": question},
        # json={"context": context, "question": question},  # TODO: update to match with generator when deployed
    )

    if response.status_code != 200:
        raise Exception(response.text)

    logging.debug(f"Generator Response: {response.json()}")

    return response.json()


def append_answer(
    question: str, document: askem.retriever.Document
) -> askem.retriever.Document:
    """Append generator answer to document."""

    document.answer = ask_generator(question, document.text)
    document.answer["answer"] = question + " " + document.answer["answer"]
    logging.info(f"Answer: {document.answer}")
    return document


@st.cache_resource
def get_retriever_client():
    return askem.retriever.get_client()


def highlight(text: str, start: int, end: int) -> str:
    """Highlight section in text."""
    return text[:start] + "**:red[" + text[start:end] + "]**" + text[end:]


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

        st.subheader("Generator settings")
        answer_score_threshold = st.slider(
            "Answer score threshold", 0.0, 1.0, 0.0, 0.01
        )

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
        for document in documents:
            try:
                document.citation = to_apa(document.paper_id, in_text=True)
            except Exception:
                document.citation = document.paper_id

        if skip_generator:
            # Bypass generator workflow
            for document in documents:
                document.html = document.text
                document.answer = {"answer": document.text}

        else:
            with st.spinner("Generating intermediate answers..."):
                logging.info(f"Asking generator: {question}")
                ask = partial(append_answer, question)

                # Parallelize (TODO: Need to configure FastAPI for maximizing speed)
                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Write answer to document
                    logging.debug(
                        f"Documents before executor: {[doc.__dict__ for doc in documents]}"
                    )
                    executor.map(ask, documents)

                logging.debug(f"Documents after executor: {documents}")

                # Append stylized text to document
                for document in documents:
                    document.html = highlight(
                        document.text,
                        document.answer["start"],
                        document.answer["end"],
                    )

            # Filter out documents with low answer score
            documents = [
                doc
                for doc in documents
                if doc.answer["score"] >= answer_score_threshold
            ]

            # Filter out answer with only a dot
            documents = [doc for doc in documents if doc.answer["answer"] != "."]

        # Summarizer
        if not documents:
            st.warning(
                "Unable to locate any relevant papers that address your question."
            )
            st.stop()

        with st.spinner("Summarizing..."):
            answers = [document.answer["answer"] for document in documents]
            simple_answer = askem.summarizer.summarize(question, contexts=answers)

        # Output to UI
        st.info(simple_answer)

        # References
        st.subheader("References")
        for document in documents:
            with st.expander(document.citation):
                st.markdown(document.html, unsafe_allow_html=True)

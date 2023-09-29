import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import streamlit as st
from auth import st_check_password
from citation import to_apa
from connector import ask_generator, query_retriever, summarize
from style import to_html

st.set_page_config(page_title="COVID-19 Question answering.", page_icon="📚")


def append_citation(document: dict) -> None:
    """Append citation to document."""

    try:
        document["citation"] = to_apa(document["paper_id"], in_text=True)
    except Exception:
        document["citation"] = document["paper_id"]


def append_answer(question: str, document: dict) -> None:
    """Append generator answer to document."""

    document["answer"] = ask_generator(question, document["text"])

    # Remove junk answer
    if document["answer"]["answer"] == ".":
        document["answer"] = None

    logging.info(f"Answer: {document['answer']}")


def retriever_workflow(
    question: str, top_k: int, distance: float, topic: str, doc_type: str
) -> List[dict]:
    """Retrieve module in the demo app."""

    logging.info(f"Retrieving {top_k} documents...")

    with st.spinner("Finding relevant documents..."):
        documents = query_retriever(
            question=question,
            top_k=top_k,
            distance=distance,
            topic=topic,
            doc_type=doc_type,
        )

    # Append citation to document
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(append_citation, documents)

    if not documents:
        st.warning("Unable to locate any relevant papers that address your question.")
        st.stop()

    return documents


def generator_workflow(question: str, documents: List[dict]) -> List[dict]:
    """Generator module in the demo app."""

    logging.info(f"Generating answers for {len(documents)} documents...")

    with st.spinner("Generating intermediate answers..."):
        logging.info(f"Asking generator: {question}")
        ask = partial(append_answer, question)

    # Append answer to document
    with ThreadPoolExecutor(max_workers=8) as executor:
        logging.debug(f"Documents before executor: {documents}")
        executor.map(ask, documents)

    logging.debug(f"Documents after executor: {documents}")
    return [doc for doc in documents if doc["answer"] is not None]


def summarizer_workflow(documents: List[dict]) -> str:
    """Summarizer module in the demo app."""

    logging.info(f"Summarizing {len(documents)} documents...")

    has_generator_answer = any("answer" in document for document in documents)

    if has_generator_answer:
        context = [document["answer"]["answer"] for document in documents]
    else:
        context = [document["text"] for document in documents]

    with st.spinner("Summarizing..."):
        summarized_answer = summarize(question, contexts=context)
    return summarized_answer


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

        doc_type = st.selectbox("Document type", ["paragraph", "figure"])

        if doc_type is "paragraph":
            st.subheader("Generator settings")
            skip_generator = st.checkbox("Skip generator", value=False)

    if st.button("Submit"):
        # Processing pipeline (Retriever, Generator, Summarizer)
        if doc_type is not "paragraph":
            skip_generator = True  # Override user input
            skip_summarizer = True
        else:
            skip_summarizer = False

        # Retriever
        documents = retriever_workflow(question, top_k, distance, topic, doc_type)

        # Generator
        if not skip_generator:
            documents = generator_workflow(question, documents)

        # Summarizer
        if not skip_summarizer:
            summarized_answer = summarizer_workflow(documents)
        else:
            summarized_answer = None

        # Output to UI
        if not skip_summarizer:
            st.header("Answer")
            st.info(summarized_answer)

        # References
        st.subheader("References")
        for document in documents:
            with st.expander(document["citation"]):
                html = to_html(
                    doc_type=document["doc_type"],
                    text=document["text"],
                    generator_answer=document["answer"] if not skip_generator else None,
                    cosmos_object_id=document["cosmos_object_id"],
                )
                st.markdown(html, unsafe_allow_html=True)

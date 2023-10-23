import logging
import os
from typing import List, Optional

import openai
import requests


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


def query_retriever(
    question: str,
    top_k: int,
    distance: Optional[float] = 0.5,
    topic: Optional[str] = None,
    doc_type: Optional[str] = None,
    preprocessor_id: Optional[str] = None,
) -> List[dict]:
    """Send request to retriever REST API service.

    Also see: askem/retriever/app.py
    """

    data = {
        "question": question,
        "top_k": top_k,
    }

    # Append optional arguments
    optional_args = {
        "distance": distance,
        "topic": topic,
        "doc_type": doc_type,
        "preprocessor_id": preprocessor_id,
    }
    for k, v in optional_args.items():
        if v is not None:
            data[k] = v

    response = requests.post(
        os.getenv("RETRIEVER_URL"),
        headers={
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        },
        json=data,
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Retriever Response: {response.json()}")
    return response.json()


def summarize(question: str, contexts: List[str]) -> str:
    """Compresses a long text to a shorter version."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    instruction = "Answer the question based on the contexts. If there is no answer in context, say 'no answer'."

    # Provide QA pairs as context
    qa_context = [f"{question}: {context}" for context in contexts]

    # Append main question
    prompt = f"Question: {question}{os.linesep} Context: {' '.join(qa_context)}"
    logging.debug(f"Summarizer prompt: {prompt}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def query_react(
    question: str,
    top_k: int,
    model_name: str,
    screening_top_k: int = None,
    retriever_endpoint: str = None,
    **kwargs,
) -> dict:
    """Send request to retriever/react API service.

    Also see: askem/retriever/app.py
    """

    data = {
        "question": question,
        "top_k": top_k,
        "model_name": model_name,
        "screening_top_k": screening_top_k,
        "retriever_endpoint": retriever_endpoint,
    }

    response = requests.post(
        os.getenv("REACT_URL"),
        headers={
            "Content-Type": "application/json",
            "Api-Key": os.getenv("RETRIEVER_APIKEY"),
        },
        json=data,
    )

    if response.status_code != 200:
        logging.debug(response.text)
        raise Exception(response.text)

    logging.debug(f"Retriever Response: {response.json()}")
    return response.json()

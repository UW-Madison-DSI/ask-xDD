import logging
import os
from contextlib import asynccontextmanager
from typing import Any, List

import requests
from auth import has_valid_api_key
from base import get_client, get_documents
from data_models import BaseQuery, Document, HybridQuery, ReactQuery
from fastapi import Depends, FastAPI
from react import ReactManager


def xdd_search(query: str, top_k: int, dataset: str) -> dict:
    """Query xdd articles API.

    Args:
        query: Query string.
        top_k: Number of documents to return. e.g.: 5.
        dataset: Dataset to query. e.g.: xdd-covid-19.
    """

    url = os.getenv("HYBRID_SEARCH_XDD_URL")

    params = {
        "term": query,
        "dataset": dataset,
        "max": top_k,
        "match": "true",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_contents(response: dict, path: list, field: str) -> List[Any]:
    """Get list of _gddid values from response."""

    for key in path:
        response = response[key]

    return [hit[field] for hit in response]


def _screening(query: str, top_k: int) -> list[str]:
    """Get a list of article ids with elastic search."""

    results = xdd_search(query, top_k, dataset="xdd-covid-19")
    return get_contents(results, ["success", "data"], "_gddid")


# FastAPI app

cached_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cache weaviate client before API startup."""
    cached_resources["weaviate_client"] = get_client()
    yield
    # Release resources when app stops
    cached_resources["weaviate_client"] = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_root():
    """Health check."""
    return {"ping": "pong!"}


@app.post("/vector", dependencies=[Depends(has_valid_api_key)])
async def get_docs_from_vector(query: BaseQuery) -> List[Document]:
    """Search relevant documents using vector search."""

    logging.info(f"Accessing vector route with: {query}")
    return get_documents(client=cached_resources["weaviate_client"], **query.dict())


@app.post("/hybrid", dependencies=[Depends(has_valid_api_key)])
async def hybrid_get_docs(query: HybridQuery) -> List[Document]:
    """Hybrid search relevant documents."""

    logging.info(f"Accessing hybrid route with: {query}")
    assert query.paper_ids is None

    # Stage 1: Screening with XDD elastic search
    query = query.dict()
    query["paper_ids"] = _screening(query["question"], query.pop("screening_top_k"))

    return get_documents(client=cached_resources["weaviate_client"], **query)


@app.post("/react", dependencies=[Depends(has_valid_api_key)])
async def react_chain(query: ReactQuery) -> dict:
    """ReAct search chain."""

    logging.info(f"Accessing react route with: {query}")

    search_config = query.dict()
    entry_question = search_config.pop("question")
    model_name = search_config.pop("model_name")
    retriever_endpoint = search_config.pop("retriever_endpoint")

    print(f"{search_config=}")

    chain = ReactManager(
        entry_query=entry_question,
        retriever_endpoint=retriever_endpoint,
        search_config=search_config,
        model_name=model_name,
        verbose=False,
    )

    answer = chain.run()
    return {"answer": answer, "used_docs": chain.used_docs}

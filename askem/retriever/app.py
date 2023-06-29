import logging
from contextlib import asynccontextmanager
from typing import List

import weaviate
from base import get_client
from data_models import Document, Query
from fastapi import FastAPI, Depends
from auth import has_valid_api_key


def to_document(result: dict) -> Document:
    """Convert a weaviate result to a `Document`."""

    return Document(
        paper_id=result["paper_id"],
        cosmos_object_id=result["cosmos_object_id"],
        doc_type=result["type"],
        text=result["text_content"],
        distance=result["_additional"]["distance"],
    )


def get_documents(
    client: weaviate.Client,
    question: str,
    top_k: int = 5,
    distance: float = 0.5,
    topic: str = None,
    doc_type: str = None,
    preprocessor_id: str = None,
) -> List[Document]:
    """Ask a question to retriever and return a list of relevant `Document`.

    Args:
        client: Weaviate client.
        query: Query string.
        top_k: Number of documents to return. Defaults to 5.
        distance: Max distance of the document. Defaults to 0.5.
        topic: Topic filter of the document. Defaults to None (No filter).
        preprocessor_id: Preprocessor filter of the document. Defaults to None (No filter).
    """

    # Get weaviate results
    results = (
        client.query.get(
            "Passage",
            [
                "paper_id",
                "text_content",
                "topic",
                "preprocessor_id",
                "type",
                "cosmos_object_id",
            ],
        )
        .with_near_text({"concepts": [question], "distance": distance})
        .with_additional(["distance"])
    )

    where_filter = {"operator": "And", "operands": []}
    # Filter by preprocessor id
    if preprocessor_id is not None:
        logging.info(f"Filtering by preprocessor_id: {preprocessor_id}")

        where_filter["operands"].append(
            {
                "path": ["preprocessor_id"],
                "operator": "Equal",
                "valueText": preprocessor_id,
            }
        )

    # Filter by topic
    if topic is not None:
        logging.info(f"Filtering by topic: {topic}")
        where_filter["operands"].append(
            {"path": ["topic"], "operator": "Equal", "valueText": topic}
        )

    # Filter by doc_type
    if doc_type is not None:
        logging.info(f"Filtering by doc_type: {doc_type}")
        where_filter["operands"].append(
            {"path": ["type"], "operator": "Equal", "valueText": doc_type}
        )

    # Apply filters
    if where_filter["operands"]:
        results = results.with_where(where_filter).with_limit(top_k).do()
    else:
        results = results.with_limit(top_k).do()
    logging.info(f"Retrieved {len(results['data']['Get']['Passage'])} results")

    # Convert results to Document and return
    return [to_document(result) for result in results["data"]["Get"]["Passage"]]


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


# POST endpoint for query
@app.post("/", dependencies=[Depends(has_valid_api_key)])
async def get_docs(query: Query) -> List[Document]:
    """Search relevant documents."""

    logging.info(f"[get_docs] querying document: {query}")
    logging.info(f"{query.dict()=}")
    return get_documents(client=cached_resources["weaviate_client"], **query.dict())

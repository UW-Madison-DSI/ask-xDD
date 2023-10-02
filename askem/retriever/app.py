import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import requests
import weaviate
from auth import has_valid_api_key
from base import get_client
from data_models import Document, Query
from fastapi import Depends, FastAPI, HTTPException


def to_document(result: dict) -> Document:
    """Convert a weaviate result to a `Document`."""

    article_terms = [result[f"article_terms_{i}"] for i in range(10)]
    article_terms = [term for term in article_terms if term]  # Remove empty terms

    paragraph_terms = [result[f"paragraph_terms_{i}"] for i in range(3)]
    paragraph_terms = [term for term in paragraph_terms if term]  # Remove empty terms

    return Document(
        paper_id=result["paper_id"],
        cosmos_object_id=result["cosmos_object_id"],
        doc_type=result["type"],
        text=result["text_content"],
        distance=result["_additional"]["distance"],
        article_terms=article_terms,
        paragraph_terms=paragraph_terms,
    )


def get_documents(
    client: weaviate.Client,
    question: str,
    top_k: int = 5,
    distance: float = 0.5,
    topic: Optional[str] = None,
    doc_type: Optional[str] = None,
    preprocessor_id: Optional[str] = None,
    article_terms: Optional[List[str]] = None,
    paragraph_terms: Optional[List[str]] = None,
    paper_ids: Optional[List[str]] = None,
    move_to: Optional[str] = None,
    move_to_weight: Optional[float] = 1.0,
    move_away_from: Optional[str] = None,
    move_away_from_weight: Optional[float] = 1.0,
) -> List[Document]:
    """Ask a question to retriever and return a list of relevant `Document`.

    Args:
        client: Weaviate client.
        question: Query string.
        top_k: Number of documents to return. Defaults to 5.
        distance: Max distance of the document. Defaults to 0.5.
        topic: Topic filter of the document. Defaults to None (No filter).
        doc_type: Doc type filter of the document. Defaults to None (No filter).
        preprocessor_id: Preprocessor filter of the document. Defaults to None (No filter).
        article_terms: List of parent's article terms (CAPITALIZED WORDS) to filter by. Defaults to None (No filter).
        paragraph_terms: List of paragraph terms to filter by. Defaults to None (No filter).
        paper_ids: List of paper ids to filter by. Defaults to None (No filter).
        move_to: Adds an optional concept string to the query vector for more targeted results. Defaults to None, meaning no additional concept is added.
        move_to_weight: Weight of the move_to vectoring (range: 0-1). Defaults to 1.0.
        move_away_from: Adds an optional concept string to the query vector for more targeted results. Defaults to None, meaning no additional concept is added.
        move_away_from_weight: Weight of the move_away_from vectoring (range: 0-1). Defaults to 1.0.
    """

    # Get weaviate results (not executed yet)
    output_fields = ["paper_id", "cosmos_object_id", "preprocessor_id"]
    output_fields.extend(["topic", "type", "text_content"])
    output_fields.extend([f"article_terms_{i}" for i in range(10)])
    output_fields.extend([f"paragraph_terms_{i}" for i in range(3)])

    # Progressively build up the get query: filter -> near_text ->  limit
    results = client.query.get("Passage", output_fields).with_additional(["distance"])

    where_filter = {"operator": "And", "operands": []}

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

    # Filter by article_terms: any of the terms must be present in the article_terms_0,1...9
    if article_terms is not None:
        logging.info(f"Filtering by article_terms: {article_terms}")

        _operands = []
        for i in range(10):
            path_name = f"article_terms_{i}"
            _operands.append(
                {
                    "path": [path_name],
                    "operator": "ContainsAny",
                    "valueText": article_terms,
                }
            )

        where_filter["operands"].append(
            {
                "operator": "Or",
                "operands": _operands,
            }
        )

    # Filter by paragraph_terms: any of the terms must be present in the paragraph_terms_0,1,2
    if paragraph_terms is not None:
        logging.info(f"Filtering by paragraph_terms: {paragraph_terms}")

        _operands = []
        for i in range(3):
            path_name = f"paragraph_terms_{i}"
            _operands.append(
                {
                    "path": [path_name],
                    "operator": "ContainsAny",
                    "valueText": paragraph_terms,
                }
            )

        where_filter["operands"].append(
            {
                "operator": "Or",
                "operands": _operands,
            }
        )

    # Filter by paper_ids
    if paper_ids is not None:
        logging.info(f"Filtering by paper_ids: {paper_ids}")

        where_filter["operands"].append(
            {
                "path": ["paper_id"],
                "operator": "ContainsAny",
                "valueText": paper_ids,
            }
        )

    if where_filter["operands"]:
        results = results.with_where(where_filter)

    # Build near text query

    near_text_query = {"concepts": [question], "distance": distance}

    if move_to is not None:
        logging.debug(f"Moving towards {move_to} with weight {move_to_weight}")
        near_text_query["moveTo"] = {"concepts": [move_to], "force": move_to_weight}

    if move_away_from is not None:
        logging.debug(
            f"Moving away from {move_away_from} with weight {move_away_from_weight}"
        )
        near_text_query["moveAwayFrom"] = {
            "concepts": [move_away_from],
            "force": move_away_from_weight,
        }

    results = results.with_near_text(near_text_query)

    # Limit and run
    results = results.with_limit(top_k).do()

    if "data" not in results:
        logging.info(f"No results found")
        logging.info(f"{results=}")
        raise HTTPException(status_code=404, detail="No results found")

    logging.info(f"Retrieved {len(results['data']['Get']['Passage'])} results")

    # Convert results to Document and return
    return [to_document(result) for result in results["data"]["Get"]["Passage"]]


# Elastic search


def xdd_search(query: str, top_k: int, dataset: str) -> dict:
    """Query xdd articles API.

    Args:
        query: Query string.
        top_k: Number of documents to return. e.g.: 5.
        dataset: Dataset to query. e.g.: xdd-covid-19.
    """

    url = "https://xdd.wisc.edu/api/articles"

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


# POST endpoint for query
@app.post("/", dependencies=[Depends(has_valid_api_key)])
async def get_docs(query: Query) -> List[Document]:
    """Search relevant documents."""

    logging.info(f"[get_docs] querying document: {query}")
    logging.info(f"{query.dict()=}")
    return get_documents(client=cached_resources["weaviate_client"], **query.dict())


@app.post("/hybrid", dependencies=[Depends(has_valid_api_key)])
async def hybrid_get_docs(query: Query) -> List[Document]:
    """Hybrid search relevant documents."""

    logging.info(f"[get_docs] querying document: {query}")
    logging.info(f"{query.dict()=}")
    assert query.paper_ids is None

    STAGE_1_TOP_K = 100  # TODO: tune? discuss any metrics for absolute quality boundary
    query.paper_ids = _screening(query.question, STAGE_1_TOP_K)

    return get_documents(client=cached_resources["weaviate_client"], **query.dict())

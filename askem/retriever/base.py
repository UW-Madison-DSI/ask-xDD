import json
import logging
import os
from typing import List, Optional

import weaviate
from data_models import Document
from fastapi import HTTPException

LATEST_SCHEMA_VERSION = 1


def get_client(url: str = None, apikey: str = None) -> weaviate.Client:
    """Get a weaviate client."""

    if url is None:
        url = os.getenv("WEAVIATE_URL")

    if apikey is None:
        apikey = os.getenv("WEAVIATE_APIKEY")

    logging.info(f"Connecting to Weaviate at {url}")
    return weaviate.Client(url, weaviate.auth.AuthApiKey(apikey))


def get_v1_schema() -> dict:
    """Obtain the v1 schema."""
    return {
        "class": "Passage",
        "description": "Paragraph chunk of a document",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": False}},
        "vectorIndexConfig": {"distance": "dot"},
        "properties": [
            {
                "name": "paper_id",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {
                "name": "topic",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {
                "name": "preprocessor_id",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {
                "name": "doc_type",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {
                "name": "cosmos_object_id",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {"name": "text_content", "dataType": ["text"]},
        ],
    }


def init_retriever(client: weaviate.Client | None = None, version: int = 1) -> None:
    """Initialize the passage retriever."""

    if client is None:
        client = get_client()

    if version == 1:
        PASSAGE_SCHEMA = get_v1_schema()

    client.schema.create_class(PASSAGE_SCHEMA)

    # Dump full schema to file
    with open(f"./askem/schema/passage_v{version}.json", "w") as f:
        json.dump(client.schema.get(PASSAGE_SCHEMA["class"]), f, indent=2)


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
    question: str,
    client: weaviate.Client,
    top_k: int = 5,
    distance: float = 0.5,
    topic: Optional[str] = None,
    doc_type: Optional[str] = None,
    preprocessor_id: Optional[str] = None,
    paper_ids: Optional[List[str]] = None,
    move_to: Optional[str] = None,
    move_to_weight: Optional[float] = 1.0,
    move_away_from: Optional[str] = None,
    move_away_from_weight: Optional[float] = 1.0,
    **kwargs,
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

    if "data" not in results or not results["data"]["Get"]["Passage"]:
        logging.info(f"No results found")
        logging.info(f"{results=}")
        raise HTTPException(status_code=404, detail="No results found")

    logging.info(f"Retrieved {len(results['data']['Get']['Passage'])} results")

    # Convert results to Document and return
    return [to_document(result) for result in results["data"]["Get"]["Passage"]]

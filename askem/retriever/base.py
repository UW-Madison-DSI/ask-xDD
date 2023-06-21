import json
import logging
import os
from pathlib import Path
from typing import List

import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

from data_models import Document


def get_client(url: str = None, apikey: str = None) -> weaviate.Client:
    """Get a weaviate client."""

    load_dotenv()
    if url is None:
        url = os.getenv("WEAVIATE_URL")

    if apikey is None:
        apikey = os.getenv("WEAVIATE_APIKEY")

    logging.info(f"Connecting to Weaviate at {url}")
    return weaviate.Client(url, weaviate.auth.AuthApiKey(apikey))


def init_retriever(force: bool = False, client=None) -> None:
    """Initialize the passage retriever."""

    if client is None:
        client = get_client()

    if force:
        client.schema.delete_all()

    # Passage schema, for all types of documents, including paragraph, figures and tables
    # TODO: If safe, rename to document?
    PASSAGE_SCHEMA = {
        "class": "Passage",
        "description": "Paragraph chunk of a document",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": False}},
        # "vectorIndexConfig": {"distance": "dot"},  #TODO: parameterize this
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
                "name": "type",  # TODO: rename to doc_type if safe
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

    client.schema.create_class(PASSAGE_SCHEMA)

    # Dump full schema to file
    with open("./askem/schema/passage.json", "w") as f:
        json.dump(client.schema.get("passage"), f, indent=2)


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
        limit: Max number of documents to return. Defaults to 5.
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

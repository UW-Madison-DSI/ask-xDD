import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

from askem.preprocessing import ASKEMPreprocessor, HaystackPreprocessor


def get_client(url: str = None, apikey: str = None) -> weaviate.Client:
    """Get a weaviate client."""

    load_dotenv()
    if url is None:
        url = os.getenv("WEAVIATE_URL")

    if apikey is None:
        apikey = os.getenv("WEAVIATE_APIKEY")

    logging.info(f"Connecting to Weaviate at {url}")
    return weaviate.Client(url, weaviate.auth.AuthApiKey(apikey))


def init_retriever(force: bool = False, client=None):
    """Initialize the passage retriever."""

    if client is None:
        client = get_client()

    if force:
        client.schema.delete_all()

    PASSAGE_SCHEMA = {
        "class": "Passage",
        "description": "Paragraph chunk of a document",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {"text2vec-transformers": {"vectorizeClassName": False}},
        # "vectorIndexConfig": {"distance": "dot"},
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
                "name": "type",
                "dataType": ["text"],  # Paragraph, Table, Figure
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


def import_passages(
    input_dir: str, topic: str, preprocessor: ASKEMPreprocessor = None, client=None
) -> None:
    """Ingest passages into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    if client is None:
        client = get_client()

    input_files = Path(input_dir).glob("**/*.txt")

    for input_file in tqdm(list(input_files)):
        passages = preprocessor.run(input_file=input_file, topic=topic)

        for passage in passages:
            client.data_object.create(data_object=passage, class_name="Passage")


@dataclass
class Paragraph:
    paper_id: str  # xdd document id
    text: str  # paragraph text
    distance: float  # distance metric of the paragraph


def to_paragraph(result: dict) -> Paragraph:
    """Convert a weaviate result to a `Paragraph`."""

    return Paragraph(
        paper_id=result["paper_id"],
        text=result["text_content"],
        distance=result["_additional"]["distance"],
    )


def get_paragraphs(
    client: weaviate.Client,
    question: str,
    top_k: int = 5,
    distance: float = 0.5,
    topic: str = None,
    preprocessor_id: str = None,
) -> List[Paragraph]:
    """Ask a question to retriever and return a list of relevant `Paragraph`.

    Args:
        client: Weaviate client.
        query: Query string.
        limit: Max number of paragraphs to return. Defaults to 5.
        distance: Max distance of the paragraph. Defaults to 0.5.
        topic: Topic filter of the paragraph. Defaults to None (No filter).
        preprocessor_id: Preprocessor filter of the paragraph. Defaults to None (No filter).
    """

    # Get weaviate results
    results = (
        client.query.get(
            "Passage", ["paper_id", "text_content", "topic", "preprocessor_id"]
        )
        .with_near_text({"concepts": [question], "distance": distance})
        .with_additional(["distance"])
    )

    if topic is not None:
        filter = {"path": ["topic"], "operator": "Equal", "valueText": topic}
        results = results.with_where(filter)

    if preprocessor_id is not None:
        filter = {
            "path": ["preprocessor_id"],
            "operator": "Equal",
            "valueText": preprocessor_id,
        }
        results = results.with_where(filter)

    results = results.with_limit(top_k).do()

    # Convert results to Paragraph and return
    return [to_paragraph(result) for result in results["data"]["Get"]["Passage"]]

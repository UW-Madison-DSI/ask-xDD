import json
import logging
import os
import weaviate

LATEST_SCHEMA_VERSION = 2


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


def to_v2(schema: dict) -> dict:
    """Convert a v1 schema to a v2 schema."""
    v2_extra_properties = []

    # 10 new paper terms
    for i in range(10):
        v2_extra_properties.append(
            {
                "name": f"article_terms_{i}",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            }
        )

    # 3 new paragraph terms
    for i in range(3):
        v2_extra_properties.append(
            {
                "name": f"paragraph_terms_{i}",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            }
        )

    schema["properties"].extend(v2_extra_properties)
    return schema


def get_v2_schema() -> dict:
    """Obtain the v2 schema."""
    return to_v2(get_v1_schema())


def init_retriever(client=None, version: int = 1) -> None:
    """Initialize the passage retriever."""

    if client is None:
        client = get_client()

    if version == 1:
        PASSAGE_SCHEMA = get_v1_schema()
    elif version == 2:
        PASSAGE_SCHEMA = get_v2_schema()

    client.schema.create_class(PASSAGE_SCHEMA)

    # Dump full schema to file
    with open(f"./askem/schema/passage_v{version}.json", "w") as f:
        json.dump(client.schema.get(PASSAGE_SCHEMA["class"]), f, indent=2)

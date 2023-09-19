import json
import logging
import os
import weaviate


def get_client(url: str = None, apikey: str = None) -> weaviate.Client:
    """Get a weaviate client."""

    if url is None:
        url = os.getenv("WEAVIATE_URL")

    if apikey is None:
        apikey = os.getenv("WEAVIATE_APIKEY")

    logging.info(f"Connecting to Weaviate at {url}")
    return weaviate.Client(url, weaviate.auth.AuthApiKey(apikey))

def to_v2(schema: dict) -> dict:
    """Convert a v1 schema to a v2 schema."""
    v2_extra_properties = []

    # 10 new paper terms
    for i in range(10):
        v2_extra_properties.append({"name": f"paper_terms_{i}", "dataType": ["text"], "moduleConfig": {"text2vec-transformers": {"skip": True}}})
    
    # 3 new paragraph terms
    for i in range(3):
        v2_extra_properties.append({"name": f"paragraph_terms_{i}", "dataType": ["text"], "moduleConfig": {"text2vec-transformers": {"skip": True}}})

    schema["class"] = "PassageV2"
    schema["properties"].extend(v2_extra_properties)
    return schema

def init_retriever(client=None, version: int=1) -> None:
    """Initialize the passage retriever."""

    if client is None:
        client = get_client()

    # Passage schema, for all types of documents, including paragraph, figures and tables
    # TODO: If safe, rename to document?
    PASSAGE_SCHEMA = {
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
            {"name": "text_content", "dataType": ["text"]}
        ],
    }

    # V2 schema diff
    if version == 2:
        PASSAGE_SCHEMA = to_v2(PASSAGE_SCHEMA)

    client.schema.create_class(PASSAGE_SCHEMA)

    # Dump full schema to file
    with open(f"./askem/schema/passage_v{version}.json", "w") as f:
        json.dump(client.schema.get(PASSAGE_SCHEMA["class"]), f, indent=2)

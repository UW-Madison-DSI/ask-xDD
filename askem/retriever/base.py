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

import json
import logging
import os
from pathlib import Path

import click
import weaviate
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm

from askem.preprocessing import HaystackPreprocessor, Preprocessor

logging.basicConfig(level=logging.DEBUG)


def get_client() -> weaviate.Client:
    """Get the Weaviate client."""

    load_dotenv()
    WEAVIATE_APIKEY = os.getenv("WEAVIATE_APIKEY")

    return weaviate.Client(
        "http://localhost:8080",
        auth_client_secret=weaviate.auth.AuthApiKey(WEAVIATE_APIKEY),
    )


def init_retriever(force: bool = False):
    """Initialize the passage retriever."""

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
                "name": "type",
                "dataType": ["text"],  # Paragraph, Table, Figure
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
            {"name": "text_content", "dataType": ["text"]},
            {
                "name": "cosmos_object_id",
                "dataType": ["text"],
                "moduleConfig": {"text2vec-transformers": {"skip": True}},
            },
        ],
    }

    client.schema.create_class(PASSAGE_SCHEMA)

    # Dump full schema to file
    with open("./askem/schema/passage.json", "w") as f:
        json.dump(client.schema.get("passage"), f, indent=2)


def ingest_passages(input_dir: str, preprocessor: Preprocessor = None) -> None:
    """Ingest passages into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    client = get_client()
    input_files = Path(input_dir).glob("**/*.txt")

    for input_file in tqdm(list(input_files)):
        passages = preprocessor.run(input_file=input_file)

        for passage in passages:
            client.data_object.create(data_object=passage, class_name="Passage")


@click.command()
@click.option(
    "--init",
    help="Force re-initialization, It will wipe everything!",
    type=bool,
    is_flag=True,
)
@click.option("--input-dir", help="Input directory.", type=str)
def main(init: bool, input_dir: str):
    """Main entrypoint."""

    logging.debug(f"Initializing passage retriever... with {init=}")
    if init:
        init_retriever(force=True)

    logging.debug(f"Ingesting passages from {input_dir}...")
    ingest_passages(input_dir=input_dir)


if __name__ == "__main__":
    main()

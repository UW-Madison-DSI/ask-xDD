import json
import logging
import os
from pathlib import Path

import click
import weaviate
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm

from askem.preprocessing import HaystackPreprocessor, Preprocessor
from askem.retriever import get_client

logging.basicConfig(level=logging.DEBUG)


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
    input_dir: str, topic: str, preprocessor: Preprocessor = None
) -> None:
    """Ingest passages into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    client = get_client()
    input_files = Path(input_dir).glob("**/*.txt")

    for input_file in tqdm(list(input_files)):
        passages = preprocessor.run(input_file=input_file, topic=topic)

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
@click.option("--topic", help="Topic.", type=str)
def main(init: bool, input_dir: str, topic: str):
    """Main entrypoint.

    Usage:
    python -m ./askem.deploy --init --input-dir data/covid_qa --topic covid

    """

    logging.debug(f"Initializing passage retriever... with {init=}")
    if init:
        init_retriever(force=True)

    logging.debug(f"Ingesting passages from {input_dir}...")
    import_passages(input_dir=input_dir, topic=topic)


if __name__ == "__main__":
    main()

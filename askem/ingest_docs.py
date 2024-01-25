import logging
import os
import pickle
from pathlib import Path

import click
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

from askem.preprocessing import ASKEMPreprocessor, HaystackPreprocessor
from askem.retriever.data_models import ClassName, DocType, Topic
from askem.utils import count_docs

load_dotenv()


def import_documents(
    client: weaviate.Client,
    input_dir: str,
    class_name: ClassName,
    topic: Topic,
    doc_type: DocType,
    duplicate_check: bool = True,
    preprocessor: ASKEMPreprocessor = None,
) -> None:
    """Ingest documents into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    input_files = Path(input_dir).glob("**/*.txt")

    existing_docids = set([])
    if duplicate_check:
        if not os.path.exists("document_counts.pkl"):
            print("Gathering document IDs ingested...")
            count_docs(client)
        with open("document_counts.pkl", "rb") as fin:
            docs = pickle.load(fin)
        for _, docids in docs.items():
            existing_docids = existing_docids.union(docids)

    # Batching
    client.batch.configure(batch_size=32, dynamic=True)
    with client.batch as batch:
        # article level loop (each file)
        for input_file in tqdm(list(input_files)):
            # Skip if document exists in the weaviate docid dump.
            if duplicate_check:
                if input_file.name.replace(".txt", "") in existing_docids:
                    continue
            docs = preprocessor.run(
                input_file=input_file, topic=topic, doc_type=doc_type
            )

            # paragraph level loop (each paragraph)
            for doc in docs:
                batch.add_data_object(data_object=doc, class_name=class_name)


@click.command()
@click.option("--input-dir", help="Input directory.", type=str)
@click.option(
    "--topic",
    help="Topic.",
    type=click.Choice([e.value for e in Topic], case_sensitive=False),
)
@click.option(
    "--doc-type",
    help="Document type.",
    type=click.Choice([e.value for e in DocType], case_sensitive=False),
)
@click.option("--weaviate-url", help="Weaviate URL.", type=str, required=False)
@click.option(
    "--duplicate-check",
    help="Protect against ingesting documents. Compares against the dumped docid list if it exists (and creates it if it doesn't)",
    type=bool,
    default=True,
    required=False,
)
def main(
    input_dir: str, topic: str, doc_type: str, weaviate_url: str, duplicate_check: bool
) -> None:
    """Ingesting data into weaviate database.

    Usage:
    python -m ./askem.ingest_docs --input-dir data/covid_qa --topic covid --doc-type paragraph --weaviate-url http://url:8080

    """

    assert doc_type in [e.value for e in DocType]
    assert topic in [e.value for e in Topic]
    assert Path(input_dir).exists()

    auth = weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_APIKEY"))
    weaviate_client = weaviate.Client(weaviate_url, auth)

    logging.debug(f"Ingesting passages from {input_dir}...")
    import_documents(
        client=weaviate_client,
        class_name="Passage",
        input_dir=input_dir,
        topic=topic,
        doc_type=doc_type,
        duplicate_check=duplicate_check,
    )


if __name__ == "__main__":
    main()

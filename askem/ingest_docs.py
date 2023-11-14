import logging
import os
from pathlib import Path

import click
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

from askem.preprocessing import ASKEMPreprocessor, HaystackPreprocessor
from askem.retriever.data_models import ClassName, DocType, Topic

load_dotenv()


def import_documents(
    client: weaviate.Client,
    input_dir: str,
    class_name: ClassName,
    topic: Topic,
    doc_type: DocType,
    preprocessor: ASKEMPreprocessor = None,
) -> None:
    """Ingest documents into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    input_files = Path(input_dir).glob("**/*.txt")

    # Batching
    client.batch.configure(batch_size=32, dynamic=True)
    with client.batch as batch:
        # article level loop (each file)
        for input_file in tqdm(list(input_files)):
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
def main(input_dir: str, topic: str, doc_type: str, weaviate_url: str) -> None:
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
    )


if __name__ == "__main__":
    main()

import logging

import click
from dotenv import load_dotenv

import askem.preprocessing
from askem.retriever import get_client, init_retriever

load_dotenv()
logging.basicConfig(level=logging.DEBUG)


def import_documents(
    input_dir: str,
    topic: str,
    doc_type: str,
    preprocessor: askem.preprocessing.ASKEMPreprocessor = None,
    client=None,
) -> None:
    """Ingest documents into Weaviate."""

    import askem.preprocessing

    if preprocessor is None:
        preprocessor = askem.preprocessing.HaystackPreprocessor()

    if client is None:
        client = get_client()

    input_files = Path(input_dir).glob("**/*.txt")

    for input_file in tqdm(list(input_files)):
        docs = preprocessor.run(input_file=input_file, topic=topic, doc_type=doc_type)

        for doc in docs:
            client.data_object.create(
                data_object=doc, class_name="Passage"
            )  # TODO: Should rename to Document if safe.


@click.command()
@click.option(
    "--init",
    help="Force re-initialization, It will wipe everything!",
    type=bool,
    is_flag=True,
)
@click.option("--input-dir", help="Input directory.", type=str)
@click.option("--topic", help="Topic.", type=str)
@click.option("--doc-type", help="Document type.", type=str)
@click.option(
    "--weaviate-url", help="Weaviate URL.", type=str, default="http://localhost:8080"
)
def main(init: bool, input_dir: str, topic: str, doc_type: str, weaviate_url: str):
    """Deployment entrypoint.

    Usage:
    python -m ./askem.deploy --init --input-dir data/covid_qa --topic covid

    """

    assert doc_type in askem.preprocessing.WEAVIATE_DOC_TYPES
    weaviate_client = get_client(url=weaviate_url)

    logging.debug(f"Initializing passage retriever... with {init=}")
    if init:
        init_retriever(force=True, client=weaviate_client)

    logging.debug(f"Ingesting passages from {input_dir}...")
    import_documents(
        input_dir=input_dir, topic=topic, doc_type=doc_type, client=weaviate_client
    )


if __name__ == "__main__":
    main()

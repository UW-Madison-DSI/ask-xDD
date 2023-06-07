import logging

import click
from dotenv import load_dotenv

from askem.preprocessing import WEAVIATE_DOC_TYPES
from askem.retriever import get_client, import_documents, init_retriever

load_dotenv()
logging.basicConfig(level=logging.DEBUG)


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

    assert doc_type in WEAVIATE_DOC_TYPES
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

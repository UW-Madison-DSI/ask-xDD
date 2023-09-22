import logging
from typing import Optional

import click
from dotenv import load_dotenv

from askem.retriever.base import LATEST_SCHEMA_VERSION, get_client, init_retriever

load_dotenv()
logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option("--weaviate-url", help="Weaviate URL.", type=str)
@click.option("--version", help="Schema version", type=int, required=False)
def main(weaviate_url: str = None, version: Optional[int] = None) -> None:
    """Initialize the retriever data structure in weaviate.

    Usage:
    # init latest version schema
    python -m askem.init_class --weaviate-url http://url:8080

    # init specific version schema
    python -m askem.init_class --weaviate-url http://url:8080 --version 2

    """

    if not version:
        version = LATEST_SCHEMA_VERSION

    client = get_client(url=weaviate_url)
    init_retriever(client=client, version=version)


if __name__ == "__main__":
    main()

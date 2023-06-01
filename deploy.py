import logging

import click

from askem.retriever import import_passages, init_retriever

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

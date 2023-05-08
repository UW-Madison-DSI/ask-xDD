import logging
from pathlib import Path

import click
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import PreProcessor, TextConverter

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--input_dir", help="Path to the directory containing the raw text files.", type=str
)
@click.option(
    "--split_length", help="Maximum number of tokens per split.", type=int, default=200
)
@click.option("--recreate_index", help="Recreate the index.", type=bool, default=True)
def main(input_dir: str, split_length: int = 200, recreate_index: bool = True):
    """Preprocess the data for the Askem pipeline."""

    # Create nodes
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=split_length,
        split_respect_sentence_boundary=False,
        split_overlap=5,
    )
    document_store = WeaviateDocumentStore(
        similarity="dot_product", recreate_index=recreate_index
    )

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_node(text_converter, name="text_converter", inputs=["File"])
    pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])
    pipeline.add_node(document_store, name="document_store", inputs=["preprocessor"])

    # Run pipeline
    files = Path(input_dir).glob("**/*.txt")
    pipeline.run_batch(file_paths=[str(file) for file in files])


if __name__ == "__main__":
    main()

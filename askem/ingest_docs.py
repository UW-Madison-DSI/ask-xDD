import logging
from pathlib import Path
from typing import List

import click
from dotenv import load_dotenv
from tqdm import tqdm

from askem.preprocessing import ASKEMPreprocessor, HaystackPreprocessor, update_count
from askem.retriever.base import get_client
from askem.retriever.data_models import ClassName, DocType, Topic
from askem.terms_extractor import MoreThanOneCapStrategy, Strategy, get_blacklist

load_dotenv()


def append_terms(docs: List[dict], extractor: Strategy) -> List[dict]:
    """Append terms to document.

    Term is defined as a word that is all capital letters.
        paragraph_terms_i: top 3 article level terms (at least occurring 3 times), and
        article_terms_i: top 10 paragraph level terms (top 3) to each paragraph.
    """

    for paragraph in docs:
        text = paragraph["text_content"]
        terms = extractor.extract_terms(text)

        if not terms:
            continue

        # Append paragraph level terms
        for i, term in enumerate(terms):
            paragraph[f"paragraph_terms_{i}"] = term

    return docs


def import_documents(
    input_dir: str,
    class_name: ClassName,
    topic: Topic,
    doc_type: DocType,
    preprocessor: ASKEMPreprocessor = None,
    client=None,
) -> None:
    """Ingest documents into Weaviate."""

    if preprocessor is None:
        preprocessor = HaystackPreprocessor()

    if client is None:
        client = get_client()

    input_files = Path(input_dir).glob("**/*.txt")

    # Terms Extractor
    terms_extractor = MoreThanOneCapStrategy(
        min_length=3, min_occurrence=1, top_k=3, blacklist=get_blacklist(topic)
    )

    # Batching
    client.batch.configure(batch_size=32, dynamic=True)
    with client.batch as batch:
        # article level loop (each file)
        for input_file in tqdm(list(input_files)):
            docs = preprocessor.run(
                input_file=input_file, topic=topic, doc_type=doc_type
            )
            docs = append_terms(docs, terms_extractor)

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
    python -m ./askem.deploy --input-dir data/covid_qa --topic covid --doc-type paragraph --weaviate-url http://url:8080

    """

    assert doc_type in [e.value for e in DocType]
    assert topic in [e.value for e in Topic]
    assert Path(input_dir).exists()

    weaviate_client = get_client(url=weaviate_url)

    logging.debug(f"Ingesting passages from {input_dir}...")
    import_documents(
        class_name="Passage",
        input_dir=input_dir,
        topic=topic,
        doc_type=doc_type,
        client=weaviate_client,
    )


if __name__ == "__main__":
    main()

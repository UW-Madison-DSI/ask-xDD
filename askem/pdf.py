from __future__ import annotations
from typing import List
import gzip
import pickle
from typing import Union
from pathlib import Path
from dataclasses import dataclass
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.high_level import extract_text
from .preprocessing import to_sentences, to_chunks


@dataclass
class Paper:
    title: str
    doi: str
    text: str
    sentences: List[str]
    chunks: List[str]
    metadata: dict

    @classmethod
    def load(cls, path: Union[str, Path]) -> Paper:
        """Load from gzip pickle format."""

        with gzip.open(path, "rb") as file:
            return pickle.load(file)

    def save(self, path: Union[str, Path]) -> None:
        """Save into gzip pickle format."""

        with gzip.open(path, "wb") as file:
            pickle.dump(self, file)

    def __str__(self) -> str:
        return f"{self.title} ({self.doi}), full-text length: {len(self.text)}."


def parse_pdf(file_path: Union[str, Path]) -> Paper:
    """Parse PDF file into a `Paper`.

    Args:
        file_path (Union[str, Path]): Path to the PDF file.

    Returns:
        Paper: A `Paper` object.

    Example:
    ```python
    pdfs = pathlib.Path('data/').glob('*.pdf')

    for i, pdf in enumerate(pdfs):
        paper = parse_pdf(pdf)
        paper.save(f'data/parsed/{i}.pkl.gz')
    ```
    """

    text = extract_text(file_path)
    sentences = to_sentences(text)
    chunks = to_chunks(sentences)

    # Extract metadata
    with open(file_path, "rb") as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)
        metadata = document.info[0]

    title = metadata.get("Title", "No title")
    doi = metadata.get("doi")

    return Paper(
        title=title,
        doi=doi,
        text=text,
        sentences=sentences,
        chunks=chunks,
        metadata=metadata,
    )

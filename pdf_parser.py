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
from preprocessing import clean_text


@dataclass
class Paper:
    title: str
    doi: str
    text: List[str]
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
    """Parse PDF file into a `Paper`."""

    text = extract_text(file_path)

    with open(file_path, "rb") as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)

        metadata = document.info[0]
        title = metadata.get("Title", "No title")
        doi = metadata.get("doi")

        return Paper(title=title, doi=doi, text=clean_text(text), metadata=metadata)

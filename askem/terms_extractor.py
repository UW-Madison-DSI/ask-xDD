import logging
import re
import unicodedata
from typing import Any, List, Optional, Protocol

import spacy


class Strategy(Protocol):
    """A strategy for extracting terms from a text."""

    def __init__(self, **kwargs: Any) -> None:
        ...

    def extract_terms(self, text: str, **kwargs) -> Optional[List[str]]:
        ...


def update_count(d: dict, words: Optional[List[str]]) -> None:
    if not words:
        return None

    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1


def get_top_k(d: dict, k: int = 10, min_n: int = 3) -> dict:
    """Get top-k most frequent words in a dictionary."""

    d = {k: v for k, v in d.items() if v >= min_n}
    return sorted(d, key=d.get, reverse=True)[:k]


def strip_punctuation(text: str) -> str:
    return "".join([c for c in text if c.isalnum() or c.isspace()])


def remove_diacritics(text: str) -> str:
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def remove_brackets(text: str) -> str:
    return text.replace("(", " ").replace(")", " ")


def remove_citations(text: str) -> str:
    # Remove the text chunks that
    return re.sub(r"\((?:[A-Za-z‐\s]+(?:et al\.)?, \d{4}(?:; )?)+\)", "", text)


class CapitalizedWordsStrategy:
    """Extracts capitalized words from a text."""

    def __init__(self, min_length: int, min_occurrence: int, top_k: int) -> None:
        self.min_length = min_length
        self.min_occurrence = min_occurrence
        self.top_k = top_k

    @staticmethod
    def preprocessing(text: str) -> str:
        text = strip_punctuation(text)
        text = remove_diacritics(text)
        text = remove_citations(text)
        return text

    def extract_terms(self, text: str) -> Optional[List[str]]:
        text = self.preprocessing(text)
        logging.info(text)

        terms = [
            word
            for word in text.split()
            if word.isupper() and len(word) >= self.min_length
        ]

        if not terms:
            return None

        counts = {}
        update_count(counts, terms)
        return get_top_k(counts, k=self.top_k, min_n=self.min_occurrence)


class MoreThanOneCapStrategy:
    def __init__(self, min_length: int, min_occurrence: int, top_k: int) -> None:
        self.min_length = min_length
        self.min_occurrence = min_occurrence
        self.top_k = top_k

    @staticmethod
    def preprocessing(text: str) -> str:
        text = remove_diacritics(text)
        text = remove_citations(text)
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        return text

    def extract_terms(self, text: str) -> Optional[List[str]]:
        text = self.preprocessing(text)
        logging.info(text)

        terms = []
        for word in text.split(" "):
            if len(word) < self.min_length:
                continue

            n_upper = sum(1 for char in word if char.isupper())
            if n_upper > 1:
                terms.append(word)

        if not terms:
            return None

        counts = {}
        update_count(counts, terms)
        return get_top_k(counts, k=self.top_k, min_n=self.min_occurrence)


class ProperNounStrategy:
    def __init__(self, min_length: int, min_occurrence: int, top_k: int) -> None:
        self.min_length = min_length
        self.min_occurrence = min_occurrence
        self.top_k = top_k
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def preprocessing(text: str) -> str:
        text = remove_diacritics(text)
        text = remove_citations(text)
        return text

    def extract_terms(self, text: str) -> Optional[List[str]]:
        text = self.preprocessing(text)
        logging.info(text)

        terms = []
        for token in self.nlp(text):
            if token.pos_ == "PROPN" and len(token.text) >= self.min_length:
                terms.append(token.text)

        if not terms:
            return None

        counts = {}
        update_count(counts, terms)
        return get_top_k(counts, k=self.top_k, min_n=self.min_occurrence)

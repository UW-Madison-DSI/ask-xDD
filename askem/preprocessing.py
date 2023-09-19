import logging
import unicodedata
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, Union

from haystack import Pipeline
from haystack.errors import HaystackError
from haystack.nodes import PreProcessor, TextConverter
from haystack.schema import Document

logging.basicConfig(level=logging.INFO)

MAX_WORDS = 250
MIN_WORDS = 100
WEAVIATE_DOC_TYPES = [
    "paragraph",
    "figure",
    "table",
]  # Valid values in Weaviate's `type` field

def update_count(d: dict, words: Optional[List[str]]) -> None:
    if not words:
        return None
    
    for word in words:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1

def get_top_k(d: dict, k: int = 10, min_occurrences: int = 1) -> dict:
    """Get top-k most frequent words in a dictionary."""

    d = {k: v for k, v in d.items() if v >= min_occurrences}
    return sorted(d, key=d.get, reverse=True)[:k]


def strip_punctuation(text: str) -> str:
    return "".join([c for c in text if c.isalnum() or c.isspace()])

def remove_diacritics(text: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def get_all_cap_words(text: str, min_length: int = 3, top_k: int = 3) -> list:
    """Get capitalized words in a text, sorted by number of occurrence."""

    text = strip_punctuation(text)
    text = remove_diacritics(text)
    
    words = text.split()
    all_cap_words = [word for word in words if word.isupper() and len(word) >= min_length]

    if not all_cap_words:
        return None
    
    # Count the number of all caps words
    counts = {word: text.count(word) for word in all_cap_words}

    # Return top-k most frequent all caps words
    return sorted(counts, key=counts.get, reverse=True)[:top_k]

class ModifiedPreProcessor(PreProcessor):
    """A modified version of the Haystack PreProcessor."""

    def __init__(self, join_paragraphs: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.join_paragraphs = join_paragraphs

    def clean(
        self,
        document: Union[dict, Document],
        clean_whitespace: bool,
        clean_header_footer: bool,
        clean_empty_lines: bool,
        remove_substrings: Optional[List[str]] = None,
        id_hash_keys: Optional[List[str]] = None,
    ) -> Document:
        if remove_substrings is None:
            remove_substrings = []
        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        if isinstance(document, dict):
            document["id_hash_keys"] = id_hash_keys
            document = Document.from_dict(document)

        if not isinstance(document, Document):
            raise HaystackError(
                "Document must not be of type 'dict' but of type 'Document'."
            )

        if type(document.content) is not str:
            logging.error("Document content is not of type str. Nothing to clean.")
            return document

        if self.join_paragraphs:
            text = self._join_paragraphs(document.content)

        if clean_header_footer:
            text = self._find_and_remove_header_footer(
                text, n_chars=300, n_first_pages_to_ignore=1, n_last_pages_to_ignore=1
            )

        headlines = document.meta["headlines"] if "headlines" in document.meta else []

        if clean_whitespace:
            text, headlines = self._clean_whitespace(text=text, headlines=headlines)

        if clean_empty_lines:
            text, headlines = self._clean_empty_lines(text=text, headlines=headlines)

        for substring in remove_substrings:
            text, _ = self._remove_substring(
                text=text, substring=substring, headlines=headlines
            )

        if text != document.content:
            document = deepcopy(document)
            document.content = text
        if headlines:
            document.meta["headlines"] = headlines

        return document

    @staticmethod
    def _join_paragraphs(text: str) -> str:
        """Join paragraphs that are split across multiple lines."""

        lines = text.splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        processed_lines = []
        current_line = lines[0]
        for next_line in lines[1:]:
            current_line_ended = current_line[-1] in {".", "?", "!"}
            next_line_started = next_line[0].isupper() or next_line[0].isdigit()
            if current_line_ended or next_line_started:
                # New paragraph
                processed_lines.append(current_line)
                current_line = next_line
            else:
                # Continue previous paragraph
                current_line += f" {next_line}"

        processed_lines.append(current_line)  # Add the last accumulated line
        return "\n\n".join(processed_lines)


def clean_paragraphs(paragraphs: List[str]) -> List[str]:
    cleaned_paragraphs = []
    for i in range(len(paragraphs)):
        paragraph = paragraphs[i]
        # print("Paragraph {} (length = {}): {}".format(i, len(paragraph.split(" ")), paragraph))
        if detect_references(paragraph):
            break
        paragraph = remove_section_header(paragraph)
        paragraph = remove_download_remnant(paragraph)
        paragraph = remove_time_remnant(paragraph)
        paragraph = concatenate_incomplete_paragraph(cleaned_paragraphs, paragraph)
        paragraph = remove_short_paragraph(paragraph)
        if paragraph:
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs


def detect_references(text: str) -> bool:
    if text.strip().lower() == "references":
        return True
    if text.strip().lower() == "reference":
        return True
    return False


def remove_section_header(text: str) -> Optional[str]:
    """Remove section header with only capital letters or capital letters and numbers."""
    if not text:
        return None
    words = text.split(" ")
    if all([word.isupper() or word.isdigit() for word in words]):
        return None
    return text


def remove_download_remnant(text: str) -> Optional[str]:
    """Remove useless download pattern like: `Download by: [UW-Madison (GeoDeepDive)]`."""
    if not text:
        return None
    words = text.split(" ")
    short = len(words) < 25
    has_download = "download" in text or "Download" in text
    if short and has_download:
        return None
    return text


def remove_time_remnant(text: str) -> Optional[str]:
    """Remove useless time pattern like: `12:00`."""
    if not text:
        return None
    words = text.split(" ")
    short = len(words) < 25
    pattern = r"\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b"
    has_time = re.search(pattern, text)

    if short and has_time:
        return None
    return text


def remove_short_paragraph(text: str) -> Optional[str]:
    """Remove short paragraphs that are less than 15 words."""
    if not text:
        return None
    words = text.split(" ")
    if len(words) < 15:
        return None
    return text


def concatenate_incomplete_paragraph(
    paragraphs: List[str], paragraph: str
) -> Optional[str]:
    """Concatenate incomplete paragraphs that do not end with a punctuation and is not capitalized"""
    if not paragraph:
        return None
    if len(paragraphs) > 0:
        if not (paragraph[0].isupper() or paragraph[0].isdigit()) and not paragraphs[
            -1
        ][-1] in {".", "?", "!"}:
            paragraphs[-1] = paragraphs[-1] + " " + paragraph
            return None
    return paragraph


def process_paragraphs(paragraphs_before_adjust: List[str]) -> List[str]:
    """Process paragraphs to make sure each paragraph has a proper length."""
    paragraphs_after_adjust = []
    for i in range(len(paragraphs_before_adjust)):
        process_single_paragraph(paragraphs_before_adjust, i, paragraphs_after_adjust)
    return paragraphs_after_adjust


def process_single_paragraph(
    paragraphs_before_adjust: List[str], index: int, paragraphs_after_adjust: List[str]
) -> None:
    """Process a single paragraph to make sure it has a proper length."""
    paragraph = paragraphs_before_adjust[index]
    paragraph_words = paragraph.split(" ")
    num_of_words = len(paragraph_words)

    if MIN_WORDS <= num_of_words and num_of_words <= MAX_WORDS:
        ## if the paragraph is within the proper length
        process_proper_paragraph(paragraphs_after_adjust, paragraph)

    elif num_of_words < MIN_WORDS:
        ## if the paragraph is too short
        process_short_paragraph(
            paragraphs_before_adjust, paragraphs_after_adjust, paragraph, index
        )

    elif num_of_words > MAX_WORDS:
        ## if the paragraph is too long
        process_long_paragraph(paragraphs_after_adjust, paragraph)


def process_proper_paragraph(
    paragraphs_after_adjust: List[str], paragraph: str
) -> None:
    paragraphs_after_adjust.append(paragraph)


def process_short_paragraph(
    paragraphs_before_adjust: List[str],
    paragraphs_after_adjust: List[str],
    paragraph: str,
    index: int,
) -> None:
    if index == len(paragraphs_before_adjust) - 1:
        # if the current paragraph is the last paragraph
        paragraphs_after_adjust.append(paragraph)
    else:
        ## append to the beginning of the next paragraph
        paragraphs_before_adjust[index + 1] = (
            paragraph + "\n" + paragraphs_before_adjust[index + 1]
        )


def process_long_paragraph(paragraphs_after_adjust: List[str], paragraph: str):
    sentences = paragraph.split(". ")
    num_of_sentences = len(sentences)

    if num_of_sentences == 1:
        # if the paragraph has only one sentence
        paragraphs_after_adjust.append(paragraph)
    else:
        # if the paragraph has more than one sentence
        passage_start_index = 0
        while passage_start_index < num_of_sentences:
            new_paragraph, passage_end_index = build_new_paragraph(
                sentences, passage_start_index
            )
            paragraphs_after_adjust.append(new_paragraph)
            passage_start_index = passage_end_index + 1


def build_new_paragraph(sentences: List[str], start_index: int) -> Tuple[str, int]:
    paragraph = sentences[start_index]
    paragraph_length = len(paragraph.split())
    end_index = start_index
    num_of_sentences = len(sentences)

    for sentence_index in range(start_index + 1, num_of_sentences):
        sentence = sentences[sentence_index]
        sentence_words = sentence.split(" ")
        sentence_length = len(sentence_words)
        if paragraph_length + sentence_length > (MAX_WORDS - 50):
            break
        else:
            paragraph += ". " + sentence
            paragraph_length += sentence_length
            end_index = sentence_index

    # add overlap at the end if the paragraph is not the last paragraph and the total length doesn't exceed the limit

    if end_index < num_of_sentences - 1:
        next_sentence = sentences[end_index + 1]
        next_sentence_length = len(next_sentence.split(" "))
        if paragraph_length + next_sentence_length < MAX_WORDS:
            paragraph = paragraph + ". " + next_sentence + "."

    # add overlap at the beginning if the paragraph is the last paragraph, until the total length exceeds the limit
    if end_index == num_of_sentences - 1:
        while start_index > 0 and paragraph_length < MIN_WORDS:
            start_index -= 1
            previous_sentence = sentences[start_index]
            previous_sentence_length = len(previous_sentence.split(" "))
            paragraph = previous_sentence + ". " + paragraph
            paragraph_length += previous_sentence_length

    return paragraph, end_index


def adjust_paragraphs(original_paragraphs: List[str]) -> List[str]:
    cleaned_paragraphs = clean_paragraphs(original_paragraphs)
    adjusted_paragraphs = process_paragraphs(cleaned_paragraphs)
    return adjusted_paragraphs


class ASKEMPreprocessor(Protocol):
    def run(self, input_file: Path, topic: str, doc_type: str) -> List[dict]:
        ...

    @property
    def preprocessor_id(self) -> str:
        ...


class HaystackPreprocessor:
    def __init__(self):
        self.haystack_pipeline = self._get_pipeline()

    @property
    def preprocessor_id(self) -> str:
        return "haystack_v0.0.2"

    @staticmethod
    def _get_pipeline() -> Pipeline:
        text_converter = TextConverter(
            remove_numeric_tables=True, valid_languages=["en"]
        )
        preprocessor = ModifiedPreProcessor(
            join_paragraphs=True,
            clean_whitespace=True,
            clean_header_footer=True,
            clean_empty_lines=False,
            split_by="passage",
            split_length=1,
            split_respect_sentence_boundary=False,
        )
        pipeline = Pipeline()
        pipeline.add_node(text_converter, name="text_converter", inputs=["File"])
        pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])
        return pipeline

    def _process_paragraph_files(self, input_file: str, topic: str) -> List[dict]:
        file_stem = Path(input_file).stem
        results = self.haystack_pipeline.run(file_paths=[input_file])

        outputs = []
        contents = [d.content for d in results["documents"]]
        adjusted_contents = adjust_paragraphs(contents)

        for content in adjusted_contents:
            outputs.append(
                {
                    "preprocessor_id": self.preprocessor_id,
                    "paper_id": file_stem,
                    "type": "paragraph",
                    "topic": topic,
                    "text_content": content,
                }
            )

        return outputs

    def _process_fig_and_table_files(
        self, input_file: str, topic: str, doc_type: str
    ) -> List[dict]:
        input_file = Path(input_file)
        paper_id = input_file.stem.split(".")[0]
        cosmos_object_id = input_file.stem.split(".")[1]

        with open(input_file, "r") as f:
            content = f.read()  # Probably no need to preprocess here.

        outputs = []
        outputs.append(
            {
                "paper_id": paper_id,
                "cosmos_object_id": cosmos_object_id,
                "type": doc_type,
                "topic": topic,
                "text_content": content,
            }
        )
        return outputs

    def run(self, input_file: Path, topic: str, doc_type: str) -> List[dict]:
        """Use haystack preprocessing to preprocess one file.

        Args:
            input_file: Input file path.
            topic: Topic.
            type: Type of the input file (e.g., paragraph, figure).
        """

        assert doc_type in WEAVIATE_DOC_TYPES

        if doc_type == "paragraph":
            return self._process_paragraph_files(input_file, topic)
        else:
            return self._process_fig_and_table_files(input_file, topic, doc_type)

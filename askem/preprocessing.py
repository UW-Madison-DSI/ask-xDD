import logging
import re
from pathlib import Path
from typing import List, Protocol

from haystack import Pipeline
from haystack.nodes import TextConverter
from preprocessor import PreProcessor
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)

MAX_WORDS = 250
MIN_WORDS = 100

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
    if text.strip().lower() == 'references':
        return True
    if text.strip().lower() == 'reference':
        return True
    return False

def remove_section_header(text: str) -> Optional[str]:
    """Remove section header with only Captial letters or Captial letters and numbers."""
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
    pattern = r'\b([01]?[0-9]|2[0-3]):[0-5][0-9]\b'
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

def concatenate_incomplete_paragraph(paragraphs: List[str], paragraph: str) -> Optional[str]:
    """Concatenate incomplete paragraphs that do not end with a punctuation and is not capitalized"""
    if not paragraph:
        return None
    if len(paragraphs) > 0:
        if not (paragraph[0].isupper() or paragraph[0].isdigit())and not paragraphs[-1][-1] in {'.', '?', '!'}:
            paragraphs[-1] = paragraphs[-1] + " " + paragraph
            return None
    return paragraph    

def process_paragraphs(paragraphs_before_adjust: List[str]) -> List[str]:
    """Process paragraphs to make sure each paragraph has a proper length."""
    paragraphs_after_adjust = []
    for i in range(len(paragraphs_before_adjust)):
        process_single_paragraph(paragraphs_before_adjust, i, paragraphs_after_adjust)
    return paragraphs_after_adjust

def process_single_paragraph(paragraphs_before_adjust: List[str], index: int, paragraphs_after_adjust: List[str]) -> None:
    """Process a single paragraph to make sure it has a proper length."""
    paragraph = paragraphs_before_adjust[index]
    paragraph_words = paragraph.split(" ")
    num_of_words = len(paragraph_words)

    if MIN_WORDS <= num_of_words and num_of_words <= MAX_WORDS:
        ## if the paragraph is within the proper length
        process_proper_paragraph(paragraphs_after_adjust, paragraph)
    
    elif num_of_words < MIN_WORDS:
        ## if the paragraph is too short
        process_short_paragraph(paragraphs_before_adjust, paragraphs_after_adjust, paragraph, index)
    
    elif num_of_words > MAX_WORDS:
        ## if the paragraph is too long
        process_long_paragraph(paragraphs_after_adjust, paragraph)

def process_proper_paragraph(paragraphs_after_adjust: List[str], paragraph: str) -> None:
    paragraphs_after_adjust.append(paragraph)

def process_short_paragraph(paragraphs_before_adjust: List[str], paragraphs_after_adjust: List[str], paragraph: str, index: int) -> None:
    if index == len(paragraphs_before_adjust) - 1:
        # if the current paragraph is the last paragraph
        paragraphs_after_adjust.append(paragraph)
    else:
        ## append to the begining of the next paragraph
        paragraphs_before_adjust[index+1] = paragraph + "\n" + paragraphs_before_adjust[index+1]

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
            new_paragraph, passage_end_index = build_new_paragraph(sentences, passage_start_index)
            paragraphs_after_adjust.append(new_paragraph)
            passage_start_index = passage_end_index + 1

def build_new_paragraph(sentences: List[str], start_index: int) -> Tuple[str, int]:
    paragraph = sentences[start_index]
    paragraph_length = len(paragraph.split())
    end_index = start_index
    num_of_sentences = len(sentences)

    for sentence_index in range(start_index + 1, num_of_sentences):
        sentence  = sentences[sentence_index]
        sentence_words = sentence.split(" ")
        sentence_length = len(sentence_words)
        if paragraph_length + sentence_length > (MAX_WORDS - 50):
            break
        else:
            paragraph += ". " + sentence
            paragraph_length += sentence_length
            end_index = sentence_index

    # add overlap at the end if the paragprah is not the last paragraph and the total length doesn't exceed the limit
    
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

class Preprocessor(Protocol):
    def run(self, input_dir: str, topic: str) -> List[dict]:
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
        text_converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
        preprocessor = PreProcessor(
            clean_whitespace=True,
            clean_header_footer=True,
            clean_empty_lines=False,
            split_by="passage",
            split_length=1,
            split_respect_sentence_boundary=False,
            # split_overlap=5,
        )
        pipeline = Pipeline()
        pipeline.add_node(text_converter, name="text_converter", inputs=["File"])
        pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])
        return pipeline

    def run(self, input_file: Path, topic: str) -> List[dict]:
        """Use haystack preprocessing to preprocess one file."""

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

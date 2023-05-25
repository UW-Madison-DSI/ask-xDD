import logging
import re
from pathlib import Path
from typing import List, Protocol

from haystack import Pipeline
# from haystack.nodes import PreProcessor, TextConverter
from haystack.nodes import TextConverter
from preprocessor import PreProcessor

logging.basicConfig(level=logging.INFO)

def contains_time(s):
    # Regex pattern for time in the format XX:XX
    pattern = r'\b([01]\d|2[0-3]):([0-5]\d)\b'
    match = re.search(pattern, s)

    return match is not None  # Returns True if a match is found, False otherwise

def contains_download(s):
    return "download" in s or "Download" in s

def adjust_paragraphs(original_paragraphs: List[str], min_words=100, max_words=250) -> List[str]:
    paragraphs = []
    adjusted_paragraphs = []
    for i in range(len(original_paragraphs)):
        paragraph = original_paragraphs[i]
        paragraph_words = paragraph.split(" ")
        num_of_words = len(paragraph_words)
        if num_of_words < 25 and (contains_time(paragraph) or contains_download(paragraph)):
            continue

        if i > 0:
            if not paragraph[0].isupper() and not paragraphs[-1][-1] in {'.', '?', '!'}:
                paragraphs[-1] = paragraphs[-1] + " " + paragraph
                continue

        if num_of_words < 10:
            continue

        paragraphs.append(paragraph)

    for i in range(len(paragraphs)):
        paragraph = paragraphs[i]
        paragraph_words = paragraph.split(" ")
        num_of_words = len(paragraph_words)

        if min_words <= num_of_words and num_of_words <= max_words:
            adjusted_paragraphs.append(paragraph)
        elif num_of_words < min_words:
            if i == len(paragraphs) - 1:
                adjusted_paragraphs.append(paragraph)
            else:
                ## append to the begining of the next paragraph
                next_paragraph = paragraphs[i+1]
                paragraphs[i+1] = paragraph + "\n" + next_paragraph

        elif num_of_words > max_words:
            sentences = paragraph.split(". ")
            num_of_sentences = len(sentences)

            if num_of_sentences == 1:
                # if the paragraph has only one sentence
                adjusted_paragraphs.append(paragraph)
            else:
                # if the paragraph has more than one sentence
                passage_start_index = 0
                while passage_start_index < num_of_sentences:
                    new_paragraph = sentences[passage_start_index]
                    new_paragraph_length = len(sentences[passage_start_index].split())
                    passage_end_index = passage_start_index
                    for sentence_index in range(passage_start_index + 1, num_of_sentences):
                        sentence  = sentences[sentence_index]
                        sentence_words = sentence.split(" ")
                        sentence_length = len(sentence_words)
                        if new_paragraph_length + sentence_length > (max_words - 50):
                            break
                        else:
                            new_paragraph += ". " + sentence
                            new_paragraph_length += sentence_length
                            passage_end_index = sentence_index
                    
                    if passage_end_index < num_of_sentences - 1 and len(sentences[passage_end_index + 1].split(" ")) < 30:
                        new_paragraph = new_paragraph + ". " + sentences[passage_end_index + 1] + "."
                    
                    if passage_end_index == num_of_sentences - 1:
                        while new_paragraph_length < min_words:
                            passage_start_index -= 1
                            new_paragraph = sentences[passage_start_index] + ". " + new_paragraph
                            new_paragraph_length += len(sentences[passage_start_index].split(" "))

                    adjusted_paragraphs.append(new_paragraph)
                    passage_start_index = passage_end_index + 1

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

import logging
from pathlib import Path
from typing import List, Protocol

from haystack import Pipeline
from haystack.nodes import PreProcessor, TextConverter

logging.basicConfig(level=logging.INFO)


class Preprocessor(Protocol):
    def run(self, input_dir: str) -> List[dict]:
        ...


class HaystackPreprocessor:
    def __init__(self):
        self.haystack_pipeline = self._get_pipeline()

    @staticmethod
    def _get_pipeline() -> Pipeline:
        text_converter = TextConverter()
        preprocessor = PreProcessor(
            clean_whitespace=True,
            clean_header_footer=True,
            clean_empty_lines=True,
            split_by="word",
            split_length=200,
            split_respect_sentence_boundary=False,
            split_overlap=5,
        )
        pipeline = Pipeline()
        pipeline.add_node(text_converter, name="text_converter", inputs=["File"])
        pipeline.add_node(preprocessor, name="preprocessor", inputs=["text_converter"])
        return pipeline

    def run_one(self, input_file: Path) -> List[dict]:
        """Use haystack preprocessing to preprocess one file."""

        file_stem = Path(input_file).stem
        results = self.haystack_pipeline.run(file_paths=[input_file])

        # Extract only stem and text split
        outputs = []
        for d in results["documents"]:
            outputs.append(
                {"paper_id": file_stem, "type": "paragraph", "text_content": d.content}
            )

        return outputs

    def run(self, input_dir: str) -> List[dict]:
        """Use haystack preprocessing to preprocess alls file."""

        input_files = Path(input_dir).glob("**/*.txt")
        outputs = []
        for input_file in input_files:
            outputs.extend(self.run_one(input_file))
        return outputs

import os
from dataclasses import dataclass
from typing import List

import weaviate


def get_client() -> weaviate.Client:
    """Returns a weaviate client."""

    apikey = os.getenv("WEAVIATE_APIKEY")
    url = os.getenv("WEAVIATE_URL")
    return weaviate.Client(url, weaviate.auth.AuthApiKey(apikey))


@dataclass
class Paragraph:
    paper_id: str  # xdd document id
    text: str  # paragraph text
    distance: float  # distance metric of the paragraph


def to_paragraph(result: dict) -> Paragraph:
    """Convert a weaviate result to a `Paragraph`."""

    return Paragraph(
        paper_id=result["paper_id"],
        text=result["text_content"],
        distance=result["_additional"]["distance"],
    )


def get_paragraphs(
    client: weaviate.Client,
    question: str,
    top_k: int = 5,
    distance: float = 0.5,
    topic: str = None,
    preprocessor_id: str = None,
) -> List[Paragraph]:
    """Ask a question to retriever and return a list of relevant `Paragraph`.

    Args:
        client: Weaviate client.
        query: Query string.
        limit: Max number of paragraphs to return. Defaults to 5.
        distance: Max distance of the paragraph. Defaults to 0.5.
        topic: Topic filter of the paragraph. Defaults to None (No filter).
        preprocessor_id: Preprocessor filter of the paragraph. Defaults to None (No filter).
    """

    # Get weaviate results
    results = (
        client.query.get(
            "Passage", ["paper_id", "text_content", "topic", "preprocessor_id"]
        )
        .with_near_text({"concepts": [question], "distance": distance})
        .with_additional(["distance"])
    )

    if topic is not None:
        filter = {"path": ["topic"], "operator": "Equal", "valueText": topic}
        results = results.with_where(filter)

    if preprocessor_id is not None:
        filter = {
            "path": ["preprocessor_id"],
            "operator": "Equal",
            "valueText": preprocessor_id,
        }
        results = results.with_where(filter)

    results = results.with_limit(top_k).do()

    # Convert results to Paragraph and return
    return [to_paragraph(result) for result in results["data"]["Get"]["Passage"]]

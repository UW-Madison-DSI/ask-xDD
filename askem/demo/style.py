import os
from typing import Optional
from askem.preprocessing import WEAVIATE_DOC_TYPES

COSMOS_URL = os.getenv("COSMOS_URL")


def highlight(text: str, start: int, end: int) -> str:
    """Highlight section in text."""
    return text[:start] + "**:red[" + text[start:end] + "]**" + text[end:]


def to_url(cosmos_object_id: str) -> str:
    """Convert cosmos object id to URL."""
    return f"{COSMOS_URL}/{cosmos_object_id}"


def to_html(
    doc_type: str,
    text: str,
    generator_answer: Optional[dict] = None,
    cosmos_object_id: Optional[str] = None,
) -> str:
    """Format text to HTML output."""

    # TODO: Move checking to tests
    assert doc_type in WEAVIATE_DOC_TYPES
    # if doc_type in ["figure", "table"]:
    #     assert cosmos_object_id is not None

    if generator_answer is None:
        html = text
    else:
        html = highlight(text, generator_answer["start"], generator_answer["end"])

    # Append link to figure / table object
    if doc_type in ["figure", "table"]:
        html += "<br>"
        html += f"<a href='{to_url(cosmos_object_id)}'>Link</a>"

    return html

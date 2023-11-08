import os
from typing import Optional

import requests

COSMOS_URL = os.getenv("COSMOS_URL")


def highlight(text: str, start: int, end: int) -> str:
    """Highlight section in text."""
    return text[:start] + "**:red[" + text[start:end] + "]**" + text[end:]


def to_url(cosmos_object_id: str) -> str:
    """Convert cosmos object id to URL."""
    return f"{COSMOS_URL}/{cosmos_object_id}"


def get_image_bytes(cosmos_object_id: str) -> str:
    resp = requests.get(to_url(cosmos_object_id))
    result = resp.json()
    try:
        img_bytes = result["success"]["data"][0]["properties"]["image"]
        return img_bytes
    except:
        return None


def to_html(
    doc_type: str,
    text: str,
    generator_answer: Optional[dict] = None,
    cosmos_object_id: Optional[str] = None,
) -> str:
    """Format text to HTML output."""

    if generator_answer is None:
        html = text
    else:
        html = highlight(text, generator_answer["start"], generator_answer["end"])

    image_css = """
        <style>
            .img_container {
                display: flex;
                justify-content: center;
            }
            .img_container img {
                max-width: 100%;
            }
        </style>
    """

    # Append link to figure / table object
    if doc_type in ["figure", "table"]:
        html += "<br>"

        # Get JPEG
        img = get_image_bytes(cosmos_object_id)
        if img is not None:
            html += image_css
            html += "<br>"
            html += "<div class='img_container'>"
            html += f"<img src='data:image/jpg;base64,{get_image_bytes(cosmos_object_id)}' />"
            html += "</div>"

        # Add hyperlink
        html += f"<a href='{to_url(cosmos_object_id)}'>Link</a>"
    return html

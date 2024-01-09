import ast
import os
from typing import AsyncGenerator

from httpx import AsyncClient


async def query_react(
    question: str,
    topic: str,
    top_k: int,
    model_name: str,
    screening_top_k: int = None,
) -> AsyncGenerator[dict, None]:
    """Access react in streaming mode.

    Usage:
    ```
    async for chunk in query_react(
        question="What is COVID-19?",
        topic="covid",
        top_k=3,
        model_name="gpt-4-1106-preview",
        screening_top_k=1000,
    ):
        print(chunk)
    ```
    """

    headers = {
        "Api-Key": os.getenv("RETRIEVER_APIKEY"),
    }

    data = {
        "question": question,
        "topic": topic,
        "top_k": top_k,
        "model_name": model_name,
        "doc_type": "paragraph",  # TODO: Move this to UI
        "screening_top_k": screening_top_k,
    }

    url = os.getenv("RETRIEVER_URL") + "/react_streaming"

    async with AsyncClient(headers=headers, timeout=300).stream(
        "POST", url, json=data
    ) as response:
        async for chunk in response.aiter_raw():
            chunk = chunk.decode("utf-8")
            if chunk:  # Only yield non-empty chunks
                yield ast.literal_eval(chunk)

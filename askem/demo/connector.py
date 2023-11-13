import os
from typing import AsyncGenerator

from httpx import AsyncClient


async def query_react(
    question: str,
    topic: str,
    top_k: int,
    model_name: str,
    screening_top_k: int = None,
) -> AsyncGenerator[str, None]:
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
        "screening_top_k": screening_top_k,
    }

    async with AsyncClient(headers=headers, timeout=300).stream(
        "POST", os.getenv("ASYNC_REACT_URL"), json=data
    ) as response:
        async for chunk in response.aiter_text():
            if chunk:  # Only yield non-empty chunks
                yield chunk.lstrip("data: ")

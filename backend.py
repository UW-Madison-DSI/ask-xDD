from typing import List
from concurrent.futures import ThreadPoolExecutor
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def _compress(chunk: str) -> str:
    """Compresses a long text to a shorter version."""

    instruction = "Summarize the upcoming passage into key points precisely."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": chunk},
        ],
    )

    return response.choices[0].message.content  # type: ignore


def compress(chunks: List[str]) -> List[str]:
    """Compress a set of long text into shorter ones."""

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_compress, chunks))

    return results


def get_answer(context: str, question: str) -> str:
    """Use Chat-GPT to get answer to a question given a context."""

    instruction = (
        "Study the upcoming passage carefully and answer the questions that follows."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": context + "Do you understand the passage?"},
            {
                "role": "user",
                "content": question
                + " Provide your answer in a precise and concise manner.",
            },
        ],
    )
    return response.choices[0].message.content  # type: ignore

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import openai
import tiktoken

ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


def count_token(text: str) -> int:
    """Count GPT tokens using gpt-3.5-turbo model."""
    n = len(ENCODING.encode(text))
    return n


def split_context(text: str, max_tokens: int = 3000) -> List[str]:
    """Split text into fix length."""

    tokens = ENCODING.encode(text)
    n = len(tokens)
    chunks = -(n // -max_tokens)  # Ceiling (upside down floor division)
    splits = [tokens[max_tokens * i : max_tokens * (i + 1)] for i in range(chunks)]
    return [ENCODING.decode(x) for x in splits]


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


def get_answer(context: str, question: str, verbose: bool = False) -> str:
    """Use Chat-GPT to get answer to a question given a context."""

    instruction = "Study the upcoming passages carefully and answer the questions that follows. \
        If cannot find the information in the passages, just answer you don't know."

    short_contexts = split_context(context)

    messages = []
    messages.append({"role": "system", "content": instruction})
    for short_context in short_contexts:
        messages.append(
            {
                "role": "user",
                "content": "passage: " + short_context + "Do you understand?",
            }
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", messages=messages
        )
        if verbose:
            print(f"Intermediate response: {response.choices[0].message.content}")

        # reset message
        messages = []

    messages.append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301", messages=messages
    )
    return response.choices[0].message.content  # type: ignore

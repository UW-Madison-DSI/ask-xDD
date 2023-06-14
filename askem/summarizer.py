import os
from typing import List

import openai


def summarize(question: str, contexts: List[str]) -> str:
    """Compresses a long text to a shorter version."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    instruction = "Answer the question based on the contexts. If there is no answer in context, say 'no answer'."
    prompt = f"Question: {question}{os.linesep} Context: {os.linesep.join(contexts)}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content

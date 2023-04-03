import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import openai
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, LongT5ForConditionalGeneration

openai.api_key = os.getenv("OPENAI_API_KEY")

############################## GPT ##############################


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


############################## Embedding ##############################


def _mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""

    embeddings = model_output["last_hidden_state"]
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
    n = torch.clamp(expanded_mask.sum(1), min=1e-9)
    return torch.sum(embeddings * expanded_mask, 1) / n


def to_embeddings(
    sentences: List[str], model: str = "sentence-transformers/all-mpnet-base-v2"
) -> torch.Tensor:
    """Convert a text into an embedding tensor with 768 units."""

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model)

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute forward pass
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])

    # Normalize embeddings
    return F.normalize(embeddings, p=2, dim=1)


def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the dot-product between two tensors."""

    a = a.unsqueeze(0) if len(a.shape) == 1 else a
    b = b.unsqueeze(0) if len(b.shape) == 1 else b
    return a @ b.T


############################## Long-T5 ##############################


def _t5(x: str, model_name: str = "google/long-t5-tglobal-base") -> str:
    """Summarize sentences using LLM."""

    # Tokenize input sentences
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs_dict = tokenizer(
        x,
        max_length=16384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Generate summary
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    y = model.generate(
        inputs_dict.input_ids, attention_mask=inputs_dict.attention_mask, max_length=512
    )
    return tokenizer.decode(y[0])


def summarize(sentences: List[str], **kwargs) -> str:
    """Summarize sentences using LLM."""

    TEMPLATE = f"summarize: {'.'.join(sentences)}"
    return _t5(TEMPLATE.format(sentences), **kwargs)


def prompt_sum(article: str, question: str, **kwargs) -> str:
    """Prompt-based text summary using LLM."""

    TEMPLATE = f"answer_me: {question} context: {article}"
    return _t5(TEMPLATE.format(question=question, article=article), **kwargs)

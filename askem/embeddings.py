from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging."""

    embeddings = model_output["last_hidden_state"]
    expanded_mask = attention_mask.unsqueeze(-1).expand(embeddings.size())
    n = torch.clamp(expanded_mask.sum(1), min=1e-9)
    return torch.sum(embeddings * expanded_mask, 1) / n


def to_embeddings(
    sentences: List[str], model: str = "sentence-transformers/all-mpnet-base-v2"
) -> torch.Tensor:
    """Convert a text into an embedding tensor with 768 dimensions."""

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

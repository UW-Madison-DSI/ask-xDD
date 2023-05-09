import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


def get_passage_embedding_from_source(text: str) -> torch.Tensor:
    """Get the passage embedding from the HuggingFace source."""
    # model_name = "vblagoje/dpr-ctx_encoder-single-lfqa-wiki"
    model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    embeddings = model(input_ids).pooler_output
    return embeddings

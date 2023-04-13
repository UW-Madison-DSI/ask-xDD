from transformers import (
    AutoTokenizer,
    LongT5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)


def format_qa(question: str, context: str) -> str:
    """Format question and context for T5."""
    return f"question: {question} context: {context}"


def t5(x: str, model_name: str = "google/long-t5-tglobal-base") -> str:
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


def mt0(x: str, model_name: str = "bigscience/mt0-base") -> str:
    """Summarize sentences using LLM."""

    # Tokenize input sentences
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs_dict = tokenizer(
        x,
        max_length=16384,
        padding="max_length",
        return_tensors="pt",
    )

    # Generate summary
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    y = model.generate(
        inputs_dict.input_ids, attention_mask=inputs_dict.attention_mask, max_length=512
    )
    return tokenizer.decode(y[0])

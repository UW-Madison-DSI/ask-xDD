DEBUG = True

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Data
dataset = load_dataset("covid_qa_deepset", split="train")
if DEBUG:
    dataset = dataset.select(range(10))

# Model and Tokenizer
# model_name = "bigscience/mt0-small"
model_name = "google/long-t5-tglobal-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = dataset[0]["context"]
question = dataset[0]["question"]
answer = dataset[0]["answers"]["text"][0]
inputs = tokenizer(
    f"{context}{tokenizer.pad_token}{question}", text_target=answer, return_tensors="pt"
)

print(model(**inputs))

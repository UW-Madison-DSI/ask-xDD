from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LongT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from askem.data import COVID_QA


def main():
    """Train a model."""

    model_name = "google/long-t5-tglobal-base"

    dataset = COVID_QA.train_test_split(test_size=0.2, seed=2023)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    train_data = dataset["train"]
    test_data = dataset["test"]

    def preprocess(examples: Dataset) -> dict:
        """Preprocess dataset.
        Args:
            example: A dict with keys ['context', 'question', 'answers'].

        Returns:
            model_input: with dict_keys(['input_ids', 'attention_mask', 'labels']).
        """

        questions = examples["question"]
        contexts = examples["context"]
        answers = [ans["text"][0] for ans in examples["answers"]]

        # Inputs
        input_texts = [
            f"question: {q} context: {c}" for q, c in zip(questions, contexts)
        ]
        model_inputs = tokenizer(input_texts, max_length=16384, truncation=True)

        # Labels
        labels = tokenizer(answers, max_length=1000, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    test_data = test_data.map(preprocess, batched=True)
    test_data = test_data.select_columns(["input_ids", "attention_mask", "labels"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=test_data,  # DEBUG MODE
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()

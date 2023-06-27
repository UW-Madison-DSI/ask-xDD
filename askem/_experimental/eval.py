import bert_score as score
import numpy as np
from datasets import Dataset

from askem.utils import wrap_print


def quick_eval(dataset: Dataset) -> Dataset:
    """Evaluate the dataset using BERTScore."""

    precision, recall, f1 = score(dataset["y_true"], dataset["y_pred"], lang="en")
    dataset = dataset.add_column("precision", precision.numpy())
    dataset = dataset.add_column("recall", recall.numpy())
    dataset = dataset.add_column("f1", f1.numpy())
    return dataset


def showcase(dataset, generator, idx=None):
    """Showcase the generator on a random example from the dataset."""
    if idx is None:
        idx = np.random.randint(len(dataset))

    # wrap_print(f"Context: {dataset[idx]['context']}")
    wrap_print(f"Question {idx}: {dataset[idx]['question']}")
    print()

    wrap_print("Correct answer:")
    wrap_print(dataset[idx]["answers"]["text"][0])
    print()

    wrap_print("Predicted answer:")
    y = generator(question=dataset[idx]["question"], context=dataset[idx]["context"])
    wrap_print(y["answer"])

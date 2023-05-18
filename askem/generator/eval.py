import bert_score as score
from datasets import Dataset

def quick_eval(dataset: Dataset) -> Dataset:
    """Evaluate the dataset using BERTScore."""

    precision, recall, f1 = score(dataset["y_true"], dataset["y_pred"], lang="en")
    dataset = dataset.add_column('precision', precision.numpy())
    dataset = dataset.add_column('recall', recall.numpy())
    dataset = dataset.add_column('f1', f1.numpy())
    return dataset

import random
from datasets import load_dataset


def get_covid_qa():
    """Get the COVID-QA dataset."""
    dataset = load_dataset("covid_qa_deepset")
    return dataset["train"]


COVID_QA = get_covid_qa()


def get_example(dataset, id=None):
    """Get an example from a dataset."""

    if id is None:
        id = random.randint(0, len(dataset))
    data = dataset[id]
    columns = ["context", "question", "is_impossible", "answers"]
    return {x: data[x] for x in columns}

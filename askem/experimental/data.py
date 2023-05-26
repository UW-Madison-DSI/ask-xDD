import random

import pandas as pd
from datasets import Dataset, load_dataset
from peewee import IntegerField, Model, SqliteDatabase, TextField


def get_covid_qa(split: str = "test") -> Dataset:
    """Get the COVID-QA dataset with train/test split."""
    dataset = load_dataset("covid_qa_deepset")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=2023)

    if split:
        return dataset[split]


BENCH_DB = SqliteDatabase("data/benchmark.db")


def get_example(dataset, id=None):
    """Get an example from a dataset."""

    if id is None:
        id = random.randint(0, len(dataset))
    data = dataset[id]
    columns = ["context", "question", "is_impossible", "answers"]
    return {x: data[x] for x in columns}


class TestResults(Model):
    """ORM for all results."""

    id = IntegerField(primary_key=True)
    model_id = TextField()
    context = TextField()
    question = TextField()
    true_answer = TextField()
    pred_answer = TextField()

    class Meta:
        database = BENCH_DB


def to_df(model: Model) -> pd.DataFrame:
    """Convert a model to a dataframe."""

    return pd.DataFrame(list(model.select().dicts()))

import random

import pandas as pd
from datasets import load_dataset
from peewee import IntegerField, Model, SqliteDatabase, TextField


def get_covid_qa():
    """Get the COVID-QA dataset."""
    dataset = load_dataset("covid_qa_deepset")
    return dataset["train"]


COVID_QA = get_covid_qa()
BENCH_DB = SqliteDatabase("data/benchmark.db")


def get_example(dataset, id=None):
    """Get an example from a dataset."""

    if id is None:
        id = random.randint(0, len(dataset))
    data = dataset[id]
    columns = ["context", "question", "is_impossible", "answers"]
    return {x: data[x] for x in columns}


class GPTBench(Model):
    id = IntegerField(primary_key=True)
    context = TextField()
    question = TextField()
    true_answer = TextField()
    gpt_answer = TextField()

    class Meta:
        database = BENCH_DB


def to_df(model: Model) -> pd.DataFrame:
    """Convert a model to a dataframe."""

    return pd.DataFrame(list(model.select().dicts()))

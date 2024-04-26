import logging
import os
import pickle

import elasticsearch
import requests
import tenacity
from tqdm import tqdm

ID2TOPIC_PATH = "tmp/id2topics.pkl"
SET_NAMES = [
    "climate-change-modeling",
    "criticalmaas",
    "dolomites",
    "geoarchive",
    "xdd-covid-19",
]


def get_text(docid: str) -> str:
    """Get text from ElasticSearch."""

    ES_HOST = os.getenv("ES_HOST")
    ES_CERT_PATH = os.getenv("ES_CERT_PATH")
    ES_USER = os.getenv("ES_USER")
    ES_PASSWORD = os.getenv("ES_PASSWORD")

    if not os.path.exists(ES_CERT_PATH):
        raise Exception("No ES certs provided!")

    client = elasticsearch.Elasticsearch(
        hosts=[ES_HOST],
        request_timeout=30,
        verify_certs=True,
        ca_certs=ES_CERT_PATH,
        basic_auth=(ES_USER, ES_PASSWORD),
    )

    article = client.get(id=docid, index="articles")

    if "contents" not in article["_source"]:
        logging.error(f"No contents found for {docid}")
        return None

    contents = article["_source"]["contents"]
    if isinstance(contents, list):
        if not contents:
            logging.error(f"Contents is empty found for {docid}")
            return None
        contents = contents[0]
    return contents


def invert(d: dict[str : list[str]]) -> dict[str : list[str]]:
    """Invert a dictionary."""
    inverted = {}
    for topic, ids in d.items():
        for id in ids:
            if id not in inverted:
                inverted[id] = [topic]
            elif topic not in inverted[id]:
                inverted[id].append(topic)
    return inverted


@tenacity.retry(wait=tenacity.wait_fixed(60), stop=tenacity.stop_after_attempt(5))
def get_xdd_ids(url: str) -> list[str]:
    """Get all ids for a topic (only one page)."""

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if "success" not in data:
        logging.error(f"Error when calling xdd articles API at {url}: {data}")
        raise ValueError("Unsuccessful xDD request.")

    return data


class DocumentTopicFactory:
    """A factory to create document-topic mapping."""

    def __init__(self, set_names: list[str] | None = None) -> None:
        if set_names is None:
            set_names = SET_NAMES
        self.set_names = set_names

        self.id2topics: dict[str : list[str]] = {}
        self.topic2ids: dict[str : list[str]] = {}

    def run(self) -> dict[str : list[str]]:
        """Run the factory."""
        for set_name in self.set_names:
            print(f"Getting ids for {set_name}")
            self.topic2ids[set_name] = self.get_ids(set_name)
            print(f"Found {len(self.topic2ids[set_name])} ids for {set_name}")

        self.id2topics = invert(self.topic2ids)

        # Write to file
        with open(ID2TOPIC_PATH, "wb") as f:
            pickle.dump(self.id2topics, f)

        return self.id2topics

    def get_ids(self, topic: str) -> list[str]:
        """Get all ids for a topic."""

        # Get total
        url = f"https://xdd.wisc.edu/api/articles?set={topic}&full_results=true&fields=_gddid&per_page=1000"
        data = get_xdd_ids(url)
        hits = data["success"]["hits"]

        progress_bar = tqdm(total=hits, desc=topic, unit="ids")
        ids = []
        while url:
            data = get_xdd_ids(url)
            ids.extend(self.data_to_ids(data))

            if "next_page" not in data["success"]:
                break

            url = data["success"]["next_page"]
            progress_bar.update(len(data["success"]["data"]))
        return ids

    def __str__(self) -> str:
        return "\n".join(
            [f"{topic}: n={len(ids)}" for topic, ids in self.topic2ids.items()]
        )

    @staticmethod
    def data_to_ids(data: dict) -> list[str]:
        """Get all ids from a xDD json response."""

        docs = data["success"]["data"]
        return [doc["_gddid"] for doc in docs]

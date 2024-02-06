import pickle
import requests
import os
import logging
from tqdm import tqdm
import elasticsearch

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
        logging.info(f"docid: {docid} has no contents.")
        return None

    contents = article["_source"]["contents"]
    if isinstance(contents, list):
        if not contents:
            logging.info(f"docid: {docid} has no text in content.")
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


class DocumentTopicFactory:
    """A factory to create document-topic mapping."""

    def __init__(self, set_names: list[str]) -> None:
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

        next_page = f"https://xdd.wisc.edu/api/articles?set={topic}&full_results=true&fields=_gddid"
        progress_bar = tqdm()
        ids = []
        while next_page:
            response = requests.get(next_page)
            data = response.json()
            ids.extend(self._parse_response(data))
            next_page = data["success"]["next_page"]
            progress_bar.update(1)
        return ids

    def __str__(self) -> str:
        return "\n".join(
            [f"{topic}: n={len(ids)}" for topic, ids in self.topic2ids.items()]
        )

    @staticmethod
    def _parse_response(data: dict) -> list[str]:
        """Get all ids from a xDD json response."""

        if "success" not in data:
            raise ValueError("Not a valid xDD response.")

        docs = data["success"]["data"]
        return [doc["_gddid"] for doc in docs]

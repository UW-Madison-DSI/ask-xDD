import argparse
import logging
import os
import pickle
import re
from functools import lru_cache
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import slack_sdk
import weaviate
from dotenv import load_dotenv
from tqdm.contrib.slack import tqdm

from askem.elastic import DocumentTopicFactory, get_text
from askem.preprocessing import HaystackPreprocessor
from askem.utils import get_ingested_ids

logging.basicConfig(
    filename="tmp/error.log", level=logging.ERROR, format="%(asctime)s - %(message)s"
)

load_dotenv()


@lru_cache
def load_cached_id2topics() -> dict[str, list[str]]:
    """Load the cached id2topics.pkl file."""
    with open("tmp/id2topics.pkl", "rb") as f:
        return pickle.load(f)


def process_file(file: Path) -> list[dict]:
    """Process a file and return a list of documents."""

    id2topics = load_cached_id2topics()
    topics = id2topics[file.stem]
    preprocessor = HaystackPreprocessor()
    return preprocessor.run(input_file=file, topics=topics, doc_type="paragraph")


def get_id(text: str) -> str | None:
    return re.findall(r"\b[0-9a-f]{24}\b", text)[0]


def parse_error_log(file: str) -> dict:
    """Sort and deduplicate error logs into `empty`, `api_error`, and `other`."""

    with open(file, "r") as f:
        errors = f.readlines()

    parsed = {"empty": [], "api_error": [], "other": []}

    for line in errors:
        # Find Empty
        if (
            ("Contents is empty found for" in line)
            or ("No text found" in line)
            or ("No contents found" in line)
        ):
            parsed["empty"].append(get_id(line))

        # Find API error
        elif ("Error: ApiError" in line) or ("Error: NotFoundError" in line):
            parsed["api_error"].append(get_id(line))

        # Other (Just get the line instead of id)
        else:
            parsed["other"].append(line)

    # Remove duplicates
    for k, v in parsed.items():
        parsed[k] = list(set(v))

    return parsed


def update_empty_ids_file(empty_ids_pickle: str, error_log: str) -> None:
    """Update the empty_ids.pkl file with new empty ids from the error log."""

    # Append or create new empty_ids.pkl
    if not Path(empty_ids_pickle).exists():
        empty_ids = []
    else:
        with open(empty_ids_pickle, "rb") as f:
            empty_ids = pickle.load(f)

    # Parse error log to get empty ids
    parsed = parse_error_log(error_log)

    empty_ids.extend(parsed["empty"])

    # Deduplicate
    empty_ids = list(set(empty_ids))

    with open(empty_ids_pickle, "wb") as f:
        pickle.dump(empty_ids, f)


def send_slack_message(message: str) -> None:
    """Send a message to TQDM Slack channel for monitoring."""

    TQDM_SLACK_TOKEN = os.getenv("TQDM_SLACK_TOKEN")
    TQDM_SLACK_CHANNEL = os.getenv("TQDM_SLACK_CHANNEL")
    client = slack_sdk.WebClient(TQDM_SLACK_TOKEN)

    try:
        response = client.chat_postMessage(
            channel=TQDM_SLACK_CHANNEL,
            text=message,
        )
        assert response["message"]["text"] == message
    except slack_sdk.errors.SlackApiError as e:
        raise f"Error sending message: {e.response['error']}"


class WeaviateIngester:
    def __init__(
        self,
        client: weaviate.Client,
        class_name: str,
        id2topics: dict[str, list[str]],
        ingested: set[str],
    ) -> None:
        self.client = client
        self.class_name = class_name
        self.id2topics = id2topics
        self.ingested = ingested

        # Misc hardcoded stuff
        self.preprocessor = HaystackPreprocessor()
        self.ingest_folder = Path("tmp/ingest")

        # Make sure no files are left in the ingest folder
        self.purge_ingest_folder()

    @property
    def awaiting_ingest_ids(self) -> list[str]:
        """Get all ids that have not been ingested yet."""
        return sorted(set(self.id2topics.keys()) - set(self.ingested))

    @property
    def files_to_ingest(self) -> list[Path]:
        """Get all files that have not been ingested yet."""
        return sorted(self.ingest_folder.glob("*.txt"))

    def ingest_all(self, batch_size: int) -> None:
        """Ingest all documents to weaviate."""
        progress_bar = tqdm(total=len(self.awaiting_ingest_ids))
        while self.awaiting_ingest_ids:
            n = self.ingest_batch(batch_size=batch_size)
            progress_bar.update(n)

    def ingest_batch(self, batch_size: int) -> int:
        """Ingest a batch of documents to weaviate."""

        # Convert docs to paragraphs
        docids = self.awaiting_ingest_ids[:batch_size]
        self.write_batch_to_file(docids)
        with Pool(4) as p:
            paragraphs = p.map(process_file, self.files_to_ingest)
        paragraphs = list(chain(*paragraphs))  # Flatten

        # Push docs to weaviate
        self.client.batch.configure(batch_size=64, dynamic=True)
        with self.client.batch as batch:
            for doc in paragraphs:
                batch.add_data_object(data_object=doc, class_name=self.class_name)

        self.purge_ingest_folder()
        self.ingested.update(docids)
        return len(docids)

    def purge_ingest_folder(self) -> None:
        """Purge the ingest folder."""
        for file in self.ingest_folder.glob("*.txt"):
            file.unlink()

    def write_batch_to_file(self, batch_ids: list[str]) -> None:
        """Write a batch of ids to a tmp file."""

        self.ingest_folder.mkdir(parents=True, exist_ok=True)

        for docid in batch_ids:
            try:
                text = get_text(docid)
                if not text:
                    logging.error(f"docid: {docid}, Error: No text found.")
                    continue
                with open(f"{self.ingest_folder}/{docid}.txt", "w") as f:
                    f.write(str(text))
            except Exception as e:
                logging.error(f"docid: {docid}, Error: {e}")
                continue


def main():
    """Ingest all documents from Elastic Search to Weaviate.

    Step 1. Get or create a new id2topics.pkl file.
    Step 2. Skip any doc_ids that are stored in empty_ids.pkl.
    Step 3. Skip any existing doc_ids in Weaviate.


    """

    parser = argparse.ArgumentParser(description="Ingest documents to weaviate.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ingest by resuming from tmp/id2topics.pkl.",
    )
    args = parser.parse_args()

    CLASS_NAME = "Paragraph"

    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_APIKEY")),
    )

    # A document-id to topics mapping in all selected `SET_NAMES` created from elastic search service
    if args.resume:
        id2topics = load_cached_id2topics()
    else:
        # Create a new id2topics.pkl file
        id2topics_factory = DocumentTopicFactory()
        id2topics = id2topics_factory.run()
        with open("tmp/id2topics.pkl", "wb") as f:
            pickle.dump(id2topics, f)

    # Skip empty documents (TODO: Remove this after fixing the empty documents in elastic search)
    # Append or create new empty_ids.pkl
    empty_ids_pickle = "tmp/empty_ids.pkl"
    if not Path(empty_ids_pickle).exists():
        empty_ids = []
    else:
        with open(empty_ids_pickle, "rb") as f:
            empty_ids = pickle.load(f)

    id2topics = {k: v for k, v in id2topics.items() if k not in empty_ids}

    # A set of ingested doc_ids from the current weaviate database
    ingested = get_ingested_ids(client=client, class_name=CLASS_NAME)

    ingester = WeaviateIngester(
        client=client,
        class_name=CLASS_NAME,
        id2topics=id2topics,
        ingested=ingested,
    )

    ingester.ingest_all(batch_size=32)

    # Post ingest
    update_empty_ids_file(
        empty_ids_pickle="tmp/empty_ids.pkl", error_log="tmp/error.log"
    )
    send_slack_message("Done ingesting documents to Weaviate.")


if __name__ == "__main__":
    main()

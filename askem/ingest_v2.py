import weaviate
import slack_sdk
import logging
from askem.utils import get_ingested_ids
from askem.preprocessing import HaystackPreprocessor
from pathlib import Path
from tqdm.contrib.slack import tqdm
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from itertools import chain
from dotenv import load_dotenv
from askem.elastic import get_text, DocumentTopicFactory, SET_NAMES
import argparse

logging.basicConfig(
    filename="error.log", level=logging.ERROR, format="%(asctime)s - %(message)s"
)

logging.basicConfig(
    filename="info.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

load_dotenv()
MAX_CPU_COUNT = 4
DOC_TYPE = "paragraph"  # Only support paragraph for now

with open("tmp/id2topics.pkl", "rb") as f:
    ID2TOPICS = pickle.load(f)


def process_file(file: Path) -> list[dict]:
    """Process a file and return a list of documents."""
    topics = ID2TOPICS[file.stem]
    preprocessor = HaystackPreprocessor()
    return preprocessor.run(input_file=file, topics=topics, doc_type=DOC_TYPE)


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
            self.ingest_batch(batch_size=batch_size)
            progress_bar.update(batch_size)

    def ingest_batch(self, batch_size: int) -> None:
        """Ingest a batch of documents to weaviate."""

        # Convert docs to paragraphs
        docids = self.awaiting_ingest_ids[:batch_size]
        self.write_batch_to_file(docids)
        with Pool(4) as p:
            paragraphs = p.map(process_file, self.files_to_ingest)
        paragraphs = list(chain(*paragraphs))  # Flatten

        # Push docs to weaviate
        self.client.batch.configure(batch_size=128, dynamic=True)
        with self.client.batch as batch:
            for doc in paragraphs:
                batch.add_data_object(data_object=doc, class_name=self.class_name)

        self.purge_ingest_folder()
        self.ingested.update(docids)

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
                    logging.info(f"docid: {docid} has no text.")
                    continue
                with open(f"{self.ingest_folder}/{docid}.txt", "w") as f:
                    f.write(text)
            except Exception as e:
                logging.error(f"docid: {docid}, Error: {e}")
                continue


def main():
    """Ingest all documents from ElasticSerarch to Weaviate."""

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
        with open("tmp/id2topics.pkl", "rb") as f:
            id2topics = pickle.load(f)
    else:
        id2topics_factory = DocumentTopicFactory(set_names=SET_NAMES)
        id2topics = id2topics_factory.run()

    # A set of ingested doc_ids from the current weaviate database
    ingested = get_ingested_ids(client=client, class_name=CLASS_NAME)

    ingester = WeaviateIngester(
        client=client,
        class_name=CLASS_NAME,
        id2topics=id2topics,
        ingested=ingested,
    )

    ingester.ingest_all(batch_size=10)


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


if __name__ == "__main__":
    main()

import hashlib
import pickle
import secrets
import string
import textwrap
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def get_hash(text: str) -> str:
    """Get SHA256 hash of text for `hashed_text` property in Weaviate."""
    return hashlib.sha256(text.encode()).hexdigest()


def generate_api_key(length=32) -> str:
    characters = string.ascii_letters + string.digits
    api_key = "".join(secrets.choice(characters) for _ in range(length))
    return api_key


def wrap_print(text, width=150) -> None:
    print(textwrap.fill(text, width=width))


def get_batch_with_cursor(
    client, class_name, class_properties, batch_size, cursor=None
):
    query = (
        client.query.get(class_name, class_properties)
        .with_additional(["id"])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()


def get_ingested_ids(
    client: weaviate.Client,
    class_name: str = "Paragraph",
    batch_size: int = 5000,
) -> set:
    """Get all ingested paper_ids from weaviate."""

    _tmp = client.query.aggregate(class_name).with_meta_count().do()
    n = _tmp["data"]["Aggregate"][class_name][0]["meta"]["count"]

    paper_ids = set()
    cursor = None

    with tqdm(total=n) as progress_bar:
        while True:
            batch = get_batch_with_cursor(
                client,
                class_name,
                ["paper_id"],
                batch_size,
                cursor=cursor,
            )

            # Exit condition
            if len(batch["data"]["Get"][class_name]) == 0:
                break

            objects_list = batch["data"]["Get"][class_name]
            for obj in objects_list:
                paper_ids.add(obj["paper_id"])

            cursor = batch["data"]["Get"][class_name][-1]["_additional"]["id"]
            progress_bar.update(batch_size)

    with open("tmp/ingested.pkl", "wb") as f:
        pickle.dump(paper_ids, f)
    return paper_ids


def get_id_topics_from_weaviate(
    client: weaviate.Client,
    class_name: str = "Paragraph",
    batch_size: int = 5000,
) -> dict:
    """Get all paper_ids and their topics from weaviate."""

    _tmp = client.query.aggregate(class_name).with_meta_count().do()
    n = _tmp["data"]["Aggregate"][class_name][0]["meta"]["count"]

    id2topics = {}
    cursor = None

    with tqdm(total=n) as progress_bar:
        while True:
            batch = get_batch_with_cursor(
                client,
                class_name,
                ["paper_id", "topic_list"],
                batch_size,
                cursor=cursor,
            )

            # Exit condition
            if len(batch["data"]["Get"][class_name]) == 0:
                break

            objects_list = batch["data"]["Get"][class_name]
            for obj in objects_list:
                id2topics[obj["paper_id"]] = obj["topic_list"]

            cursor = batch["data"]["Get"][class_name][-1]["_additional"]["id"]
            progress_bar.update(batch_size)

    with open("tmp/id2topics_weaviate.pkl", "wb") as f:
        pickle.dump(id2topics, f)
    return id2topics

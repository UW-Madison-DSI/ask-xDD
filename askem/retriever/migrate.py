from typing import Protocol

import tqdm
import weaviate


def get_batch_with_cursor(
    client: weaviate.Client,
    class_name: str,
    class_properties: list[str],
    batch_size: int,
    cursor: str | None = None,
) -> dict:
    """Get a batch of data from the source client with a cursor."""
    query = (
        client.query.get(class_name, class_properties)
        .with_additional(["id", "vector"])
        .with_limit(batch_size)
    )

    if cursor is not None:
        return query.with_after(cursor).do()
    else:
        return query.do()


class ResponseParser(Protocol):
    """Response parsing function."""

    def __call__(self, response: dict) -> tuple[dict, list[float], str]:
        ...


def convert_data(response: dict) -> tuple[dict, list[float], str]:
    """Convert a single response from the source to a payload for the destination."""

    vectors = []
    data = response["data"]["Get"]["Passage"]
    for x in data:
        # Unpack additional fields
        additional = x.pop("_additional")
        vectors.append(additional["vector"])
        cursor = additional["id"]  # only need the last one as cursor

        # Restructure the data
        x["doc_type"] = x.pop("type")
        if "cosmos_object_id" in x and x["cosmos_object_id"] is None:
            x.pop("cosmos_object_id")
    return data, vectors, cursor


class MigrationManager:
    def __init__(
        self,
        source_client: weaviate.Client,
        destination_client: weaviate.Client,
        class_name: str,
    ) -> None:
        self.source_client = source_client
        self.destination_client = destination_client
        self.class_name = class_name

    @property
    def source_n(self) -> int:
        """Number of objects in the source."""
        response = (
            self.source_client.query.aggregate(self.class_name).with_meta_count().do()
        )
        return response["data"]["Aggregate"]["Passage"][0]["meta"]["count"]

    def clone(
        self,
        source_properties: list[str],
        parsing_function: ResponseParser,
        batch_size: int = 10000,
    ) -> None:
        """Clone all data from the source to the destination."""

        cursor = None
        self.destination_client.batch.configure(batch_size=batch_size, dynamic=True)

        progress_bar = tqdm.tqdm(total=(self.source_n // batch_size) + 1)

        while True:
            # Pull a batch of data from the source
            response = get_batch_with_cursor(
                self.source_client,
                self.class_name,
                source_properties,
                batch_size=batch_size,
                cursor=cursor,
            )

            # Stop if there is no more data
            if len(response["data"]["Get"][self.class_name]) == 0:
                break

            # Process data and add it to the destination
            data, vectors, cursor = convert_data(response)
            with self.destination_client.batch as batch:
                for i, x in enumerate(data):
                    batch.add_data_object(x, self.class_name, vector=vectors[i])

            progress_bar.update(1)
        progress_bar.close()

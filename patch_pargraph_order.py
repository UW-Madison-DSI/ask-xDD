import sys
import pickle
import pickle
from askem.preprocessing import HaystackPreprocessor
from askem.ingest_v2 import WeaviateIngester
from askem.retriever.base import get_client
from tqdm import tqdm
import json


WEAVIATE_CLIENT = get_client()

def get_weaviate_paragraph(doc_id: str, hashed_text: str | None = None) -> dict:
    """Get a paragraph from weaviate by paper_id and hashed_text"""

    where_filter = {"operator": "And", "operands": []}
    where_filter["operands"].append({"path":"paper_id", "operator":"Equal", "valueText": doc_id})
    if hashed_text:
        where_filter["operands"].append({"path":"hashed_text", "operator":"Equal", "valueText": hashed_text})
    return WEAVIATE_CLIENT.query.get("Paragraph", ["paper_id", "hashed_text", "paragraph_order"]).with_where(where_filter).with_additional("id").with_limit(10000).do()


    

def main() -> None:

    # Generate list of id to be patched
    with open("tmp/id2topics.pkl", "rb") as input_file:
        id2topics = pickle.load(input_file)
        
    ids_to_patch = []
    target_topics = ["criticalmaas", "geoarchive"]
    for k, v in id2topics.items():
        if any([t in v for t in target_topics]):
            ids_to_patch.append(k)

    print(f"Found {len(ids_to_patch)} documents to patch")

    # Instantiate preprocessor and ingester
    preprocessor = HaystackPreprocessor()
    ingester = WeaviateIngester(
        client=WEAVIATE_CLIENT,
        class_name="Paragraph",
        id2topics=id2topics,
        ingested=set(),
    )

    
    def patch(doc_id: str) -> None:
        ingester.write_batch_to_file([doc_id])
        input_file = ingester.files_to_ingest[0]
        new_paragraphs = preprocessor.run(input_file=input_file, topics=id2topics[doc_id], doc_type="paragraph")

        # Check if all paragraphs are unchanged
        new_hashes = {p["hashed_text"] for p in new_paragraphs}
        assert len(new_hashes) <= 10000
        old_paragraphs = get_weaviate_paragraph(doc_id=doc_id)
        old_records = old_paragraphs["data"]["Get"]["Paragraph"]
        old_hashes = {p["hashed_text"] for p in old_records}
        assert old_hashes == new_hashes, f"Old hashes: {len(old_hashes)}, New hashes: {len(new_hashes)}"

        # Create hash to uuid mapping
        hash2uuid = {p["hashed_text"]: p["_additional"]["id"] for p in old_records}

        # Create skip list (already has order, for resuming)
        skip = {p["hashed_text"] for p in old_records if p["paragraph_order"] is not None}

        # Patch on batch
        for new in new_paragraphs:
            
            # Skip already has order
            if new["hashed_text"] in skip:
                continue

            WEAVIATE_CLIENT.data_object.update(
                uuid=hash2uuid[new["hashed_text"]],
                class_name="Paragraph",
                data_object={
                    "paragraph_order": new["paragraph_order"],
                }
            )

    # Do the patch
    status = {'success': [], 'fail': []}

    for doc_id in tqdm(ids_to_patch):
        ingester.purge_ingest_folder()
        try:
            patch(doc_id)
            status['success'].append(doc_id)
        except Exception as e:
            print(f"Failed to patch {doc_id}: {e}")
            status['fail'].append(doc_id)
            continue

    with open("tmp/patch_status.json", "w") as f:
        json.dump(status, f)
        

if __name__ == "__main__":
    sys.path.append("/hdd/clo36/repo/ask-xDD/askem/retriever")
    main()

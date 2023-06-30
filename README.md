---
title: Ask xdd-COVID
emoji: 📑
sdk: streamlit
sdk_version: 1.19.0
app_file: askem/demo/app.py
pinned: false
---

repo: <https://github.com/AFIDSI/askem>

## For end-users

The end users of our system are ASKEM performers who access it using REST API.

You can visit our [demo](http://cosmos0001.chtc.wisc.edu:8501/) to try how this system can power a traceable COVID-19 search engine.

### Retriever

The retriever is a embedding-based search engine powered by [Dense Passage Retriever](https://arxiv.org/abs/2004.04906). It takes a `question` and returns a list of relevant documents from our [XDD database](https://xdd.wisc.edu/), a document can be a `paragraph` or `figure`. It accepts a **POST** request and requires an **APIKEY**. If you are an ASKEM performer and need your own API key, please do not hesitate to [contact me](mailto:jason.lo@wisc.edu).

- Endpoint: <http://cosmos0001.chtc.wisc.edu:4502>

- Example usage:

    ```python
    import requests

    APIKEY = "insert_api_key_here"
    ENDPOINT = "http://cosmos0001.chtc.wisc.edu:4502"

    headers = {"Content-Type": "application/json", "Api-Key": APIKEY}
    data = {
        "question": "What is the incubation period of COVID-19?",
        "top_k": 3,
        "doc_type": "paragraph",
    }

    response = requests.post(ENDPOINT, headers=headers, json=data)
    response.json()
    ```

- Request body schema:

    ```python
    {
        "question": str,
        "top_k": Optional[int] = 5, # Number of documents to return
        "distance": Optional[float] = 0.5, # Max cosine distance between question and document
        "topic": Optional[str] = None, # Only "covid" is available now
        "doc_type": Optional[str] = None,  # "paragraph" or "figure"
        "preprocessor_id": Optional[str] = None,  # "for future use"
    }
    ```

- Response body schema:

    ```python
    [
        {
            "paper_id": str,
            "doc_type": str,
            "text": str,
            "distance": float,
            "cosmos_object_id": str  # only available for doc_type="figure"
        },
        ...
    ]
    ```

- Retrieve COSMOS figure image from `cosmos_object_id` (Also see COSMOS [documentation](https://uw-cosmos.github.io/Cosmos/) for details)

    ```python
    response = requests.get(f"https://xdd.wisc.edu/askem/object/{cosmos_object_id}")
    jpeg_bytes = response.json()["success"]["data"][0]["properties"]["image"]
    html = f"<img src='data:image/jpg;base64,{jpeg_bytes}' />"
    ```

- [Automatically generated docs](http://cosmos0001.chtc.wisc.edu:4502/docs#/default/get_docs__post)

### Generator (Extractive question-answering BERT)

The generator is an extractive question-answering system that takes a `question` and a `document` and returns an answer using the exact wording from the `document`. It is based on `BertForQuestionAnswering` model from [HuggingFace](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForQuestionAnswering), fine-tuned with [COVID QA deepset](https://huggingface.co/datasets/covid_qa_deepset). An in-house benchmark shows that it can achieve 0.90 BERT-F1 score on the our [testset](https://github.com/AFIDSI/askem/blob/ee8e53d95893083685cd696afe5117ff4064216d/notebooks/make_benchmark_gpt.ipynb), out performing zero-shot Chat-GPT `gpt-3.5-turbo-0301` (BERT-F1 = 0.82, *with minimal hand prompt-tuning*). It accepts a **POST** request and requires no authentication.

- Endpoint: <http://cosmos0001.chtc.wisc.edu:4503>

- Example usage:

    ```python
    import requests

    ENDPOINT = "http://cosmos0001.chtc.wisc.edu:4503"

    headers = {"Content-Type": "application/json"}
    data = {
        "question": "What is the incubation period of COVID-19?",
        "context": "The incubation period of COVID-19 is 14 days.",
    }

    response = requests.post(ENDPOINT, headers=headers, json=data)
    response.json()
    ```

- Request body schema:

    ```python
    {
        "question": str,
        "context": str,
    }
    # The combined length of question and context should not exceed 384 tokens
    ```

- Response body schema:

    ```python
    {
        "answer": str,
        "start": int,  # Start index of the answer in the context
        "end": int,  # End index of the answer in the context
        "score": float,  # Confidence score of the answer. Range in [0, 1], higher is better.
    }
    ```

- [Automatically generated docs](http://cosmos0001.chtc.wisc.edu:4503/docs#/default/get_answer__post)

<details>
    <summary style="font-size: 1.5em;">For developer</summary>

### To deploy the system

1. Make a .env file in the project root directory with these variables

    ```txt
    WEAVIATE_URL=http://weaviate:8080
    WEAVIATE_APIKEY=<generate it yourself using askem.utils.generate_api_key>

    GENERATOR_URL=http://generator:4503

    OPENAI_API_KEY=<key_here>
    OPENAI_ORGANIZATION=<org_id_here>
    ...

    ```

    see shared [dotenv](https://docs.google.com/document/d/1TyGeHxbOShv_jzTIM7vn-equH0XB3wM0mBuAvYI0AR0/edit) file for the actual values

1. Run launch test

    ```sh
    bash ./scripts/launch_test.sh
    ```

1. Ingest figures
    put all text files in a folder, with file format as `<ingest_dir>/<paper-id>.<cosmos_object_id>.txt`
    then run this:

    ```sh
    python askem/deploy.py --input-dir "data/debug_data/figure_test" --topic "covid-19" --doc-type "figure" --weaviate-url "url_to_weaviate"
    ```

</details>

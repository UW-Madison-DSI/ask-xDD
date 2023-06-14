---
title: Ask xdd-COVID
emoji: 📑
sdk: streamlit
sdk_version: 1.19.0
app_file: askem/demo/app.py
pinned: false
---

repo: <https://github.com/JasonLo/askem>

To deploy the system

1. Make a .env file in the project root directory with these variables

    ```txt
    WEAVIATE_URL=http://weaviate:8080
    WEAVIATE_APIKEY=<generate it yourself using askem.utils.generate_api_key>

    GENERATOR_URL=http://generator:4503

    OPENAI_API_KEY=<key_here>
    OPENAI_ORGANIZATION=<org_id_here>

    ```

1. Run launch test

    ```sh
    bash ./scripts/launch_test.sh
    ```

1. Ingest figures
    put all text files in a folder, with file format as `<ingest_dir>/<paper-id>.<cosmos_object_id>.txt`
    then run this:

    ```sh
    python askem/deploy.py --input-dir "data/debug_data/figure_test" --topic "covid-19" --doc-type "figure"
    ```

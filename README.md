# ASKEM

repo: <https://github.com/JasonLo/askem>

To deploy the system

1. Make a .env file in the project root directory with these variables

    ```txt
    WEAVIATE_URL=http://localhost:8080
    WEAVIATE_APIKEY=<generate it yourself using askem.utils.generate_api_key()>
    ```

1. Run launch test

    ```sh
    bash ./scripts/launch_test.sh
    ```

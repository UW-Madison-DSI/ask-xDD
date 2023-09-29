import os
import sys

import pytest
import weaviate
from dotenv import load_dotenv

load_dotenv()
sys.path.append("askem/retriever")


@pytest.fixture
def weaviate_client():
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_APIKEY")),
    )
    return client

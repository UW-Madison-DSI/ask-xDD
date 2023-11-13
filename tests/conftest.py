import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.append("askem/retriever")

from app import app

API_HEADER = {
    "Api-Key": os.getenv("RETRIEVER_APIKEY"),
    "Content-Type": "application/json",
}
API_KEY = os.getenv("RETRIEVER_APIKEY")


@pytest.fixture
def weaviate_client():
    import weaviate

    key = os.getenv("WEAVIATE_APIKEY")
    secret = weaviate.AuthApiKey(api_key=key)
    with weaviate.Client(os.getenv("WEAVIATE_URL"), secret) as client:
        yield client


@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient

    with TestClient(app, headers=API_HEADER) as client:
        yield client


@pytest.fixture
def async_test_client():
    from httpx import AsyncClient

    return AsyncClient(app=app, base_url=os.getenv("RETRIEVER_URL"), headers=API_HEADER)

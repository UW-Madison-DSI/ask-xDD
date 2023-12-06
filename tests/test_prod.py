import requests
import pytest
import httpx


def test_health(prod_api_url):
    response = requests.get(prod_api_url)
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}


def test_vector_search(prod_api_url, prod_api_header):
    query = {
        "topic": "covid",
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
    }
    response = requests.post(
        f"{prod_api_url}/vector", json=query, headers=prod_api_header
    )
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_hybrid_search(prod_api_url, prod_api_header):
    query = {
        "topic": "covid",
        "question": "What is COVID?",
        "top_k": 5,
        "screening_top_k": 1000,
    }
    response = requests.post(
        f"{prod_api_url}/hybrid", json=query, headers=prod_api_header
    )
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_react_sync(prod_api_url, prod_api_header):
    query = {
        "topic": "covid",
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
        "screening_top_k": 1000,
    }
    response = requests.post(
        f"{prod_api_url}/react", json=query, headers=prod_api_header
    )
    assert response.status_code == 200

    results = response.json()
    assert "answer" in results
    assert "used_docs" in results

    assert "answer" is not ""


@pytest.mark.asyncio
async def test_react_async(prod_api_url, prod_api_header):
    query = {
        "topic": "covid",
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
        "screening_top_k": 1000,
    }

    client = httpx.AsyncClient(
        base_url=prod_api_url, headers=prod_api_header, timeout=300
    )

    async with client.stream("POST", "/react_streaming", json=query) as response:
        async for chunk in response.aiter_raw():
            chunk = chunk.decode("utf-8")
            if chunk:
                print(chunk)

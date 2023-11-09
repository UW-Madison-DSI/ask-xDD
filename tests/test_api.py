import pytest


def test_health(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}


def test_vector_search(test_client):
    query = {"question": "What is the incubation period of COVID-19?", "top_k": 5}
    response = test_client.post("/vector", json=query)
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_hybrid_search(test_client):
    query = {
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
        "topic": "covid",
        "screening_top_k": 500,
    }
    response = test_client.post("/hybrid", json=query)
    assert response.status_code == 200
    assert len(response.json()) <= 5


def test_react_search(test_client):
    query = {
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
        "topic": "covid",
        "screening_top_k": 1000,
    }
    response = test_client.post("/react", json=query)
    assert response.status_code == 200

    results = response.json()
    assert "answer" in results
    assert "used_docs" in results

    assert "answer" is not ""


@pytest.mark.anyio
async def test_react_search_streaming(async_test_client):
    query = {
        "question": "What is the incubation period of COVID-19?",
        "top_k": 5,
        "topic": "covid",
        "screening_top_k": 1000,
    }
    response = await async_test_client.post("/react_streaming", json=query)
    assert response.status_code == 200
    results = [line for line in response.iter_lines() if line]
    print(results)

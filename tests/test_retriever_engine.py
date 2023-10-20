from askem.retriever.base import get_documents


def test_get_doc_base(weaviate_client):
    get_documents(
        client=weaviate_client, question="What is the incubation period of COVID-19?"
    )


def test_get_doc_topk(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        top_k=10,
    )


def test_get_doc_distance(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        distance=0.8,
    )


def test_get_doc_topic(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        topic="covid",
    )


def test_get_doc_doctype(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        doc_type="paragraph",
    )


def test_get_doc_preprocessor(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        preprocessor_id="haystack_v0.0.2",
    )


def test_get_doc_term(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        paragraph_terms=["SIR"],
    )


def test_get_doc_move_to(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        move_to="mathematical model",
        move_to_weight=0.5,
    )


def test_get_doc_move_away(weaviate_client):
    get_documents(
        client=weaviate_client,
        question="What is the incubation period of COVID-19?",
        move_away_from="mathematical model",
        move_away_from_weight=0.5,
    )

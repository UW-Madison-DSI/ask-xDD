import logging
import os
from contextlib import asynccontextmanager
from typing import List

from base import get_client, get_documents
from data_models import Document, Query
from fastapi import FastAPI

cached_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cache weaviate client before API startup."""
    cached_resources["weaviate_client"] = get_client()
    yield
    # Release resources when app stops
    cached_resources["weaviate_client"] = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_root():
    """Health check."""
    return {"ping": "pong!"}


# POST endpoint for query
@app.post("/")
async def get_docs(query: Query) -> List[Document]:
    """Search relevant documents."""

    logging.info(f"[get_docs] querying document: {query}")
    logging.info(f"{query.dict()=}")
    return get_documents(client=cached_resources["weaviate_client"], **query.dict())

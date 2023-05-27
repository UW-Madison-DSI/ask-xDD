import os
import logging
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi import FastAPI
from functools import lru_cache
from pydantic import BaseModel

# TODO: Fix GPU, currently not working on OLVI
# ENABLE_CUDA = os.getenv("ENABLE_CUDA") == "1"
# DEVICE = os.getenv("CUDA_CORE") if ENABLE_CUDA else "cpu"
DEVICE = "cpu"
app = FastAPI()


def get_generator(device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained("mbialo/autotrain-test-58072133169")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "mbialo/autotrain-test-58072133169"
    )
    return pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=device
    )


# Cache generator on startup
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logging.info("Starting up...")
    GENERATOR = get_generator()


# IO Models
class Query(BaseModel):
    """Generator query input data model."""

    paragraph: str
    question: str


class Answer(BaseModel):
    """Generator answer output data model."""

    answer: str
    start: int
    end: int
    score: float


@app.get("/")
def get_root():
    """Health check."""
    return {"ping": "pong!"}


# Main endpoint
@app.post("/")
async def get_answer(query: Query) -> Answer:
    """Generate answer for a given question and paragraph."""
    generator = get_generator()
    answer = generator(question=query.question, context=query.paragraph)
    print(answer.keys())
    return Answer(**answer)

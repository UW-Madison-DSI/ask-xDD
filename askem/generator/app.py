import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


# GPU support
# TODO: Fix GPU, currently not working on OLVI
ENABLE_CUDA = os.getenv("ENABLE_CUDA") == "1"
DEVICE = os.getenv("CUDA_CORE") if ENABLE_CUDA else "cpu"


def get_generator(device=DEVICE):
    """Get generator pipeline from Hugging-face hosting."""
    tokenizer = AutoTokenizer.from_pretrained("mbialo/autotrain-test-58072133169")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "mbialo/autotrain-test-58072133169"
    )
    return pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=device
    )


cached_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cache generator before API startup."""
    cached_resources["generator"] = get_generator()
    yield
    # Release resources when app stops
    cached_resources["generator"] = None


app = FastAPI(lifespan=lifespan)


# IO Models
class Query(BaseModel):
    """Generator query input data model."""

    context: str
    question: str


class Answer(BaseModel):
    """Generator answer output data model."""

    answer: str
    start: int
    end: int
    score: float


@app.get("/")
async def get_root():
    """Health check."""
    return {"ping": "pong!"}


# POST endpoint for query
@app.post("/")
async def get_answer(query: Query) -> Answer:
    """Generate answer for a given question and context."""
    answer = cached_resources["generator"](
        question=query.question, context=query.context
    )
    print(answer.keys())
    return Answer(**answer)

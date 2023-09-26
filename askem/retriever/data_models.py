from enum import Enum
from typing import List

from pydantic import BaseModel


class DocType(str, Enum):
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"


class Topic(str, Enum):
    COVID = "covid"


class ClassName(str, Enum):
    PASSAGE = "Passage"


class Query(BaseModel):
    """Retriever query input data model."""

    question: str
    top_k: int = 5
    distance: float = 0.5
    topic: str = None
    doc_type: str = None
    preprocessor_id: str = None
    article_terms: List[str] = None
    paragraph_terms: List[str] = None


class Document(BaseModel):
    """Retriever document output data model."""

    paper_id: str  # xdd document id
    doc_type: str  # document type (paragraph, figure, table)
    text: str  # paragraph text
    distance: float  # distance metric of the document
    cosmos_object_id: str = None
    article_terms: List[str] = None
    paragraph_terms: List[str] = None

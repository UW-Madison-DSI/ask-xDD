from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class DocType(str, Enum):
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"


class Topic(str, Enum):
    COVID = "covid"
    DOLOMITES = "dolomites"


class ClassName(str, Enum):
    PASSAGE = "Passage"


class BaseQuery(BaseModel):
    """Base retriever query input data model."""

    question: str
    top_k: int = 5
    distance: float = 0.5

    # Where per-filters
    topic: Optional[str] = None
    doc_type: Optional[str] = None
    preprocessor_id: Optional[str] = None
    paper_ids: Optional[List[str]] = None

    # Search vectoring
    move_to: Optional[str] = None
    move_to_weight: Optional[float] = None
    move_away_from: Optional[str] = None
    move_away_from_weight: Optional[float] = None


class HybridQuery(BaseQuery):
    screening_top_k: int = 100


class ReactQuery(HybridQuery):
    retriever_endpoint: str = "http://retriever:4502/hybrid"
    model_name: str = "gpt-4"


class Document(BaseModel):
    """Retriever document output data model."""

    paper_id: str  # xdd document id
    doc_type: str  # DocType
    text: str  # paragraph text
    distance: float  # distance metric of the document
    cosmos_object_id: str = None

from enum import Enum

from pydantic import BaseModel, validator


class ClassName(str, Enum):
    PASSAGE = "Passage"  # Deprecating
    PARAGRAPH = "Paragraph"


class DocType(str, Enum):
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    VALUE = "value"


class Topic(str, Enum):
    """Topic enum, must match with xDD dataset values and Weaviate topics."""

    COVID = "xdd-covid-19"
    DOLOMITES = "dolomites"
    CLIMATE_CHANGE = "climate-change-modeling"
    CRITICAL_MAAS = "criticalmaas"
    GEOARCHIVE = "geoarchive"


class BaseQuery(BaseModel):
    """Base retriever query (for vector serach)."""

    question: str
    top_k: int = 5
    distance: float = None

    # Filters
    topic: Topic | None = None
    doc_type: DocType | None = None
    preprocessor_id: str | None = None
    paper_ids: list[str] | None = None

    # Search vectoring
    move_to: str | None = None
    move_to_weight: float | None = None
    move_away_from: str | None = None
    move_away_from_weight: float | None = None


class HybridQuery(BaseQuery):
    topic: Topic  # Override topic to be required
    screening_top_k: int = 100


class ReactQuery(HybridQuery):
    openai_model_name: str = "gpt-4-1106-preview"


class Document(BaseModel):
    """Retriever document output data model.

    Args:
        paper_id: xdd document id
        preprocessor_id: preprocessor id
        topic_list: list of document topic
        doc_type: document type
        text_content: paragraph text
        cosmos_object_id: cosmos object id
        distance: distance to query vector
    """

    paper_id: str
    preprocessor_id: str
    doc_type: str
    topic_list: list[str]
    text_content: str
    hashed_text: str | None = None
    cosmos_object_id: str | None = None
    distance: float | None = None

    @validator("topic_list")
    @classmethod
    def check_and_normalize_topic(cls, v: list[str]):
        normalized = []
        for topic in v:
            if topic in ["covid-19", "covid", "xdd-covid-19"]:
                topic = Topic.COVID
            if isinstance(topic, str):
                topic = Topic(topic)
            normalized.append(topic)
        return normalized

    @validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str):
        v = v.lower()
        assert v.upper() in DocType.__members__, f"{v=} is not a valid doc_type"
        return DocType(v)

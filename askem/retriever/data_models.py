from pydantic import BaseModel


class Query(BaseModel):
    """Retriever query input data model."""

    question: str
    top_k: int = 5
    distance: float = 0.5
    topic: str = None
    doc_type: str = None
    preprocessor_id: str = None


class Document(BaseModel):
    """Retriever document output data model."""

    paper_id: str  # xdd document id
    doc_type: str  # document type (paragraph, figure, table)
    text: str  # paragraph text
    distance: float  # distance metric of the document
    cosmos_object_id: str = None

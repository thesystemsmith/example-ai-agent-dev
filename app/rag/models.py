from dataclasses import dataclass

from app.knowledge.models import SearchResult


@dataclass(frozen=True)
class RagResult:
    question: str
    answer: str
    #derived from retrieval, not invented
    sources: tuple[str, ...]
    retrieved_chunks: tuple[str, ...]
    
    
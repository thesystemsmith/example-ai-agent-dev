from dataclasses import dataclass

@dataclass(frozen=True)
class SourceDocument:
    source: str
    content: str
    
    
@dataclass(frozen=True)
class DocumentChunk:
    # Source and chunk ID preserve traceability after retrieval.
    source: str
    chunk_id: int
    content: str
    
@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
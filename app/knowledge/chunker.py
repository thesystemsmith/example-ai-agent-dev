from app.knowledge.models import DocumentChunk, SourceDocument

def chunk_document(
    document: SourceDocument,
    chunk_size: int=120,
    overlap: int=20,
) -> list[DocumentChunk]:
    
    if chunk_size<= 0:
        raise ValueError("chunk_size must be greater than zero")
    
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be betweenn zero and chunk_size")
    
    words = document.content.split()
    if not words:
        return []
    
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_id = 0
    
    while(start < len(words)):
        end = min(start + chunk_size, len(words))
        chunk_content = " ".join(words[start:end])
        
        chunks.append(
            DocumentChunk(
                source=document.source,
                chunk_id=chunk_id,
                content=chunk_content,
            )
        )
        
        if end == len(words):
            break
        
        # Overlap preserves context across chunk boundaries.
        start = end - overlap
        chunk_id += 1
        
        
    return chunks


def chunk_documents(
    documents: list[SourceDocument],
) -> list[DocumentChunk]:
    
    chunks: list[DocumentChunk] = []
    
    for document in documents:
        chunks.extend(chunk_document(document))
        
    return chunks
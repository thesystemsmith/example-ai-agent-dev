from app.knowledge.chunker import chunk_documents
from app.knowledge.embedder import embed_texts
from app.knowledge.loader import load_documents
from app.knowledge.models import DocumentChunk, SearchResult
from app.knowledge.vector_store import FaissVectorStore


class KnowledgeRetriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError('no document chunks were available for indexing')
        
        chunk_embeddings = embed_texts(
            [chunk.content for chunk in chunks]
        )
        
        self._store = FaissVectorStore(
            chunks=chunks,
            embeddings=chunk_embeddings
        )
        
    
    @classmethod
    def from_directory(
        cls,# class itself is called
        directory:str,
    ) -> 'KnowledgeRetriever':
        documents = load_documents(directory)
        chunks = chunk_documents(documents)
        
        return cls(chunks)
    
    @property
    def chunk_count(self) -> int:
        return self._store.size
    
    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        clean_question = question.strip()
        
        if not clean_question:
            raise ValueError('question cannot be empty')
        
        #The question must use the same embedding model as the documents.
        question_embedding = embed_texts([clean_question])[0]
        
        return self._store.search(
            query_embedding=question_embedding,
            top_k=top_k,
        )
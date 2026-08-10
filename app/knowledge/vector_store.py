import faiss
import numpy as np 

from app.knowledge.models import DocumentChunk, SearchResult


class FaissVectorStore:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        
        if not chunks:
            raise ValueError('cannot build an index without chunks')
        
        if embeddings.ndim != 2:
            raise ValueError('embeddings must be a 2d matrix')
        
        if embeddings.shape[0] != len(chunks):
            raise ValueError('every chunk must have exactly one embedding')
        
        self._chunks = list(chunks)
        self._dimesion = int(embeddings.shape[1])
        
        vectors = np.ascontiguousarray(
            embeddings,
            dtype=np.float32
        )
        
        #Normalized vectors let inner product represent cosine similarity.
        faiss.normalize_L2(vectors)
        
        self._index = faiss.IndexFlatIP(self._dimesion)
        self._index.add(vectors)
        

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError('top_k must be > 0')
        
        query_vector = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1,-1) # into 2d matrix for fast processing
        
        if query_vector.shape[1] != self._dimesion:
            raise ValueError('query and document embedding dont match')
        
        faiss.normalize_L2(query_vector)
        
        result_count = min(top_k, self.size)
        scores, positions = self._index.search(
            query_vector,
            result_count,
        )
        
        results: list[SearchResult] = []
        
        for score, position in zip(scores[0], positions[0]):
            results.append(
                SearchResult(
                    chunk=self._chunks[int(position)],
                    score=float(score)
                )
            )
            
        return results
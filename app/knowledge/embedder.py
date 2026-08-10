import numpy as np
from ollama import Client, ResponseError

from app.config import settings


client = Client(host =settings.ollama_host)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("at least one text is required for embeddings")
    
    try:
        response = client.embed(
            model = settings.embedding_model,
            input = texts,
        )
    except ResponseError as error:
        raise RuntimeError(
            f"ollama embedding failed: {error.error}"
        )
    
    #faiss expects vectors represented as 32 bit floating values
    embeddings = np.asarray(
        response.embeddings,
        dtype=np.float32
    )
    
    if embeddings.ndim !=2:
        raise RuntimeError("ollama returned an invalid embedding matrix")
    
    if embeddings.shape[0] != len(texts):
        raise RuntimeError('embeddings count does not match the input count')
    
    return embeddings
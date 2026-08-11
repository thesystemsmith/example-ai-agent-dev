from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.config import settings
from app.knowledge.models import SearchResult
from app.knowledge.retriever import KnowledgeRetriever
from app.rag.models import RagResult
from app.rag.prompts import RAG_PROMPT


def format_context(results: list[SearchResult]) -> str:
    formatted_chunks: list[str] = []
    
    for result in results:
        reference = (
            f'{result.chunk.source}'
            f'{result.chunk.chunk_id}'
        )
        
        formatted_chunks.append(
            f'[{reference}]\n{result.chunk.content}'
        )
        
    return '\n\n'.join(formatted_chunks)


class RagService:
    def __init__(
        self,
        retriever: KnowledgeRetriever,
    )-> None:
        self._retriever = retriever
        
        model = ChatOllama(
            model=settings.model,
            base_url=settings.ollama_host,
            temperature=settings.temperature,
            num_ctx=settings.context_size,
        )
        
        #parsing keeps the service independent of AIMessage objects.
        self._answer_chain = (
            RAG_PROMPT | model | StrOutputParser()
        )
        
    
    @classmethod
    def from_directory(
        cls,
        directory: str,
    ) -> 'RagService':
        
        retriever = KnowledgeRetriever.from_directory(directory)
        return cls(retriever)
    
    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> RagResult:
        
        clean_question = question.strip()
        if not clean_question:
            raise ValueError('questin cannot be empty')
        
        retrieved_chunks = self._retriever.search(
            clean_question,
            top_k=top_k
        )
        
        context = format_context(retrieved_chunks)
        
        answer = self._answer_chain.invoke(
            {
                'question':clean_question,
                'context':context 
            }
        )
        
        #dict.fromkeys removes duplicates while preserving order.
        sources = tuple(
            dict.fromkeys( result.chunk.source for result in retrieved_chunks )
        )
        
        return RagResult(
            question=clean_question,
            answer=answer,
            sources=sources,
            retrieved_chunks=tuple(retrieved_chunks),
        )
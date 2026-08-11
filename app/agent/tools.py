from langchain.tools import tool
from langchain_core.tools import BaseTool

from app.knowledge.retriever import KnowledgeRetriever


def create_search_notes_tool(
    retriever: KnowledgeRetriever,
) -> BaseTool:
    
    @tool
    def search_notes(query: str) -> str:
        '''
        search the learners local study notes
        
        Use this tool when answering questions about the study material
        or when creating a quiz from the notes.

        Args:
            query: The question or topic to search for.   
        '''
        
        results = retriever.search(
            query,
            top_k=3
        )
        
        formatted_chunks: list[str] = []
        
        for result in results:
            reference = (
                f'{result.chunk.source}'
                f'{result.chunk.chunk_id}'
            )
            
            formatted_chunks.append(
                f'[{reference}]\n'
                f'{result.chunk.content}'
            )
            
        return "\n\n".join(formatted_chunks)
    
    return search_notes #closure
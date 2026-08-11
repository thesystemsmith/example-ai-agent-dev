from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from app.agent.models import AgentResult
from app.agent.tools import create_search_notes_tool
from app.config import settings
from app.knowledge.retriever import KnowledgeRetriever
from app.rag.prompts import RAG_SYSTEM_PROMPT


AGENT_SYSTEM_PROMPT = f'''
{RAG_SYSTEM_PROMPT.strip()}

You have access to a search_notes tool.

Additional rules:
- Decide whether the user's request requires the study notes.
- Use search_notes for questions or quizzes about the notes.
- Do not use search_notes for greetings or simple conversation.
- After calling the tool, treat its output as retrieved context.
- Do not call the same tool repeatedly for one request.
'''


class LearningAgent:
    def __init__(
        self,
        retriever: KnowledgeRetriever
    ) -> None:
        
        search_notes = create_search_notes_tool(retriever)
        
        model = ChatOllama(
            model=settings.model,
            base_url=settings.ollama_host,
            temperature=settings.temperature,
            num_ctx=settings.context_size
        )
        
        #create_agent controls the model/tool execution loop.
        self._agent = create_agent(
            model=model,
            tools=[search_notes],
            system_prompt=AGENT_SYSTEM_PROMPT.strip(),

        )
        
        
    @classmethod
    def from_directory(
        cls,
        directory: str
    ) -> 'LearningAgent':
        
        retriever = KnowledgeRetriever.from_directory(directory)
        
        return cls(retriever)
    
    
    def ask(self, question: str) -> AgentResult:
        clean_question = question.strip()
        
        if not clean_question:
            raise ValueError('question cannot be empty')
        
        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": clean_question,
                    }
                ]
            },
            #Prevent accidental endless tool calls.
            config={'recursion_limit': 5}
        )
        
        tools_used = tuple(
            dict.fromkeys(
                tool_call["name"]
                for message in result["messages"]
                for tool_call in getattr(message,"tool_calls",[],)
            )
        )
        
        final_message = result['messages'][-1]
        
        return AgentResult(
            answer=str(final_message.content),
            tools_used=tools_used
        )
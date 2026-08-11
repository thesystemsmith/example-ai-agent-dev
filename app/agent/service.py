from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.models import AgentResult
from app.agent.tools import create_search_notes_tool
from app.config import settings
from app.knowledge.retriever import KnowledgeRetriever
from app.rag.prompts import RAG_SYSTEM_PROMPT


AGENT_SYSTEM_PROMPT = f'''
{RAG_SYSTEM_PROMPT.strip()}

You have access to a search_notes tool.

The rule about using only retrieved context applies after calling
search_notes. You may respond normally to greetings.

Modes:
- Learn: search the notes and explain the requested topic simply.
- Quiz: search the notes and ask exactly one short-answer question.
- If the user answers a quiz, give brief feedback using conversation
  history and retrieved evidence.

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
        
        #checkpoint
        self._memory = InMemorySaver()
        
        #create_agent controls the model/tool execution loop.
        self._agent = create_agent(
            model=model,
            tools=[search_notes],
            system_prompt=AGENT_SYSTEM_PROMPT.strip(),
            checkpointer=self._memory
        )
        
        
    @classmethod
    def from_directory(
        cls,
        directory: str
    ) -> 'LearningAgent':
        
        retriever = KnowledgeRetriever.from_directory(directory)
        
        return cls(retriever)
    
    
    def ask(
        self,
        question: str,
        thread_id: str,
    ) -> AgentResult:
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
            config={
                #reusing this id continues the same conversation
                'configurable': {
                    'thread_id': thread_id
                },
                'recursion_limit': 5
            }
        )
        
        #memory
        current_turn_messages = []
        for message in reversed(result['messages']):
            current_turn_messages.append(message)
            
            if isinstance(message, HumanMessage):
                break
            
        current_turn_messages.reverse()
        
        tools_used = tuple(
            dict.fromkeys(
                tool_call["name"]
                for message in current_turn_messages
                for tool_call in getattr(message,"tool_calls",[],)
            )
        )
        
        final_message = result['messages'][-1]
        
        return AgentResult(
            answer=str(final_message.content),
            tools_used=tools_used
        )
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.models import AgentResult
from app.agent.tools import create_search_notes_tool
from app.config import settings
from app.knowledge.retriever import KnowledgeRetriever


AGENT_SYSTEM_PROMPT = '''
You are a concise local AI learning coach with access to search_notes.

Tool rules:
- For study questions or quizzes, call search_notes immediately.
- Use the native tool interface; never describe or simulate a tool call.
- Do not explain your plan before calling a tool.
- After receiving tool results, answer only from those results.
- Cite retrieved information using the exact source labels provided.
- Do not call search_notes repeatedly for the same request.
- For greetings and simple conversation, answer directly without tools.

Learning rules:
- For a learn request, explain the topic simply with a practical example.
- For a quiz request, ask exactly one short-answer question from the notes.
- When the user answers a quiz, give brief feedback using conversation history.
- If the notes do not contain the answer, say so clearly.
- Never expose internal reasoning or planning.
- Keep user-facing answers under 150 words.
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
            num_ctx=settings.context_size,
            num_predict=384,
            keep_alive='30m'
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

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResult:
    answer: str
    
    #shows which actions the model selected
    tools_used: tuple[str, ...]
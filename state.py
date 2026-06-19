from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    sub_questions: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    needs_reretrieval: bool
    iteration_count: int
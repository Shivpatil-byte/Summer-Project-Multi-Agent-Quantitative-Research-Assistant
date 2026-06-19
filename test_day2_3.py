from orchestrator import orchestrator_node
from retriever import retriever_node

if __name__ == "__main__":
    # 1. Initialize the shared memory clipboard (AgentState format)
    shared_state = {
        "query": "How did Purina Petcare and Nespresso perform globally in 2024?",
        "sub_questions": [],
        "retrieved_chunks": []
    }
    
    print("=== STARTING DAY 2-3 PIPELINE TEST ===")
    
    # 2. Pass state to Orchestrator Node
    orchestrator_updates = orchestrator_node(shared_state)
    shared_state.update(orchestrator_updates) # Simulates LangGraph state update
    
    # 3. Pass updated state to Retriever Node
    retriever_updates = retriever_node(shared_state)
    shared_state.update(retriever_updates)
    
    print("\n=== FINAL VERIFICATION ===")
    print(f"Total sub-questions in state: {len(shared_state['sub_questions'])}")
    print(f"Total unique chunks collected: {len(shared_state['retrieved_chunks'])}")
    
    if shared_state["retrieved_chunks"]:
        print("\nSample Chunk Meta from Top Hit:")
        print(shared_state["retrieved_chunks"][0]["metadata"])
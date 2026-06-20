from langgraph.graph import StateGraph, START, END
from state import AgentState
from orchestrator import orchestrator_node
from retriever import retriever_node
from analyst import analyst_node
from critic import critic_node

# 1. Initialize the Graph Blueprint
# We strictly define that this graph will use your AgentState dictionary structure
workflow = StateGraph(AgentState)

# 2. Register the Nodes (The "Workers")
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)

# 3. Define the Standard Edges (The "Conveyor Belt")
# This dictates the strict, unchangeable order of operations
workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", "retriever")
workflow.add_edge("retriever", "analyst")
workflow.add_edge("analyst", "critic")

# 4. Define the Routing Logic (The "Switchboard")
def routing_function(state: AgentState):
    """
    This function looks at the clipboard AFTER the Critic is done.
    It decides where the graph should go next based on the state variables.
    """
    needs_reretrieval = state.get("needs_reretrieval", False)
    iterations = state.get("iteration_count", 0)
    
    if needs_reretrieval and iterations < 2:
        print(f"\n[ROUTER] Critic rejected answer (Iteration {iterations}). Looping back to Retriever...")
        return "retriever"
    elif needs_reretrieval and iterations >= 2:
        print("\n[ROUTER] Max iterations reached! Forcing END to prevent infinite loop.")
        return END
    else:
        print("\n[ROUTER] Critic approved the answer! Moving to END.")
        return END

# 5. Attach the Conditional Edge
# We attach this specific routing function to the output of the "critic" node
workflow.add_conditional_edges("critic", routing_function)

# 6. Compile the Graph into a runnable application
app = workflow.compile()

# ==========================================
# END-TO-END TESTING SUITE
# ==========================================
if __name__ == "__main__":
    # We will test 3 distinct queries to see how the graph handles different scenarios
    test_queries = [
        "What was the organic sales growth of Purina PetCare in 2024?",
        "Did Nespresso release any new product lines in 2024, and what was their revenue?",
        "What are the company's projected carbon emission goals for 2030?"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST QUERY {i}: '{q}'")
        print(f"{'='*80}")

        # Initialize the blank starting state
        initial_state = {
            "query": q,
            "sub_questions": [],
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
            "confidence": 1.0,
            "needs_reretrieval": False,
            "iteration_count": 0
        }

        # app.invoke() kicks off the LangGraph process. It will block the terminal
        # and automatically run through all the nodes until it hits END.
        final_state = app.invoke(initial_state)

        print("\n" + "*"*50)
        print("FINAL OUTPUT DELIVERED")
        print("*"*50)
        print(f"Answer:\n{final_state.get('answer', 'N/A')}\n")
        
        print(f"Citations: {len(final_state.get('citations', []))}")
        for idx, c in enumerate(final_state.get('citations', []), 1):
            print(f"  {idx}. Page {c.get('page_number')} ({c.get('year')}): \"{c.get('quote')}\"")
            
        print(f"\nFinal Critic Confidence: {final_state.get('confidence')}")
        print(f"Total Loops Executed: {final_state.get('iteration_count')}")
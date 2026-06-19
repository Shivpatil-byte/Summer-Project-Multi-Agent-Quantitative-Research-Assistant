from hybrid import AdvancedRetriever

# Initialize the Week 2 hybrid engine globally so it loads into memory only once
print("\n[Retriever] Warming up embedding models and database indices...")
retriever_engine = AdvancedRetriever()
print("[Retriever] Core search engine ready.")

def retriever_node(state):
    sub_questions = state.get("sub_questions", [])
    
    # Fallback just in case the orchestrator node was skipped
    if not sub_questions:
        print("[Retriever] No sub-questions found. Falling back to original query.")
        sub_questions = [state["query"]]
        
    print(f"\n[Retriever] Executing hybrid searches for {len(sub_questions)} targets...")
    
    unique_chunks = {}

    # Run the retrieval loop
    for idx, sq in enumerate(sub_questions, 1):
        print(f"  -> Searching Target [{idx}]: '{sq}'")
        
        # Pull top 3 highly-refined chunks per sub-question using your Week 2 logic
        results = retriever_engine.advanced_search(
            query=sq, 
            top_k_initial=10, 
            final_top_k=3
        )
        
        # Deduplicate on the fly using raw text as the unique dictionary key
        for res in results:
            text = res["text"]
            if text not in unique_chunks:
                unique_chunks[text] = res

    # Convert the deduplicated dictionary back into a clean list for the state
    final_chunks = list(unique_chunks.values())
    print(f"[Retriever] Complete. Gathered and deduplicated down to {len(final_chunks)} total chunks.")
    
    return {"retrieved_chunks": final_chunks}
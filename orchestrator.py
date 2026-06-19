import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq 

# This line automatically finds your .env file and loads the keys
load_dotenv()

def orchestrator_node(state):
    query = state["query"]
    
    print(f"\n[Orchestrator] Decomposing query: '{query}'")
    
    # Initialize the Groq LLM (Make sure GROQ_API_KEY is in your .env file)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quantitative research router. Break the user's complex financial query into 2 to 3 distinct, highly specific search questions. Output ONLY a valid JSON list of strings, with no markdown formatting or extra text."),
        ("user", "{query}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query))
    
    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        sub_questions = json.loads(clean_text)
        print(f"[Orchestrator] Successfully generated {len(sub_questions)} sub-questions.")
    except json.JSONDecodeError:
        print("[Orchestrator] Warning: LLM failed to format JSON. Falling back to original query.")
        sub_questions = [query]
        
    return {"sub_questions": sub_questions}

if __name__ == "__main__":
    # A dummy state to test the router
    test_state = {"query": "How did Purina Petcare and Nespresso perform globally in 2024?"}
    
    # Run the node
    result = orchestrator_node(test_state)
    
    print("\n--- LLM OUTPUT ---")
    for sq in result["sub_questions"]:
        print(f"- {sq}")
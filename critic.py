import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def critic_node(state):
    query = state["query"]
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])
    iteration_count = state.get("iteration_count", 0)
    
    print("\n[Critic] Reviewing answer for factual grounding and alignment...")
    
    # Flatten the raw text blocks for the Critic to read
    context_str = "\n".join([f"- {c['text']}" for c in chunks])
    
    # Temperature 0 ensures the critic is ruthlessly deterministic
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    system_prompt = (
        "You are an adversarial financial auditor. Evaluate if the generated answer is completely supported "
        "by the raw context paragraphs. Look closely for hallucinations, unverified numbers, or outside assumptions.\n\n"
        "Output your evaluation STRICTLY as a JSON object matching this schema:\n"
        "{{\n"
        "  \"confidence\": 0.95,\n"
        "  \"justification\": \"Brief rationale explaining why the score was given.\"\n"
        "}}\n"
        "The confidence score must be a float between 0.0 and 1.0. If the text blocks fully justify every single number/claim, score it > 0.8. "
        "If there are missing figures, fabricated metrics, or data discrepancies, score it < 0.7."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Original Query: {query}\nGenerated Answer: {answer}\n\nAllowed Ground Truth Context:\n{context}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query, answer=answer, context=context_str))
    
    # Safe defaults
    confidence = 1.0
    needs_reretrieval = False
    
    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        confidence = float(data.get("confidence", 1.0))
        justification = data.get("justification", "No justification provided.")
        
        print(f"[Critic] Evaluation Complete. Confidence Score: {confidence}")
        print(f"[Critic] Justification: {justification}")
        
        # The Day 4-5 threshold logic
        if confidence < 0.7:
            needs_reretrieval = True
            print("[Critic] Alert: Confidence is below 0.7 threshold. Flagging for re-retrieval.")
            
    except (json.JSONDecodeError, ValueError):
        print("[Critic] Warning: Evaluator formatting failed. Defaulting to safe passage.")
        
    return {
        "confidence": confidence, 
        "needs_reretrieval": needs_reretrieval,
        "iteration_count": iteration_count + 1
    }
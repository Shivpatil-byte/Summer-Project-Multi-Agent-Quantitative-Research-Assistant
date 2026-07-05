import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def analyst_node(state):
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    
    print("\n[Analyst] Synthesizing final answer from retrieved context...")
    
    # Format the collected chunks into a readable string for the LLM
    context_str = ""
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        context_str += f"--- Context Block {idx} ---\n"
        context_str += f"Source: {meta.get('source')} | Page: {meta.get('page_number')} | Year: {meta.get('year')}\n"
        context_str += f"Text: {chunk['text']}\n\n"
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = (
        "You are an expert quantitative financial analyst. Synthesize a comprehensive answer to the user's query "
        "using ONLY the provided text blocks. Along with your answer, you must extract exact citations to justify your claims.\n\n"
        "Output your response STRICTLY as a JSON object matching this schema:\n"
        "{{\n"
        "  \"answer\": \"Your detailed analytical sentence or paragraph summary here.\",\n"
        "  \"citations\": [\n"
        "    {{\n"
        "      \"year\": 2024,\n"
        "      \"page_number\": 12,\n"
        "      \"quote\": \"Exact matching quote from the text, STRICTLY under 15 words.\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Ensure no markdown wrappers like ```json or trailing text are returned."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User Query: {query}\n\nRetrieved Context:\n{context}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=query, context=context_str))
    
    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        answer = data.get("answer", "No answer synthesized.")
        citations = data.get("citations", [])
        print(f"[Analyst] Answer generated successfully with {len(citations)} strict citations.")
    except json.JSONDecodeError:
        print("[Analyst] Warning: Failed to parse Analyst JSON structure. Falling back to plain text formatting.")
        answer = response.content
        citations = []
        
    return {"answer": answer, "citations": citations}
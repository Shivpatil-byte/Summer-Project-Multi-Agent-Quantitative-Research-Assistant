import json
import os
import time
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import your actual class from retriever.py
from retriever import AdvancedRetriever

load_dotenv()

CLEANED_FILE = "cleaned_evaluation_results.json"
DENSE_OUT = "dense_ablation_results.json"
BM25_OUT = "bm25_ablation_results.json"

def load_checkpoint(file_path):
    """Loads existing progress if a crash happened, otherwise returns empty list"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except:
            print(f"⚠️ Checkpoint file {file_path} was corrupted. Starting fresh for this strategy.")
            return []
    return []

def run_ablation_strategy(strategy_type):
    out_file = DENSE_OUT if strategy_type == "dense" else BM25_OUT
    print(f"\n=========================================")
    print(f" STARTING ABLATION RUN: {strategy_type.upper()}")
    print(f"=========================================")
    
    if not os.path.exists(CLEANED_FILE):
        print(f"❌ Error: Base file '{CLEANED_FILE}' not found!")
        sys.exit(1)
        
    with open(CLEANED_FILE, "r", encoding="utf-8") as f:
        base_queries = json.load(f)
        
    existing_results = load_checkpoint(out_file)
    completed_queries = {item["user_input"] for item in existing_results}
    
    if len(completed_queries) > 0:
        print(f"ℹ️ Found existing checkpoint. {len(completed_queries)}/{len(base_queries)} queries already completed.")
    
    retriever = AdvancedRetriever()
    llm = ChatGroq(model_name="llama-3.1-8b-instant", max_tokens=1024)
    
    for idx, item in enumerate(base_queries):
        user_query = item["user_input"]
        
        if user_query in completed_queries:
            print(f"[{idx+1}/{len(base_queries)}] ⏩ Skipping (Already processed in checkpoint)")
            continue
            
        print(f"[{idx+1}/{len(base_queries)}] Processing: {user_query[:50]}...")
        
        try:
            # 1. CALLING RETRIEVAL METHODS
            if strategy_type == "dense":
                raw_results = retriever._get_chroma_top_n(user_query, top_n=5)
            else:
                raw_results = retriever._get_bm25_top_n(user_query, top_n=5)
            
            contexts = [res["text"] for res in raw_results]
            
            if not contexts:
                contexts = ["No relevant context found for this strategy."]
            
            # Construct preliminary context text
            context_text = "\n\n".join(contexts)
            
            # 2. DEFENSIVE SHIELD: Check character length to guarantee staying under 6000 tokens
            # 15,000 characters roughly equals ~3,800-4,000 tokens max.
            MAX_CHAR_LIMIT = 15000 
            if len(context_text) > MAX_CHAR_LIMIT:
                print(f"  ⚠️ Warning: Context length ({len(context_text)} chars) is too heavy for your Groq tier. Trimming context...")
                # Allocate chunk allocation size budget evenly
                allowed_chunk_len = MAX_CHAR_LIMIT // len(contexts)
                contexts = [c[:allowed_chunk_len] + "... [Trimming to fit API token limits]" for c in contexts]
                context_text = "\n\n".join(contexts)
            
            # Construct the final safe system prompt
            system_prompt = f"Answer the question based ONLY on this context:\n{context_text}"
            
            # 3. GENERATION
            response_msg = llm.invoke(f"{system_prompt}\n\nQuestion: {user_query}")
            response = response_msg.content
            
            # 4. APPEND RECORD
            existing_results.append({
                "user_input": user_query,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": item["reference"] 
            })
            
            # 5. LIVE CHECKPOINT SAVE
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=4)
                
            print(f"  ✅ Saved Query {idx+1} to {out_file}")
            
            print("  ⏳ Flushing API limits for 60 seconds...")
            time.sleep(60)
            
        except Exception as e:
            print(f"  ❌ Error processing query {idx+1}: {e}")
            print("  🛑 Terminating entire script run execution to protect dataset sequence integrity.")
            sys.exit(1)

    print(f"\n🎉 SUCCESS: Finished all {len(base_queries)} queries for {strategy_type.upper()}!")

if __name__ == "__main__":
    # DO NOT delete your dense file! The checkpoint engine will pick up exactly at Query 3.
    run_ablation_strategy("dense")
import json
import time
import math
import os
from datasets import Dataset
from ragas import evaluate

# Stable legacy paths
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# We only need to finish BM25 now!
FILES_TO_SCORE = [
    # "dense_ablation_results.json", 
    "bm25_ablation_results.json"
]

def is_corrupted(val):
    return val is None or (isinstance(val, float) and math.isnan(val))

def score_files():
    # SWITCHED TO GEMMA 2: Fresh 500k Token Bucket!
    evaluator_llm = ChatGroq(model_name="llama-3.3-70b-versatile", max_tokens=1024)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    metrics_to_run = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]

    for file_path in FILES_TO_SCORE:
        print(f"\n=========================================")
        print(f" STARTING RAGAS SCORING FOR: {file_path}")
        print(f"=========================================")

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}. Skipping...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for index, item in enumerate(data):
            # Checkpoint Logic: Skip if it already has a healthy faithfulness score
            if "faithfulness" in item and not is_corrupted(item.get("faithfulness")):
                print(f"[{index+1}/{len(data)}] ⏩ Skipping (Already perfectly scored)")
                continue
                
            print(f"\n[{index+1}/{len(data)}] Scoring Query: {item.get('user_input')[:50]}...")
            
            # SMART RETRY LOOP
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    single_row = {
                        "question": [item.get("user_input")],
                        "answer": [item.get("response")],
                        "contexts": [item.get("retrieved_contexts")],
                        "ground_truth": [item.get("reference")]
                    }
                    
                    dataset = Dataset.from_dict(single_row)
                    
                    # Run RAGAS
                    result = evaluate(
                        dataset=dataset, 
                        metrics=metrics_to_run,
                        llm=evaluator_llm, 
                        embeddings=evaluator_embeddings
                    )
                    
                    scores = result.to_pandas().to_dict('records')[0]
                    
                    # Verify RAGAS actually returned numbers, not NaNs
                    if is_corrupted(scores.get("faithfulness")):
                        raise ValueError("RAGAS returned NaN. Likely a hidden API timeout.")
                    
                    item["faithfulness"] = scores.get("faithfulness")
                    item["answer_relevancy"] = scores.get("answer_relevancy")
                    item["context_recall"] = scores.get("context_recall")
                    item["context_precision"] = scores.get("context_precision")
                    
                    # LIVE SAVE
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                        
                    print(f"  ✅ Saved Scores for Index {index+1}")
                    
                    print("  ⏳ Pacing API... waiting 15 seconds.")
                    time.sleep(15)
                    break 
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check for Daily Token Limit
                    if "tpd" in error_msg or "tokens per day" in error_msg:
                        print(f"  🛑 DAILY LIMIT HIT (Attempt {attempt+1}/{max_retries}). Groq needs to refresh your bucket.")
                        print("  💤 Deep sleep for 15 minutes...")
                        time.sleep(900) # Sleep for 15 minutes
                        
                    # Check for standard Rate Limits or Timeouts
                    elif "rate limit" in error_msg or "429" in error_msg or "timeout" in error_msg or "nan" in error_msg:
                        print(f"  ⚠️ Timeout/Rate Limit (Attempt {attempt+1}/{max_retries}). Cooling down for 90 seconds...")
                        time.sleep(90)
                        
                    else:
                        print(f"  ❌ Unknown RAGAS Error at Index {index+1}: {e}")
                        print("  🛑 Skipping to next row to prevent total crash.")
                        break 

        print(f"\n🎉 Finished scoring {file_path}!")

if __name__ == "__main__":
    score_files()
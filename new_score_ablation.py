import json
import time
import math
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 1. KILL RAGAS TELEMETRY (Prevents the socket.gaierror network crash)
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

load_dotenv()

FILES_TO_SCORE = ["bm25_ablation_results.json"]

def is_corrupted(val):
    return val is None or (isinstance(val, float) and math.isnan(val))

# =====================================================================
# THE BULLETPROOF WRAPPER
# =====================================================================
class SafeGroq(ChatGroq):
    def generate(self, *args, **kwargs):
        if "n" in kwargs:
            kwargs.pop("n")
        return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        if "n" in kwargs:
            kwargs.pop("n")
        return await super().agenerate(*args, **kwargs)
# =====================================================================

def score_files():
    # 2. SWITCH TO LLAMA 4 SCOUT: 100% Active, Untouched 500k Token Bucket
    evaluator_llm = SafeGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=2048)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    metrics_list = [
        Faithfulness(), 
        AnswerRelevancy(strictness=1), 
        ContextRecall(), 
        ContextPrecision()
    ]

    for file_path in FILES_TO_SCORE:
        print(f"\n=========================================")
        print(f" STARTING NAN-REPAIR SCORING FOR: {file_path}")
        print(f"=========================================")

        if not os.path.exists(file_path):
            print(f"⚠️ File not found. Skipping...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for index, item in enumerate(data):
            # We know the first 11 are solid now based on your last run
            if index < 11:
                continue 

            # Smart NaN Hunter
            metrics_to_run = []
            for m in metrics_list:
                metric_name = m.name
                if metric_name not in item or is_corrupted(item.get(metric_name)):
                    metrics_to_run.append(m)

            if not metrics_to_run:
                print(f"[{index+1}/{len(data)}] ⏩ Skipping (All metrics perfectly scored)")
                continue
                
            print(f"\n[{index+1}/{len(data)}] Repairing {len(metrics_to_run)} NaN metric(s) for query...")
            
            try:
                single_row = {
                    "question": [item.get("user_input")],
                    "answer": [item.get("response")],
                    "contexts": [item.get("retrieved_contexts")],
                    "ground_truth": [item.get("reference")]
                }
                
                for metric in metrics_to_run:
                    print(f"  -> Evaluating: {metric.name}")
                    dataset = Dataset.from_dict(single_row)
                    result = evaluate(
                        dataset=dataset, 
                        metrics=[metric], 
                        llm=evaluator_llm, 
                        embeddings=evaluator_embeddings
                    )
                    scores = result.to_pandas().to_dict('records')[0]
                    
                    metric_name = metric.name
                    item[metric_name] = scores.get(metric_name)

                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=4)
                        
                    time.sleep(5) 
                    
                print(f"  ✅ Repaired Index {index+1}")
                time.sleep(3) 
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "tpd" in error_msg or "tokens per day" in error_msg:
                    print(f"  🛑 Rate Limit Hit. Sleeping for 10 minutes to recover bucket...")
                    time.sleep(600)
                else:
                    print(f"  ❌ Error at {index+1}: {e}.")
                    print("  💤 Cooling down for 30 seconds...")
                    time.sleep(30)
                continue

        print(f"\n🎉 Finished repairing {file_path}!")

if __name__ == "__main__":
    score_files()
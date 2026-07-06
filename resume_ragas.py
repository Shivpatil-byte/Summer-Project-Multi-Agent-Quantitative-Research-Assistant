import json
import time
import math
from datasets import Dataset
from ragas import evaluate

# 1. FIXED: Imported the Capitalized Classes from the stable legacy path
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = "final_evaluation_results.json"

def is_corrupted(val):
    """Helper function to check if a value is None or NaN"""
    return val is None or (isinstance(val, float) and math.isnan(val))

def safely_resume_ragas():
    evaluator_llm = ChatGroq(model_name="llama-3.1-8b-instant", max_tokens=4096)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # 2. FIXED: Instantiating the Capitalized Classes
    metrics_to_run = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]

    try:
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Critical File Error: {e}")
        return

    print(f"Loaded {len(data)} queries. Scanning ALL metrics for missing/NaN values...")
    processed_count = 0

    for index, item in enumerate(data):
        
        f_score = item.get("faithfulness")
        ar_score = item.get("answer_relevancy")
        cr_score = item.get("context_recall")
        cp_score = item.get("context_precision")
        
        if is_corrupted(f_score) or is_corrupted(ar_score) or is_corrupted(cr_score) or is_corrupted(cp_score):
            print(f"\n[{index+1}/{len(data)}] Repairing Query: {item.get('user_input')[:60]}...")
            
            try:
                single_row = {
                    "question": [item.get("user_input")],
                    "answer": [item.get("response")],
                    "contexts": [item.get("retrieved_contexts")],
                    "ground_truth": [item.get("reference")]
                }
                
                dataset = Dataset.from_dict(single_row)
                
                result = evaluate(
                    dataset=dataset, 
                    metrics=metrics_to_run,
                    llm=evaluator_llm, 
                    embeddings=evaluator_embeddings
                )
                
                scores = result.to_pandas().to_dict('records')[0]
                
                item["faithfulness"] = scores.get("faithfulness")
                item["answer_relevancy"] = scores.get("answer_relevancy")
                item["context_recall"] = scores.get("context_recall")
                item["context_precision"] = scores.get("context_precision")
                
                with open(FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                    
                print(f"  ✅ Repaired Index {index}. Faithfulness: {item['faithfulness']}")
                processed_count += 1
                
                print("  ⏳ Sleeping for 60 seconds to flush token limits...")
                time.sleep(60)
                
            except Exception as e:
                print(f"  ❌ RAGAS API Error at Index {index}: {e}")
                print("  🛑 Moving to next row.")
        else:
            print(f"[{index+1}/{len(data)}] ⏩ Skipping (All 4 metrics are perfect)")

    print(f"\nJob Finished. Successfully repaired {processed_count} queries.")

if __name__ == "__main__":
    safely_resume_ragas()
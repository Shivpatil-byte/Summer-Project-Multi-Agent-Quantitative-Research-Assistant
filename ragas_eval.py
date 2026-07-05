import json
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from workflow import app 

load_dotenv()

# =====================================================================
# SYSTEM CONFIGURATION
# =====================================================================
# Fix for the Groq API n=3 constraint
answer_relevancy.strictness = 1

print("Initializing Llama-3.1-8B-Instant as Judge...")
groq_judge_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=4096,
    timeout=120, 
    max_retries=3
)

eval_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# =====================================================================
# PHASE 1: LANGGRAPH PIPELINE EXECUTION
# =====================================================================
with open("golden_dataset.json", "r") as f:
    benchmark_data = json.load(f)

eval_records = []
print(f"\n--- PHASE 1: Executing {len(benchmark_data)} queries through LangGraph ---")

for index, record in enumerate(benchmark_data):
    print(f"[{index + 1}/{len(benchmark_data)}] Agent Processing: '{record['query']}'")
    
    initial_state = {
        "query": record["query"],
        "sub_questions": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "needs_reretrieval": False,
        "iteration_count": 0
    }
    
    final_state = app.invoke(initial_state)
    contexts = [chunk.get("text", "") for chunk in final_state.get("retrieved_chunks", [])]
    
    eval_records.append({
        "question": record["query"],
        "ground_truth": record["ground_truth"],
        "answer": final_state.get("answer", "No answer generated."),
        "contexts": contexts
    })

print("\nLangGraph execution complete. Moving to evaluation.")

# =====================================================================
# PHASE 2: SAFE RAGAS EVALUATION (ONE-BY-ONE)
# =====================================================================
print("\n--- PHASE 2: Ragas Evaluation via Groq 70B ---")
all_results = []

for i, row in enumerate(eval_records):
    print(f"\nGrading Row {i+1} of {len(eval_records)}...")
    
    single_dataset = Dataset.from_dict({
        "question": [row["question"]],
        "contexts": [row["contexts"]],
        "answer": [row["answer"]],
        "ground_truth": [row["ground_truth"]]
    })
    
    try:
        result = evaluate(
            dataset=single_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=groq_judge_llm,
            embeddings=eval_embeddings,
            run_config=RunConfig(max_workers=1, max_retries=3, timeout=120),
            raise_exceptions=False 
        )
        
        scores = result.to_pandas().to_dict(orient="records")[0]
        all_results.append(scores)
        print(f"✅ Success for Row {i+1}!")
        
    except Exception as e:
        print(f"❌ Row {i+1} failed: {e}")
        row['error'] = str(e)
        all_results.append(row)
    
    # CRITICAL FIX: 60-second cooldown guarantees the 12,000 TPM bucket resets
    if i < len(eval_records) - 1:
        print("Sleeping for 60 seconds to completely flush Groq's rolling token limit...")
        time.sleep(60)

# =====================================================================
# PHASE 3: EXPORT
# =====================================================================
print("\nSaving final scores to 'final_evaluation_results.json'...")
df = pd.DataFrame(all_results)
df.to_json("final_evaluation_results.json", orient="records", indent=4)
print("Finished! Evaluation complete.")
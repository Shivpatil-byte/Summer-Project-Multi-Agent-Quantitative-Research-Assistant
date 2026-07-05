import json
import time
import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from dotenv import load_dotenv
load_dotenv()

# Fix for the n=3 Groq constraint
answer_relevancy.strictness = 1

print("Loading existing evaluation file...")
with open("evaluation_results.json", "r", encoding="utf-8") as f:
    results_data = json.load(f)

rows = results_data.get("individual_results", [])
print(f"Successfully loaded {len(rows)} rows to evaluate!")

print("Initializing Llama-3.1-8B-Instant as Judge...")
groq_judge_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=2048,
    timeout=60, 
    max_retries=2
)

eval_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
all_results = []

# =====================================================================
# THE FIX: A strict, synchronous for-loop with a built-in rate limiter
# =====================================================================
print("\nStarting strict row-by-row evaluation...")

for i, row in enumerate(rows):
    print(f"\n--- Evaluating Row {i+1} of {len(rows)} ---")
    
    # Isolate a single row into its own dataset
    single_row_dict = {
        "question": [row["question"]],
        "contexts": [row["contexts"]],
        "answer": [row["answer"]],
        "ground_truth": [row["ground_truth"]]
    }
    single_dataset = Dataset.from_dict(single_row_dict)
    
    try:
        # Evaluate just this one question
        result = evaluate(
            dataset=single_dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=groq_judge_llm,
            embeddings=eval_embeddings,
            run_config=RunConfig(max_workers=1, max_retries=2, timeout=60),
            raise_exceptions=False # If one row fails, don't crash the loop
        )
        
        # Extract the scores and append to our master list
        scores = result.to_pandas().to_dict(orient="records")[0]
        all_results.append(scores)
        print(f"✅ Success for Row {i+1}!")
        
    except Exception as e:
        print(f"❌ Row {i+1} failed: {e}")
        row['error'] = str(e)
        all_results.append(row)
    
    # The Magic Ingredient: Let Groq's token bucket reset before the next question
    if i < len(rows) - 1:
        print("Sleeping for 10 seconds to clear Groq rate limits...")
        time.sleep(10)

# Save the final compiled results
print("\nSaving clean scores to 'patched_evaluation_results.json'...")
df = pd.DataFrame(all_results)
df.to_json("patched_evaluation_results.json", orient="records", indent=4)
print("Finished! Check 'patched_evaluation_results.json' for your real scores.")
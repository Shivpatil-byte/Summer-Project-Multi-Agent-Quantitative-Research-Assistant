import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load the Data
files = {
    "Hybrid (RRF + Cross-Encoder)": "cleaned_evaluation_results.json",
    "Dense Only (BGE-M3)": "dense_ablation_results.json",
    "BM25 Only (Sparse)": "bm25_ablation_results.json"
}

results = []
metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

print("Loading data and calculating RAGAS metrics...")

for strategy, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: {filepath} not found. Skipping.")
        continue
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Calculate the mean for each metric
    strategy_scores = {"Strategy": strategy}
    for metric in metrics:
        # Explicitly filter out any accidental NaNs just in case
        valid_scores = [item.get(metric) for item in data if item.get(metric) is not None and not np.isnan(item.get(metric))]
        strategy_scores[metric] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
    results.append(strategy_scores)

# 2. Build the DataFrame
df = pd.DataFrame(results)

# 3. Print the Markdown Table to Terminal
print("\n" + "="*50)
print(" WEEK 4 DELIVERABLE: ABLATION STUDY RESULTS")
print("="*50 + "\n")
markdown_table = df.to_markdown(index=False, floatfmt=".4f")
print(markdown_table)

# Optional: Save table to a text file for easy copying
with open("ablation_table.md", "w") as f:
    f.write("### Week 4 Deliverable: Ablation Study Results\n\n")
    f.write(markdown_table)
print("\n✅ Saved table to ablation_table.md")

# 4. Plot and Save the Graph
print("📊 Generating chart...")
df.set_index("Strategy").plot(
    kind="bar", 
    figsize=(10, 6), 
    colormap="viridis", 
    edgecolor="black"
)
plt.title("Retrieval Ablation Study: RAGAS Metrics Comparison", fontsize=14)
plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the image instead of just showing it
plt.savefig("ablation_chart.png", dpi=300)
print("✅ Saved chart to ablation_chart.png")

# Finally, display it on screen (close the window to end the script)
plt.show()
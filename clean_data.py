import json
import math

INPUT_FILE = "final_evaluation_results.json"
OUTPUT_FILE = "cleaned_evaluation_results.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only the rows where faithfulness is a valid, non-NaN number
cleaned_data = [
    item for item in data 
    if item.get("faithfulness") is not None 
    and not (isinstance(item.get("faithfulness"), float) and math.isnan(item.get("faithfulness")))
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4)

print(f"Filtered out broken rows. {len(cleaned_data)} perfect queries saved to {OUTPUT_FILE}!")
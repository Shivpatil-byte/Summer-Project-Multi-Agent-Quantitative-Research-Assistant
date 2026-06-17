import json
import random

def inspect_random_chunks(num_samples=10):
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: 'chunks.json' not found. Please run chunker.py first.")
        return

    if not chunks:
        print("The chunks file is empty.")
        return

    print(f"=== Total Chunks Generated: {len(chunks)} ===")
    print(f"=== Displaying {num_samples} Random Samples for Quality Control ===\n")
    
    # Select 10 random indices safely
    samples = random.sample(chunks, min(num_samples, len(chunks)))
    
    for idx, sample in enumerate(samples, 1):
        meta = sample["metadata"]
        print(f"--- SAMPLE #{idx} ---")
        print(f"Source: {meta['source']} | Year: {meta['year']} | Page: {meta['page_number']}")
        print(f"Section Header Guess: {meta['section_header']}")
        print(f"Sentence Range: {meta['sentence_start_idx']} to {meta['sentence_end_idx']}")
        print(f"Content:\n{sample['chunk_text']}\n")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    inspect_random_chunks()
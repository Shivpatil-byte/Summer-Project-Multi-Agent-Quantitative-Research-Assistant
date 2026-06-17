import json
import pickle
from rank_bm25 import BM25Okapi

def build_bm25_index():
    print("Loading chunks.json...")
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: 'chunks.json' not found.")
        return

    print("Tokenizing corpus for BM25...")
    # Basic tokenization: lowercase and split by whitespace
    # For a more advanced pipeline, you could use NLTK or SpaCy here
    tokenized_corpus = [chunk["chunk_text"].lower().split() for chunk in chunks]

    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the index AND the chunks together so we can easily map scores back to text
    output_file = "bm25_index.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    print(f"Success! BM25 index saved to '{output_file}'.")

if __name__ == "__main__":
    build_bm25_index()
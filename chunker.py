import json
import re

def split_into_sentences(text):
    # We split the look-behinds into distinct, fixed-width groups:
    # Group 1 (2 chars): Mr, St, Ms, Dr, Co
    # Group 2 (3 chars): Mrs, Ltd, Inc, e.g, i.e
    # Group 3 (4 chars): Corp
    sentence_end = re.compile(
        r'(?<!\b(?:Mr|St|Ms|Dr|Co)\.)'
        r'(?<!\b(?:Mrs|Ltd|Inc|e\.g|i\.e)\.)'
        r'(?<!\bCorp\.)'
        r'(?<=[.!?])\s+'
        r'(?=[A-Z0-9])'
    )
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]

def create_sentence_window_chunks(window_size=5, overlap=1):
    try:
        with open("parsed_pages.json", "r", encoding="utf-8") as f:
            pages = json.load(f)
    except FileNotFoundError:
        print("Error: 'parsed_pages.json' not found. Please run parser.py first.")
        return []
        
    all_chunks = []
    
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        
        sentences = split_into_sentences(text)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            continue
            
        # Sliding window algorithm
        step = window_size - overlap
        for i in range(0, num_sentences, step):
            window_sentences = sentences[i : i + window_size]
            
            # Skip tiny trailing sentence fragments if we've already chunked the page
            if len(window_sentences) <= overlap and i > 0:
                continue
                
            chunk_text = " ".join(window_sentences)
            
            chunk_metadata = metadata.copy()
            chunk_metadata["sentence_start_idx"] = i
            chunk_metadata["sentence_end_idx"] = i + len(window_sentences) - 1
            
            all_chunks.append({
                "chunk_text": chunk_text,
                "metadata": chunk_metadata
            })
            
    output_file = "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"Chunking complete! Generated {len(all_chunks)} chunks and saved to '{output_file}'.")
    return all_chunks

if __name__ == "__main__":
    create_sentence_window_chunks()
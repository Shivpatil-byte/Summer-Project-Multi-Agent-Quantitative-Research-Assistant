import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_retrieval(query):
    print(f"==================================================")
    print(f"QUERY: '{query}'")
    print(f"==================================================\n")

    # --- 1. TEST DENSE RETRIEVAL (CHROMA + BGE-M3) ---
    print("Loading BGE-M3 and ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    chroma_results = vectorstore.similarity_search(query, k=5)
    
    print("\n--- TOP 5 DENSE RESULTS (CHROMA) ---")
    for i, res in enumerate(chroma_results, 1):
        meta = res.metadata
        print(f"[{i}] Year: {meta['year']} | Page: {meta['page_number']} | Source: {meta['source']}")
        # Print the first 200 characters of the chunk to verify relevance
        print(f"Text: {res.page_content[:200]}...\n")


    # --- 2. TEST SPARSE RETRIEVAL (BM25) ---
    print("Loading BM25 Index...")
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top 5 highest scores
    top_n = 5
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]

    print("--- TOP 5 SPARSE RESULTS (BM25) ---")
    for i, idx in enumerate(top_indices, 1):
        chunk = chunks[idx]
        score = bm25_scores[idx]
        meta = chunk['metadata']
        print(f"[{i}] Score: {score:.2f} | Year: {meta['year']} | Page: {meta['page_number']}")
        print(f"Text: {chunk['chunk_text'][:200]}...\n")

if __name__ == "__main__":
    # Test with a highly specific query that contains both concepts and precise keywords
    sample_query = "What was the organic sales growth and performance of Purina PetCare in 2024?"
    test_retrieval(sample_query)
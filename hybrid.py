import pickle
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(self, chroma_dir="chroma_db", bm25_path="bm25_index.pkl"):
        print("Initializing Dense Retriever (BGE-M3)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=self.embeddings)
        
        print("Initializing Sparse Retriever (BM25)...")
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.chunks = bm25_data["chunks"]

        print("Initializing Cross-Encoder Reranker (MS-MARCO)...")
        # MS-MARCO is a model trained by Microsoft specifically for search relevance
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    def _get_bm25_top_n(self, query, top_n=20, year_filter=None):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = []
        rank = 1
        for idx in top_indices:
            if scores[idx] <= 0: continue
            chunk = self.chunks[idx]
            if year_filter and chunk["metadata"].get("year") != year_filter: continue
                
            results.append({
                "text": chunk["chunk_text"],
                "metadata": chunk["metadata"],
                "score": scores[idx], # Keep raw score for the comparison tool
                "rank": rank
            })
            rank += 1
            if len(results) >= top_n: break
        return results

    def _get_chroma_top_n(self, query, top_n=20, year_filter=None):
        filter_dict = {"year": year_filter} if year_filter else None
        # Use relevance scores to show how Dense scores it internally
        chroma_res = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=top_n, filter=filter_dict
        )
        
        results = []
        for rank, (doc, score) in enumerate(chroma_res, 1):
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "rank": rank
            })
        return results

    def advanced_search(self, query, top_k_initial=10, final_top_k=5, year_filter=None):
        """Executes RRF fusion, then strictly reranks the top results using a Cross-Encoder."""
        
        # --- STAGE 1: Fast Retrieval & Fusion ---
        bm25_results = self._get_bm25_top_n(query, top_n=20, year_filter=year_filter)
        chroma_results = self._get_chroma_top_n(query, top_n=20, year_filter=year_filter)
        
        fused_docs = {}
        for item in bm25_results:
            text = item["text"]
            fused_docs[text] = {"metadata": item["metadata"], "rrf_score": 1.0 / (60 + item["rank"])}

        for item in chroma_results:
            text = item["text"]
            if text in fused_docs:
                fused_docs[text]["rrf_score"] += 1.0 / (60 + item["rank"])
            else:
                fused_docs[text] = {"metadata": item["metadata"], "rrf_score": 1.0 / (60 + item["rank"])}

        # Get the Top 10 from the fusion
        sorted_fused = sorted(fused_docs.items(), key=lambda x: x[1]["rrf_score"], reverse=True)[:top_k_initial]
        
        # --- STAGE 2: Deep Cross-Encoder Reranking ---
        # Format pairs for the Cross-Encoder: [[query, doc1], [query, doc2], ...]
        sentence_pairs = [[query, doc_text] for doc_text, _ in sorted_fused]
        
        # Predict relevance scores (Higher is better)
        cross_scores = self.reranker.predict(sentence_pairs)
        
        # Attach the new strict scores to our documents
        reranked_results = []
        for i, (text, info) in enumerate(sorted_fused):
            reranked_results.append({
                "text": text,
                "metadata": info["metadata"],
                "rrf_score": info["rrf_score"],
                "cross_score": float(cross_scores[i]) # The ultimate decider
            })
            
        # Sort entirely by the new Cross-Encoder score
        final_sorted = sorted(reranked_results, key=lambda x: x["cross_score"], reverse=True)
        
        return final_sorted[:final_top_k]

# ==========================================
# DIAGNOSTIC COMPARISON TOOL
# ==========================================
if __name__ == "__main__":
    retriever = AdvancedRetriever()
    
    query = "What was the organic sales growth and performance of Purina PetCare in 2024?"
    filter_year = 2024
    
    print(f"\n" + "="*80)
    print(f"QUERY: '{query}'")
    print("="*80)

    # 1. DENSE ONLY (Chroma)
    print("\n--- [1] DENSE ONLY (Top 3) ---")
    dense_res = retriever._get_chroma_top_n(query, top_n=3, year_filter=filter_year)
    for i, res in enumerate(dense_res, 1):
        print(f"  {i}. [Score: {res['score']:.4f}] {res['text'][:100]}...")

    # 2. SPARSE ONLY (BM25)
    print("\n--- [2] SPARSE ONLY (Top 3) ---")
    sparse_res = retriever._get_bm25_top_n(query, top_n=3, year_filter=filter_year)
    for i, res in enumerate(sparse_res, 1):
        print(f"  {i}. [Score: {res['score']:.2f}] {res['text'][:100]}...")

    # 3. HYBRID + RERANKER (The Final Pipeline)
    print("\n--- [3] HYBRID + CROSS-ENCODER (Top 5) ---")
    final_res = retriever.advanced_search(query, top_k_initial=10, final_top_k=5, year_filter=filter_year)
    for i, res in enumerate(final_res, 1):
        print(f"  {i}. [Cross-Score: {res['cross_score']:>6.2f} | RRF Base: {res['rrf_score']:.4f}]")
        print(f"     Source: Page {res['metadata']['page_number']}")
        print(f"     {res['text'][:150]}...\n")


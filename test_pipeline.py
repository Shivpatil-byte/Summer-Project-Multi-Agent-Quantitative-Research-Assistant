import unittest
import logging
import sys
from hybrid import AdvancedRetriever

# Configure production-grade logging
# This forces logs to output to the console with a clean, timestamped format
logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TestRetrievalPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        setUpClass runs exactly ONCE before any tests start.
        We initialize the heavy AI models here so they don't reload for every single test.
        """
        logger.info("TEST SETUP: Loading AI Models and Databases... (This may take a moment)")
        cls.retriever = AdvancedRetriever()
        logger.info("TEST SETUP: Models loaded successfully. Starting tests.")

    def log_metadata(self, test_name, results, metric_key):
        """Helper function to cleanly format and log the required metadata."""
        logger.info(f"--- LOGGING METADATA FOR: {test_name} ---")
        for i, res in enumerate(results, 1):
            meta = res['metadata']
            score = res.get(metric_key, "N/A")
            
            # Format to 4 decimal places if it's a float
            if isinstance(score, float):
                score = f"{score:.4f}"
                
            logger.info(
                f"Rank [{i}] | Score ({metric_key}): {score} | "
                f"Source: {meta.get('source')} | Page: {meta.get('page_number')} | Year: {meta.get('year')}\n"
                f"Snippet: {res['text'][:100]}..."
            )

    def test_sparse_retrieval(self):
        """Test if BM25 correctly tokenizes and fetches exact keyword matches."""
        query = "Felix Megamix and Gourmet Nature"
        logger.info(f"Executing test_sparse_retrieval with query: '{query}'")
        
        results = self.retriever._get_bm25_top_n(query, top_n=5)
        
        # 1. Assert we got results back
        self.assertGreater(len(results), 0, "BM25 failed to return any results.")
        
        # 2. Assert the required metadata keys exist in the first result
        first_meta = results[0]['metadata']
        self.assertIn("source", first_meta, "Metadata missing 'source' key.")
        self.assertIn("page_number", first_meta, "Metadata missing 'page_number' key.")
        
        # 3. Log the metadata
        self.log_metadata("SPARSE (BM25)", results, metric_key="score")

    def test_dense_retrieval_with_filter(self):
        """Test if ChromaDB semantic search respects hard metadata filtering."""
        query = "environmental sustainability and carbon emissions"
        target_year = 2023
        logger.info(f"Executing test_dense_retrieval_with_filter with query: '{query}' | Year: {target_year}")
        
        results = self.retriever._get_chroma_top_n(query, top_n=5, year_filter=target_year)
        
        self.assertGreater(len(results), 0, "Chroma failed to return any results.")
        
        # 1. Assert that the database filter actually worked
        for res in results:
            self.assertEqual(
                res['metadata']['year'], 
                target_year, 
                f"Filter failed! Expected {target_year}, got {res['metadata']['year']}"
            )
            
        # 2. Log the metadata
        self.log_metadata("DENSE (CHROMA)", results, metric_key="score")

    def test_advanced_hybrid_reranker(self):
        """Test the full pipeline: BM25 + Chroma -> RRF -> Cross-Encoder."""
        query = "What was the organic sales growth of Purina PetCare in 2024?"
        target_year = 2024
        logger.info(f"Executing test_advanced_hybrid_reranker with query: '{query}'")
        
        results = self.retriever.advanced_search(
            query, 
            top_k_initial=10, 
            final_top_k=3, 
            year_filter=target_year
        )
        
        # 1. Assert we narrowed down exactly to the top 3 requested
        self.assertEqual(len(results), 3, "Advanced search did not return exactly 3 results.")
        
        # 2. Assert that the RRF and Cross-Encoder scores were successfully injected
        first_result = results[0]
        self.assertIn("rrf_score", first_result, "Pipeline failed to inject RRF base score.")
        self.assertIn("cross_score", first_result, "Pipeline failed to inject Cross-Encoder score.")
        
        # 3. Assert sorting order is strictly descending by cross_score
        scores = [res["cross_score"] for res in results]
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), 
                        "Results are not properly sorted by cross_score!")
        
        # 4. Log the metadata
        self.log_metadata("ADVANCED (HYBRID + RERANK)", results, metric_key="cross_score")

if __name__ == "__main__":
    # We use verbosity=2 to get detailed PASS/FAIL readouts in the terminal
    unittest.main(verbosity=2)
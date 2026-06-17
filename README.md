# Multi-Agent Quantitative Research Assistant

This repository contains the foundational data ingestion and hybrid retrieval pipeline for a financial quantitative research assistant. Week 1 focused on building a local, robust, and privacy-preserving ingestion engine capable of handling dense, multi-column corporate financial documents.

---

## Why Nestlé?

Nestlé's annual reports serve as an ideal benchmark for financial Retrieval-Augmented Generation (RAG) systems. These documents are notoriously difficult to parse because they feature complex multi-column layouts, mixed text-and-graph structures, embedded financial tables, and extensive footprints of dense corporate metrics. Successfully processing Nestlé reports ensures the pipeline is robust enough to handle almost any standard corporate financial document.

---

## Architecture Flow

```
[Raw PDFs] ➔ [parser.py (PyMuPDF)] ➔ [chunker.py (Regex Split)]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
         [embedder.py (BGE-M3 + ChromaDB)]           [bm25.py (Rank_BM25 Index)]
                 (Semantic/Dense)                             (Keyword/Sparse)
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                 [verify_retrieval.py]

```

---

## Prerequisites & Installation

To set up the environment, the following core libraries were installed:

```bash
pip install pymupdf langchain-chroma langchain-huggingface rank_bm25

```

* **`pymupdf` (imported as `fitz`):** A high-performance PDF parsing library used to extract raw text blocks page-by-page.
* **`langchain-chroma` & `langchain-huggingface`:** Standard standalone modular packages used to manage local vector databases and run embedding models.
* **`rank_bm25`:** A library implementing the BM25 algorithm for traditional keyword-based lexical ranking.

---

## Ingestion Pipeline Breakdown (Week 1)

### 1. Extraction Layer (`parser.py`)

* **What it does:** Reads the raw annual reports folder, dynamically extracts the document year from the filename via regular expressions, and extracts text page-by-page.
* **Key Engineering Details:** Rather than executing a blind dump, it includes a cleaning engine that strips out layout noise, such as chart borders, isolated table pipeline markers (`|`), and fragmented single-letter timeline axes markers (like `J F M A M J...`). It stores the cleaned pages in `parsed_pages.json` along with structured metadata (`source`, `year`, `page_number`, `section_header`).

### 2. Processing Layer (`chunker.py`)

* **Correction Note:** The pipeline processes text at the **sentence level**, not raw layout lines. Slicing by literal layout lines breaks mid-phrase due to column wraps.
* **What it does:** Uses a negative look-behind regex compiler to split text into distinct sentences without getting tripped up by financial abbreviations (e.g., *Inc.*, *Ltd.*, *Mio.*).
* **The Sliding Window:** Groups the text into chunks of **5 sentences** with a **1-sentence overlap** between contiguous chunks. This ensures that overlapping context is preserved if an important insight spans the boundary between chunks. Outputs are stored in `chunks.json`.

### 3. Dense & Sparse Indexing Layer (`embedder.py` & `bm25.py`)

* **Correction Note:** The pipeline generates vectors for **entire text chunks**, not for individual isolated sentences or single words. This preserves the multi-sentence context within a single spatial coordinate.
* **Dense Retrieval (`embedder.py`):** Loads the open-source **`BAAI/bge-m3`** embedding model via HuggingFace. It compresses the semantic meaning of each 5-sentence chunk into a 1,024-dimensional vector space and saves it into a local **ChromaDB** instance (`chroma_db/`).
* **Sparse Retrieval (`bm25.py`):** Simultaneously constructs an in-memory lexical keyword lookup index using the **BM25Okapi** algorithm. It then serializes this index alongside the text data into a binary file (`bm25_index.pkl`) using Python's native **`pickle`** library for instant retrieval sessions later.

### 4. QA & Diagnostics (`verify_retrieval.py`)

* **What it does:** Serves as a testing ground to compare semantic query matching against exact term frequency matching side-by-side.
* **The Workflow:** Given a targeted query like *"What was the organic sales growth and performance of Purina PetCare in 2024?"*, it queries ChromaDB for the top 5 closest conceptual matches while running a tokenized keyword match through the unpickled BM25 index. This allows manual inspection to ensure both embedding alignment and keyword tracking are accurate before moving into the orchestration phase.
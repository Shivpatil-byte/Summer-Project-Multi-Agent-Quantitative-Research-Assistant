# Multi-Agent Quantitative Research Assistant

This repository contains the foundational data ingestion and hybrid retrieval pipeline for a financial quantitative research assistant. Week 1 focused on building a local, robust, and privacy-preserving ingestion engine capable of handling dense, multi-column corporate financial documents.

---
*************************************************WEEK 1*********************************************

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


*******************************************WEEK 2*******************************************************

## Week 2: Advanced Hybrid Retrieval Pipeline

This phase focused on building a production-grade, two-stage ingestion and retrieval engine. The architecture combines semantic vector search with statistical keyword matching, unified by mathematical rank fusion, and finalized with an AI-driven cross-encoder for absolute contextual precision.

### Structural Trade-offs & Engineering Decisions

**1. Architectural Selection: The Retrieval Stack Benefits**
Instead of relying on a single retrieval method, our pipeline leverages a tailored multi-engine stack to maximize accuracy:
* **BGE-M3 (Dense Vector Search):** Chosen over standard embedding models because of its state-of-the-art multi-linguality and ability to capture deep semantic nuances and sentence context, ensuring the engine understands the *intent* behind financial queries.
* **BM25 (Sparse Keyword Search):** Maintained alongside the vector database because semantic models often struggle with exact matches. BM25 serves as a mathematical guarantee that specific alphanumeric terms, product codes (e.g., "Nespresso"), and precise financial nomenclature are never overlooked.
* **ChromaDB (Local Vector Database):** Selected over cloud-hosted alternatives like Pinecone to eliminate external network latency and infrastructure costs. It provides native, lightning-fast metadata filtering out of the box, which is vital for isolating specific document segments locally.

**2. Overcoming Temporal Confusion**
* **The Issue:** Financial documents across different years (e.g., 2023 vs. 2024 Annual Reviews) share highly identical language, structures, and section headers. Without strict controls, semantic search suffers from **Temporal Confusion**—accidentally retrieving a highly relevant-looking chunk from 2023 to answer a question explicitly asked about 2024 performance, causing catastrophic hallucination.
* **The Resolution:** We implemented hard **Metadata Filtering** directly inside the ChromaDB and BM25 queries. By extracting and passing a strict `year_filter` through the state, the retrieval engines programmatically drop all data chunks that do not match the target year before semantic or keyword scoring even begins.

**3. Binary Serialization via Pickle (.pkl vs .json)**
The BM25 sparse index is serialized directly into a binary `.pkl` (Pickle) format rather than a standard text-based `.json`. JSON requires the application to sequentially parse string data and rebuild complex keyword frequency matrices line-by-line during every boot sequence. Pickle serializes the exact live Python class object straight into binary. This allows the system to instantly load the pre-computed mathematical index directly into RAM, dropping loading times down to milliseconds.

**4. The Latency Trap (Cross-Encoder Optimization)**
While a Cross-Encoder is incredibly precise because it evaluates the query and document simultaneously via intensive multi-head attention, it scales terribly. Processing a single document takes roughly 100ms. If we were to pass 100 or more candidate documents to the Cross-Encoder, the user would suffer a 10-second loading bottleneck. To prevent this, we built a **RAG Funnel**: fast, pre-computed Stage 1 retrievers filter thousands of chunks down to 10 candidates using Reciprocal Rank Fusion (RRF), allowing the heavy Cross-Encoder to execute its precise math strictly on the top 10 choices without killing user experience.
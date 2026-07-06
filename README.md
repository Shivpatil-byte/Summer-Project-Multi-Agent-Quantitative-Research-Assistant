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

Following the initial data ingestion, the focus shifted to building a robust, self-correcting retrieval pipeline, proving its efficacy through rigorous empirical evaluation, and deploying it into a production-grade user interface.

The LangGraph Multi-Agent Workflow (Week 3)
Instead of a linear retrieval chain, we constructed a cyclic, stateful multi-agent system using LangGraph. The workflow is managed via a shared AgentState dictionary and consists of three primary nodes:

The Orchestrator: Receives the user query and decomposes complex financial questions into 2-3 targeted sub-questions.

The Retriever: Executes a Hybrid Retrieval (Dense + BM25) for each sub-question, aggregates the chunks, and deduplicates them.

The Analyst & Critic: The Analyst synthesizes the answer and extracts exact citations (document, year, page). The Critic then evaluates the answer against the retrieved chunks. If the confidence score falls below 0.7, it triggers a conditional edge, forcing the system back to the Retriever node for a second iteration before terminating.

Evaluation Framework and Benchmarking (Week 4)
To empirically prove the system's accuracy, we built a rigorous 30-query benchmark dataset manually derived from the corporate reports. This included:

8 Factual queries (e.g., exact net profit figures).

10 Multi-hop queries requiring cross-referencing over multiple years.

7 Trend-analysis queries.

5 Adversarial queries (data deliberately missing from the corpus).

We ran this benchmark through the RAGAS evaluation pipeline. Furthermore, an ablation study was conducted on this dataset to compare Dense-only, BM25-only, and Hybrid retrieval strategies, conclusively proving the Hybrid approach yields the highest Context Recall.

Frontend Deployment (Week 5)
The multi-agent backend was integrated into a Streamlit application designed for financial professionals. Features include:

A chat interface with a rolling 5-query session memory.

Dynamic Confidence Badges (Green, Amber, Red) based on the Critic node's evaluation.

An expandable Retrieval Diagnostics sidebar exposing the exact chunks and relevance scores.

Strict handling of adversarial queries, returning an explicit "insufficient information" refusal rather than hallucinating financial data.

Part 2: Engineering Roadblocks and Resolutions
Building this system required navigating complex state management and dependency issues. Below is a breakdown of the primary roadblocks and their resolutions.

1. The Faithfulness Paradox and Hallucination Control
The Issue: During early testing, the model would attempt to answer adversarial queries using its parametric memory rather than the provided context, leading to high "faithfulness" scores in RAGAS despite poor retrieval precision.

The Solution: We implemented the Critic node within the LangGraph workflow. By forcing a dedicated LLM call to act purely as a verification agent, the system checks every generated claim against the retrieved chunks. If the data is absent, the system explicitly refuses to answer, successfully neutralizing financial hallucinations.

2. Streamlit State Management and Memory Wipes
The Issue: When integrating the LangGraph backend with Streamlit, the assistant's responses would occasionally render and then immediately vanish.

The Solution: This was a race condition in Streamlit's rendering lifecycle. The code was triggering a UI rerun to update the sidebar before the assistant's final response was appended to the session state. The execution order was strictly refactored to format the response, save it to the session memory, and only then trigger the UI refresh.

3. Dependency Conflicts and Missing Retrievers
The Issue: The backend crashed entirely with a module not found error regarding langchain.retrievers. LangChain's transition to version 0.3 deprecated and relocated several core retrieval modules used in our hybrid setup.

The Solution: Rather than rewriting the retrieval architecture to chase moving API targets, we installed the langchain-classic package and pinned our core, community, and partner packages to a synchronized version state, restoring the Ensemble and Cross-Encoder retrievers.

4. Vector Database Disconnect (FAISS vs. ChromaDB)
The Issue: The application successfully launched but utilized a fallback dummy dataset, failing to answer queries about the actual financial reports ingested during Week 2.

The Solution: The backend was initially looking for a FAISS index directory. We updated the connection logic to target the persistent chroma_db sqlite implementation used during the initial ingestion phase. Additionally, we wrote a custom extraction layer to pull the raw text back out of ChromaDB into memory, ensuring the BM25 sparse retriever was perfectly synchronized with the dense vector store.

Part 3: Architectural Decisions and Alternatives
Every component in this pipeline was chosen to solve specific problems inherent in processing quantitative financial text.

LangGraph Multi-Agent vs. Linear LangChain
Why we chose LangGraph: A standard linear chain retrieves data once and attempts to answer. If the retrieval is poor, the answer is poor. By using LangGraph, we introduced cyclic state management. The Critic node allows the system to realize it lacks sufficient data and autonomously loop back to the Retriever with refined parameters. This significantly increases accuracy on complex, multi-hop financial queries compared to a single-pass system.

Hybrid Search (Dense + Sparse) vs. Dense-Only Search
Why we chose Hybrid: Dense embeddings (like BGE-M3) excel at semantic similarity but fail at exact keyword matching. In finance, queries often depend on specific years, ticker symbols, or acronyms. By fusing ChromaDB (semantic) with BM25 (sparse keyword search) via Reciprocal Rank Fusion, we ensure both the contextual meaning and the exact quantitative terminology are captured.

RAGAS Evaluation vs. Human-in-the-Loop Testing
Why we chose RAGAS: Relying solely on manual "vibe checks" is insufficient for a production financial tool. RAGAS allowed us to quantify performance across four strict dimensions (Faithfulness, Relevancy, Precision, Recall). By building a static 30-query ground-truth dataset, we could run programmatic ablation studies, proving with hard numbers that our architectural changes (like adding the Cross-Encoder) actually improved the system.

Streamlit vs. Custom React/FastAPI Stack
Why we chose Streamlit: Building a separated frontend and backend API introduces massive overhead for data pipelines. Streamlit allowed us to keep the entire stack in Python while natively handling the complex session state required for the 5-query conversation memory. Its rapid prototyping components were ideal for building the dynamic sidebar diagnostics necessary for users to audit the retrieved financial data.

Code snippet
graph TD
    A[User Query] --> B(Streamlit UI)
    B --> C[Orchestrator Node: Query Decomposition]
    
    C --> D[Retriever Node: Hybrid Search]
    D --> E[(ChromaDB + BM25)]
    E --> F[Cross-Encoder Re-ranker]
    
    F --> G[Analyst Node: Synthesis & Citations]
    G --> H[Critic Node: Verification]
    
    H -->|Confidence < 0.7 & Iterations < 2| D
    H -->|Confidence >= 0.7| I[Final Response Payload]
    I --> B
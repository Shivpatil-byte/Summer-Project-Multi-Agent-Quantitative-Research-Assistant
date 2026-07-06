import os
import json
from dotenv import load_dotenv

# LangChain & Groq Core Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrieval Components (Swapped to Chroma!)
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# Classic Imports (From our dependency fix)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Telemetry and Environment Control
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
load_dotenv()

# Pointing to your exact folder from the screenshot
INDEX_DIR = "chroma_db"

# ==========================================
# 1. INITIALIZE SHARED MODELS
# ==========================================
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# ==========================================
# 2. AUTO-INITIALIZE RETRIEVERS (Chroma Version)
# ==========================================
def initialize_retrievers():
    default_docs = [
        Document(
            page_content="Fallback dummy data. If you see this, Chroma didn't load properly.",
            metadata={"source": "System", "year": "2024", "page": "1"}
        )
    ]

    # 1. Setup Dense Retriever (Chroma)
    if os.path.exists(INDEX_DIR):
        print(f"📁 Found existing Chroma index at '{INDEX_DIR}'. Loading vector database...")
        vectorstore = Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
    else:
        print(f"⚠️ No folder named '{INDEX_DIR}' found. Bootstrapping dummy data...")
        vectorstore = Chroma.from_documents(default_docs, embeddings, persist_directory=INDEX_DIR)
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. Setup Sparse Retriever (BM25)
    # We must extract the raw text back out of Chroma so BM25 knows what words to search for.
    all_docs = default_docs
    if os.path.exists(INDEX_DIR):
        try:
            db_data = vectorstore.get()
            if db_data['documents']:
                all_docs = [
                    Document(page_content=txt, metadata=meta or {})
                    for txt, meta in zip(db_data['documents'], db_data['metadatas'])
                ]
                print(f"✅ Successfully extracted {len(all_docs)} chunks from Chroma for BM25 keyword search.")
        except Exception as e:
            print(f"⚠️ Error extracting documents for BM25: {e}")

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 10

    # 3. Formulate the Full Hybrid RRF + Cross-Encoder Pipeline
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], 
        weights=[0.5, 0.5]
    )
    
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

active_retriever = initialize_retrievers()

# ==========================================
# 3. RUNTIME PIPELINE LOGIC
# ==========================================
def get_hybrid_answer(query: str, chat_history: list, retriever=active_retriever):
    # 1. Retrieve & Re-rank Chunks
    docs = retriever.invoke(query)
    
    # 2. Format Context Window
    context_text = "\n\n".join([
        f"Source: {d.metadata.get('source', 'Unknown')} | Year: {d.metadata.get('year', 'N/A')} | Page: {d.metadata.get('page', 'N/A')}\n{d.page_content}" 
        for d in docs
    ])
    
    # 3. Generation Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a specialized quantitative research assistant. "
            "Answer the user's question based strictly on the provided context. "
            "If the answer is not contained within the text, explicitly state that you do not have the data."
        )),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # 4. Chain Execution
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": query})
    
    # 5. Extract Metadata, Scores, and Citations
    retrieved_chunks = []
    citations = []
    
    for d in docs:
        source = d.metadata.get("source", "Financial Report")
        year = d.metadata.get("year", "N/A")
        page = d.metadata.get("page", "N/A")
        
        score = getattr(d, 'metadata', {}).get('relevance_score', 0.85)
        if hasattr(d, 'relevance_score'):
            score = d.relevance_score

        citations.append({"doc": source, "year": year, "page": page})
        retrieved_chunks.append({
            "score": float(score),
            "document": source,
            "year": year,
            "page": page,
            "text": d.page_content
        })
        
    unique_citations = [dict(t) for t in {tuple(d.items()) for d in citations}]

    return {
        "answer": answer,
        "citations": unique_citations,
        "confidence": retrieved_chunks[0]["score"] if retrieved_chunks else 0.5,
        "retrieved_chunks": retrieved_chunks
    }
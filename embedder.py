import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_vector_store():
    print("Loading chunks.json...")
    try:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: 'chunks.json' not found. Run chunker.py first.")
        return

    # Convert raw JSON chunks into LangChain Document objects
    documents = [
        Document(page_content=chunk["chunk_text"], metadata=chunk["metadata"]) 
        for chunk in chunks
    ]
    print(f"Loaded {len(documents)} documents.")

    print("Initializing BGE-M3 Embedding Model...")
    # BGE-M3 settings. Using normalize_embeddings=True is highly recommended for BGE models.
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'} # Change to 'cuda' or 'mps' if you have a dedicated GPU
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("Embedding documents and building ChromaDB... (This may take several minutes)")
    persist_directory = "chroma_db"
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Success! Vector store saved to '{persist_directory}/'.")

if __name__ == "__main__":
    build_vector_store()
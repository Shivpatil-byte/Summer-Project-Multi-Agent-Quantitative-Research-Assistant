import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_reports(directory="reports"):
    documents = []
    
    # Check if directory exists and contains files
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Error: The '{directory}' folder is missing or empty. Please add the PDFs.")
        return documents

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filename}...")
            
            # Load the PDF using PyMuPDF
            loader = PyMuPDFLoader(filepath)
            documents.extend(loader.load())
    
    print(f"Successfully loaded {len(documents)} total pages.")
    return documents


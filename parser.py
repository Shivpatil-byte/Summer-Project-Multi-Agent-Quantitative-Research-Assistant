import os
import json
import re
import fitz  # PyMuPDF

def extract_year_from_filename(filename):
    # Uses a robust regex to find any 4-digit year starting with 20 (e.g., 2021-2029)
    match = re.search(r'\b(20\d{2})\b', filename)
    if match:
        return int(match.group(1))
    return "Unknown"

def clean_extracted_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = line.strip()
        
        # 1. Skip pure chart structural lines (e.g., "|", "|||")
        if re.match(r'^[|—\s-]+$', cleaned_line):
            continue
            
        # 2. Skip horizontal axis label patterns (e.g., "J F M A M J J A S O N D")
        # Matches strings consisting mostly of single capital letters separated by spaces
        if re.match(r'^[A-Z](\s+[A-Z]){4,}$', cleaned_line):
            continue
            
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
            
    return "\n".join(cleaned_lines)

def parse_pdf_documents(directory="reports"):
    parsed_data = []
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return []

    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(directory, filename)
        year = extract_year_from_filename(filename)
        print(f"Parsing {filename} (Detected Year: {year})...")
        
        doc = fitz.open(filepath)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            
            # Run text through our cleaning engine
            text = clean_extracted_text(raw_text)
            
            if not text:
                continue
                
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Skip page tracking numbers if they act as headers
            section_header = "General Content"
            for line in lines:
                if line.isdigit() or len(line) < 3:
                    continue
                section_header = line
                break
            
            if len(section_header) > 60:
                section_header = section_header[:57] + "..."

            parsed_data.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "year": year,
                    "page_number": page_num + 1,
                    "section_header": section_header
                }
            })
            
    output_file = "parsed_pages.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nParsing complete! Cleaned and stored {len(parsed_data)} pages in '{output_file}'.")
    return parsed_data

if __name__ == "__main__":
    parse_pdf_documents()
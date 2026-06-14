import os
import requests

def download_nestle_reports():
    # Create a directory to hold the PDFs
    os.makedirs("reports", exist_ok=True)

    # URLs for the last 5 Nestlé Annual Reviews (2021 - 2025)
    # The URL pattern typically includes the year it was published (e.g., 2025 report published in early 2026)
    reports = {
        "Nestle_Annual_Review_2025.pdf": "https://www.nestle.com/sites/default/files/2026-02/annual-review-2025-en.pdf",
        "Nestle_Annual_Review_2024.pdf": "https://www.nestle.com/sites/default/files/2025-02/annual-review-2024-en.pdf",
        "Nestle_Annual_Review_2023.pdf": "https://www.nestle.com/sites/default/files/2024-02/annual-review-2023-en.pdf",
        "Nestle_Annual_Review_2022.pdf": "https://www.nestle.com/sites/default/files/2023-02/annual-review-2022-en.pdf",
        "Nestle_Annual_Review_2021.pdf": "https://www.nestle.com/sites/default/files/2022-02/annual-review-2021-en.pdf"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for filename, url in reports.items():
        filepath = os.path.join("reports", filename)
        print(f"Downloading {filename}...")
        
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status() # Check for HTTP errors
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully saved: {filepath}")
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}. Error: {e}")

if __name__ == "__main__":
    download_nestle_reports()
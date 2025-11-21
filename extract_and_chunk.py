# extract_and_chunk.py
import os
import fitz  # PyMuPDF
import pandas as pd
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------- Config ----------
DEFAULT_CHUNK_CSV = "policy_chunks.csv"
DEFAULT_PDF_DIR = "policy_pdfs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# ---------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads and extracts text from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts)
    except Exception as e:
        print(f"[WARN] Failed to read {pdf_path}: {e}")
        return ""

def simple_chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks using a robust splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text or "")

def extract_text_from_pdfs(file_paths: List[str]) -> pd.DataFrame:
    """
    Read a list of PDF paths and return a DataFrame of chunks:
    columns = [document_name, chunk_id, chunk_text]
    """
    all_chunks = []
    for pdf in file_paths:
        base = os.path.basename(pdf)
        print(f"[INFO] Processing PDF → {base}")
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            continue
        chunks = simple_chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "document_name": base,
                "chunk_id": f"{base}_chunk_{i}",
                "chunk_text": ch
            })
    return pd.DataFrame(all_chunks)

def process_folder(folder_path=DEFAULT_PDF_DIR, output_csv=DEFAULT_CHUNK_CSV) -> pd.DataFrame:
    """Process all PDFs in a folder and write/overwrite the chunk CSV."""
    os.makedirs(folder_path, exist_ok=True)
    pdfs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"[WARN] No PDFs found in '{folder_path}'.")
        return pd.DataFrame(columns=["document_name", "chunk_id", "chunk_text"])
    df = extract_text_from_pdfs(pdfs)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"[OK] Saved {len(df)} chunks → {output_csv}")
    return df

if __name__ == "__main__":
    os.makedirs(DEFAULT_PDF_DIR, exist_ok=True)
    print("[RUN] Processing all PDFs in 'policy_pdfs' ...")
    process_folder()

# incremental_index.py
import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from extract_and_chunk import extract_text_from_pdfs

# ---------- Config ----------
EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_CSV = "policy_chunks.csv"
FAISS_INDEX = "policy_index.faiss"
FILE_TRACK = "file_index.json"    # track processed files (size+mtime)
UPLOAD_DIR = "uploaded_pdfs"      # where Streamlit saves uploaded PDFs
# ----------------------------

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _hash_file(path: str) -> str:
    st = os.stat(path)
    key = f"{path}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def _load_file_index() -> dict:
    if os.path.exists(FILE_TRACK):
        try:
            return json.load(open(FILE_TRACK, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_file_index(idx: dict):
    with open(FILE_TRACK, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

def _load_embed_model() -> SentenceTransformer:
    print(f"[INFO] Loading embedding model: {EMBED_MODEL} on {_device()}")
    return SentenceTransformer(EMBED_MODEL, device=_device())

def _ensure_folders():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

def _read_csv_safe(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["document_name", "chunk_id", "chunk_text"])

def _build_new_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def _append_to_index(index: faiss.IndexFlatIP, embeddings: np.ndarray):
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

def rebuild_all(pdfs_dir: str = UPLOAD_DIR) -> Tuple[pd.DataFrame, str]:
    """
    Rebuild the CSV + FAISS index from ALL PDFs in UPLOAD_DIR.
    """
    _ensure_folders()
    pdfs = [os.path.join(pdfs_dir, f) for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("[WARN] No PDFs to rebuild from.")
        # If nothing to rebuild, just keep existing files if any
        return _read_csv_safe(CHUNK_CSV), FAISS_INDEX

    df = extract_text_from_pdfs(pdfs)
    df.to_csv(CHUNK_CSV, index=False)

    model = _load_embed_model()
    emb = model.encode(df["chunk_text"].tolist(), batch_size=64, convert_to_numpy=True)
    index = _build_new_index(np.array(emb, dtype="float32"))
    faiss.write_index(index, FAISS_INDEX)

    # refresh file index
    fi = {}
    for p in pdfs:
        fi[os.path.basename(p)] = _hash_file(p)
    _save_file_index(fi)

    print(f"[OK] Rebuilt index with {len(df)} chunks.")
    return df, FAISS_INDEX

def incremental_update(new_pdf_paths: List[str]) -> Tuple[int, int]:
    """
    Incrementally process a list of new PDFs:
    - Create chunks
    - Append to CSV
    - Add vectors to FAISS
    Returns: (#new_chunks, #new_vectors_added)
    """
    if not new_pdf_paths:
        return (0, 0)

    # Load previous file index and check for already processed files
    file_idx = _load_file_index()
    to_process = []
    for p in new_pdf_paths:
        if not os.path.exists(p):
            continue
        base = os.path.basename(p)
        cur = _hash_file(p)
        if base not in file_idx or file_idx[base] != cur:
            to_process.append(p)

    if not to_process:
        print("[INFO] No new/changed PDFs to process.")
        return (0, 0)

    # Chunk new files
    df_new = extract_text_from_pdfs(to_process)
    if df_new.empty:
        print("[WARN] No text chunks found in new PDFs.")
        return (0, 0)

    # Load or init CSV
    df_existing = _read_csv_safe(CHUNK_CSV)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.to_csv(CHUNK_CSV, index=False)

    # Compute embeddings
    model = _load_embed_model()
    emb_new = model.encode(df_new["chunk_text"].tolist(), batch_size=64, convert_to_numpy=True)
    emb_new = np.array(emb_new, dtype="float32")

    # Load or create FAISS
    if os.path.exists(FAISS_INDEX):
        index = faiss.read_index(FAISS_INDEX)
        _append_to_index(index, emb_new)
    else:
        index = _build_new_index(emb_new)

    faiss.write_index(index, FAISS_INDEX)

    # Update file index
    for p in to_process:
        file_idx[os.path.basename(p)] = _hash_file(p)
    _save_file_index(file_idx)

    print(f"[OK] Appended {len(df_new)} chunks and {len(emb_new)} vectors.")
    return (len(df_new), len(emb_new))

# build_embeddings.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_embeddings(csv_path="policy_chunks.csv", index_path="policy_index.faiss"):
    """
    Builds FAISS vector index from all policy text chunks.
    Uses the same embedding model as inference to ensure dimension match.
    """
    print("🔍 Loading policy chunks from CSV...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} text chunks.")

    # ✅ Use the same model as raginference.py
    MODEL_NAME = "all-mpnet-base-v2"
    print(f"🔧 Using embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode text chunks
    print("⚙️ Encoding text chunks into embeddings...")
    embeddings = model.encode(df["chunk_text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    d = embeddings.shape[1]
    print(f"📏 Embedding dimension: {d}")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index with {len(embeddings)} vectors to {index_path}")

if __name__ == "__main__":
    build_embeddings()

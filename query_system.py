# query_system.py
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

# ================================
# Step 1: Load data and model
# ================================
def load_index_and_data(index_path="policy_index.faiss", csv_path="policy_chunks.csv"):
    print("🔍 Loading FAISS index and chunk data...")
    index = faiss.read_index(index_path)
    df = pd.read_csv(csv_path)
    return df, index


# ================================
# Step 2: Search function
# ================================
def search_query(query, df, index, model, top_k=5):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        row = df.iloc[idx]
        results.append({
            "document_name": row["document_name"],
            "chunk_text": row["chunk_text"],
            "similarity_score": float(score)
        })
    return results


# ================================
# Step 3: Simple claim decision logic
# ================================
def claim_decision(query, retrieved_clauses):
    text_combined = " ".join([c["chunk_text"].lower() for c in retrieved_clauses])

    exclusion_terms = ["not covered", "excluded", "not payable", "rejected", "not included"]
    approval_terms = ["covered", "included", "payable", "approved"]

    if any(term in text_combined for term in exclusion_terms):
        decision = "❌ Rejected"
        reason = "Found exclusion terms in retrieved clauses."
    elif any(term in text_combined for term in approval_terms):
        decision = "✅ Approved"
        reason = "Found coverage terms in retrieved clauses."
    else:
        decision = "⚠️ Inconclusive"
        reason = "Could not find clear coverage information."

    return {"decision": decision, "reason": reason}


# ================================
# Step 4: Main interactive loop
# ================================
def main():
    df, index = load_index_and_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = input("\nEnter your insurance claim question: ")
    top_results = search_query(query, df, index, model)

    # Count which PDFs were most referenced
    pdf_counts = Counter([c['document_name'] for c in top_results])
    print("\n📄 Documents involved in retrieval:")
    for doc, count in pdf_counts.items():
        print(f" - {doc}: {count} chunks")

    decision_info = claim_decision(query, top_results)

    output = {
        "query": query,
        "decision": decision_info["decision"],
        "reason": decision_info["reason"],
        "retrieved_clauses": top_results
    }

    print("\n================= CLAIM DECISION =================")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

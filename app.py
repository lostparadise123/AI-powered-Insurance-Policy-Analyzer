import streamlit as st
import pandas as pd
import os

# ✅ import correct functions
from incremental_index import incremental_update
from extract_and_chunk import extract_text_from_pdfs
from raginference import run_query, reload_index_and_chunks


st.set_page_config(page_title="Policy Analyzer", layout="wide")

st.title("📄 Insurance Policy Analyzer (RAG + Fine-Tuned Model)")


# --------------------------
# ✅ PDF UPLOAD SECTION
# --------------------------
st.subheader("📥 Upload New Policy PDF")

uploaded = st.file_uploader("Upload a policy PDF", type=["pdf"])

if uploaded:
    save_path = os.path.join("uploaded_pdfs", uploaded.name)
    os.makedirs("uploaded_pdfs", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"✅ Uploaded: {uploaded.name}")

    # Extract + chunk + update FAISS index
    with st.spinner("Extracting text, chunking, and updating FAISS index..."):
        pdf_texts = extract_text_from_pdfs([save_path])
        incremental_update(pdf_texts)

    # ✅ CORRECT FUNCTION
    reload_index_and_chunks()

    st.success("✅ New policy processed and FAISS index updated successfully!")


st.markdown("---")


# --------------------------
# ✅ QUERY SECTION
# --------------------------
st.subheader("🔍 Ask a question about insurance policies")

query = st.text_input("Enter your policy query:")

if st.button("Analyze"):
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    with st.spinner("Analyzing policy documents…"):
        answer, clauses = run_query(query, top_k=10)

    df = pd.DataFrame(clauses)

    # Determine similarity column
    score_col = "similarity_score" if "similarity_score" in df.columns else (
        "similarity" if "similarity" in df.columns else None
    )

    if score_col:
        df_sorted = df.sort_values(score_col, ascending=False)
        closest_clause = df_sorted.iloc[0]["chunk_text"]
        closest_doc = df_sorted.iloc[0]["document_name"]
    else:
        closest_clause = ""
        closest_doc = "Unknown"


    # ✅ Enhanced Human-style Final Answer
    enriched_answer = f"""
Covered under: {closest_doc}

### 🔎 Most Relevant Clause
\"\"\"\n{closest_clause.strip()}\n\"\"\"

### 🧠 Refined Interpretation  
{answer}
"""

    st.subheader("💬 Final Answer")
    st.markdown(enriched_answer)

    # Show retrieved clauses table
    st.subheader("📚 Supporting Clauses")
    if score_col:
        df[score_col] = df[score_col].astype(float).round(3)
    st.dataframe(df, use_container_width=True)

    # Show documents involved
    st.subheader("📄 Documents Involved")
    docs = sorted(list({c.get("document_name", "unknown") for c in clauses}))
    st.write(", ".join(docs))

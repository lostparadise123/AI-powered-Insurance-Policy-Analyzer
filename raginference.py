# raginference.py
import torch
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# ============================================================
# ✅ 1. Load Embedding Model (GPU if available)
# ============================================================
EMBED_MODEL_NAME = "all-mpnet-base-v2"
embed_model = SentenceTransformer(
    EMBED_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# ✅ 2. Global Variables
# ============================================================
_index = None
_chunks_df = None

def reload_index_and_chunks(index_path="policy_index.faiss", csv_path="policy_chunks.csv"):
    """Reloads FAISS index + chunk metadata."""
    global _index, _chunks_df
    _index = faiss.read_index(index_path)
    _chunks_df = pd.read_csv(csv_path)
    print("✅ Index Reloaded:", len(_chunks_df), "chunks")
    return _index, _chunks_df

# ============================================================
# ✅ 3. Load Fine-Tuned LLM (Phi-3.5 + LoRA adapter)
# ============================================================
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_PATH = "./finetuned_phi3_final"

print("✅ Loading tokenizer & base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("✅ Merging LoRA weights into base model...")
model = model.merge_and_unload()

print("✅ Building GPU text-generation pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=350,
    temperature=0.2,
    do_sample=False,
    device_map="auto"
)

# ============================================================
# ✅ 4. RAG Query Function
# ============================================================
def run_query(query, top_k=20):
    """
    Retrieve most relevant chunks + generate answer.
    Returns: (final_answer, list_of_retrieved_chunks)
    """

    index, df = reload_index_and_chunks()

    # ----------------------------
    # ✅ Encode query → vector
    # ----------------------------
    query_vec = embed_model.encode([query])
    faiss.normalize_L2(query_vec)
    scores, idx = index.search(np.array(query_vec, dtype="float32"), k=top_k)

    # ----------------------------
    # ✅ Prepare retrieved clauses
    # ----------------------------
    retrieved = df.iloc[idx[0]].copy()
    retrieved["similarity_score"] = scores[0]

    # Combine all retrieved text
    source_docs = "\n\n".join([
        f"[{row.document_name}] {row.chunk_text}"
        for _, row in retrieved.iterrows()
    ])

    # ============================
    # ✅ Final Answer Prompt
    # ============================
    prompt = f"""
You are a senior insurance claim analyst.
Answer strictly using ONLY the evidence provided.

If coverage is confirmed → start with: Covered
If excluded → start with: Not Covered
If unclear → start with: Not Clearly Mentioned

Write 3–4 clear sentences.
Do NOT add emojis.
Do NOT invent anything.
Do NOT mention instructions or rules.

Evidence:
{source_docs}

Question: {query}

Final Answer:
"""

    # ----------------------------
    # ✅ Generate answer using LLM
    # ----------------------------
    output = pipe(prompt)[0]["generated_text"]
    answer = output.replace(prompt, "").strip()

    return answer, retrieved.to_dict(orient="records")

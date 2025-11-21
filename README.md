# AI‑Powered Insurance Policy Analyzer

*RAG + Fine‑Tuned LLM + FAISS + Sentence Transformers*

An AI assistant that reads insurance policy PDFs, retrieves clause-level evidence with FAISS, and answers questions with a fine‑tuned Phi‑3.5 model via Retrieval‑Augmented Generation (RAG).

---

## 🚀 Features

* *Policy PDF ingestion* — Extract text with PyMuPDF, chunk long documents, save metadata.
* *Embeddings & FAISS* — Use all-mpnet-base-v2 to embed chunks and store them in a FAISS inner‑product index. Supports incremental updates.
* *RAG retrieval* — Retrieve top‑k relevant chunks by similarity and feed them into the LLM prompt so answers are clause‑grounded.
* *Fine‑tuned LLM* — Phi‑3.5 Mini fine‑tuned with LoRA + QLoRA (4‑bit) for insurance Q&A to reduce hallucinations.
* *Streamlit UI* — Query documents and display: final answer, most relevant clause, supporting clauses, and documents involved.

---

## 🏗 Architecture


PDF Documents
    │
    ├─ Text extraction (PyMuPDF)
    │
    └─ Chunking + Embedding (all-mpnet-base-v2)
            └─ FAISS vector index
                    └─ Top‑K clause retrieval
                            └─ LLM (Phi‑3.5) + RAG prompting
                                    └─ Final human‑readable explanation



---
## Dataflow diagram
<img width="914" height="573" alt="Screenshot 2025-11-21 225422" src="https://github.com/user-attachments/assets/464f89f0-986b-4170-be8b-e495c8eb0bad" />


## 📊 Model evaluation

| Metric             | Score | Note                              |
| ------------------ | ----: | --------------------------------- |
| Precision          |  0.88 | Mostly correct & relevant answers |
| Recall             |  0.85 | Retrieves most important info     |
| F1‑Score           |  0.84 | Balanced performance              |
| Retrieval Accuracy |  1.00 | Relevant clauses retrieved        |

---

## ✔ Output format

Each query returns:

* *Final Answer* — concise, human‑readable explanation grounded in policy text.
* *Most Relevant Clause* — top exact clause extracted.
* *Supporting Clauses* — table of retrieved chunks + similarity scores.
* *Documents Involved* — list of policy PDFs used for the response.

---

## 🧠 Tech stack

* *Extraction:* PyMuPDF
* *Embeddings:* SentenceTransformers (all-mpnet-base-v2)
* *Indexing:* FAISS
* *LLM:* Phi‑3.5 Mini (LoRA / QLoRA fine‑tuned)
* *UI:* Streamlit
* *Data:* Pandas, NumPy
* *DL:* PyTorch, Transformers

---

## 💡 Future enhancements 

* OCR for scanned policies
* multilingual papers reading
* GPT‑based clause summarization
* Cloud deployment (Azure/AWS) for scale

---
## Streamlit UI
<img width="1919" height="982" alt="Screenshot 2025-11-21 125815" src="https://github.com/user-attachments/assets/473c4a27-e401-4a06-b4ab-5e007fbb46f3" />

## Insurance policy analysis
<img width="1911" height="970" alt="image" src="https://github.com/user-attachments/assets/071b1f94-cafe-43db-a247-41867d6d57f2" />

## Supporting Clauses 
<img width="1913" height="744" alt="image" src="https://github.com/user-attachments/assets/920c22f5-6603-4df2-afd7-18d978731de2" />




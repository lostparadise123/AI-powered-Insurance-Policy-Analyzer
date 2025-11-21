import pandas as pd
import json

# Load your chunk CSV
df = pd.read_csv("policy_chunks.csv")

training_data = []

# Convert each chunk into an instruction dataset
for _, row in df.iterrows():
    training_data.append({
        "instruction": "Explain the meaning of this insurance policy clause in simple terms.",
        "input": row["chunk_text"],
        "output": f"This clause is from {row['document_name']}. It describes: "
    })

# Save training data
with open("policy_train.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("✅ Training dataset created: policy_train.json")

import json
import argparse
import torch
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# 🔧 FIX FOR PYTORCH 2.6 CHECKPOINT LOADING (CRITICAL)
# ============================================================
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.UInt32DType,    # <-- required for your error
    np.dtypes.Int64DType,
    np.dtypes.Float64DType
])
# ============================================================


# -------------------- ARGUMENT PARSER --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
opt = parser.parse_args()

# -------------------- CONFIG --------------------
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
TRAIN_FILE = "policy_train.json"
OUTPUT_DIR = "./finetuned_phi3_final"

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- LOAD DATA --------------------
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

records = []
for r in raw:
    q = r.get("question", "")
    a = r.get("answer", "")
    c = r.get("clause", "")
    d = r.get("document", "")[:1000]  # truncate long docs

    text = f"Question: {q}\nClause: {c}\nPolicy: {d}\nAnswer: {a}"
    records.append({"text": text})

dataset = Dataset.from_list(records)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]


# -------------------- QUANTIZATION CONFIG --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


# -------------------- TOKENIZER --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# -------------------- MODEL --------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

# LoRA
lora_cfg = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.15,
    task_type="CAUSAL_LM",
    bias="none",
    target_modules=["o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_cfg)

model.config.use_cache = False

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# -------------------- TOKENIZATION FUNCTION --------------------
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")


tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# -------------------- TRAINING ARGS --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-6,
    weight_decay=0.01,
    lr_scheduler_type="constant",
    warmup_steps=0,
    max_grad_norm=1.0,
    fp16=True,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)


# -------------------- TRAINER --------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# -------------------- TRAIN / RESUME --------------------
if opt.resume_from_checkpoint:
    print(f"\n🔥 Resuming training from checkpoint: {opt.resume_from_checkpoint}\n")
    trainer.train(resume_from_checkpoint=opt.resume_from_checkpoint)
else:
    trainer.train()


# -------------------- SAVE --------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training finished — model saved!\n")

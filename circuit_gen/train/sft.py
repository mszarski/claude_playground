# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.20.0",
#     "peft>=0.12.0",
#     "transformers>=4.51.0",
#     "accelerate>=0.34.0",
#     "datasets>=3.0.0",
# ]
# ///
"""
LoRA SFT of a small open model on verified circuit-design conversations.

Each training example is one complete, successful design session from
`python -m circuit_gen export` (sft_agentic.jsonl): system prompt, board
description, the teacher model's tool calls (library search, check_circuit,
simulate) with their results, and the final summary. Only sessions whose
circuit passed every check, its frozen requirements across tolerances, and
routed to DRC-clean Gerbers are included; benchmark boards are held out.

Runs on Hugging Face Jobs, e.g.:
    hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN \\
        -e DATASET=user/circuit-gen-sft -e OUTPUT_REPO=user/qwen3-8b-circuit-gen \\
        https://huggingface.co/datasets/user/circuit-gen-sft/resolve/main/sft.py
"""

import os

from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

DATASET = os.environ["DATASET"]
OUTPUT_REPO = os.environ["OUTPUT_REPO"]
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "32768"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
# Repeat the (few, varied) LLM sessions so the (many, templated) procgen ones don't dominate.
LLM_REPEAT = int(os.environ.get("LLM_REPEAT", "2"))

# Comma-separated files in the dataset repo, e.g. procgen and LLM-designed sessions.
DATA_FILES = os.environ.get("DATA_FILES", "procgen/sft_agentic.jsonl,llm/sft_agentic.jsonl")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def fits(row):
    """Drop sessions longer than MAX_LENGTH: truncation would cut off the passing check."""
    text = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    return len(tokenizer(text)["input_ids"]) <= MAX_LENGTH


# Load each file on its own: the sources carry different metadata columns.
# Split before repeating, so no copy of an eval session is trained on.
train_parts, eval_parts = [], []
for f in DATA_FILES.split(","):
    part = load_dataset(DATASET, data_files=f, split="train").select_columns(["messages", "tools"])
    kept = part.filter(fits)
    print(f"{f}: {len(kept)}/{len(part)} sessions fit in {MAX_LENGTH} tokens")
    split = kept.train_test_split(test_size=0.1, seed=0)
    train_parts += [split["train"]] * (LLM_REPEAT if f.startswith("llm/") else 1)
    eval_parts.append(split["test"])
train_dataset = concatenate_datasets(train_parts).shuffle(seed=0)
eval_dataset = concatenate_datasets(eval_parts)
print(f"train {len(train_dataset)}, eval {len(eval_dataset)}")

config = SFTConfig(
    # /data is the job's bucket mount on HF Jobs: checkpoints survive a failed push.
    output_dir="/data/circuit-gen-sft" if os.path.isdir("/data") else "circuit-gen-sft",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_private_repo=True,
    hub_strategy="every_save",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,  # a float below 1 is a ratio of total steps
    max_length=MAX_LENGTH,
    bf16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    run_name=f"sft-{BASE_MODEL.split('/')[-1]}",
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
)
trainer.train()
trainer.push_to_hub()
print(f"done: https://huggingface.co/{OUTPUT_REPO}")

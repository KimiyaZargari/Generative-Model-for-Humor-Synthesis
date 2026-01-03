import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# 1. Load JSON via pandas
# -----------------------------

df = pd.read_json("data/training-data/reddit_jokes.json")
# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)


# -----------------------------
# 2. Combine title + body into a single column
# -----------------------------
def combine_columns(example):
    text = f"### Instruction:\n{example['title']}\n\n### Response:\n{example['body']}"
    return {"text": text}


dataset = dataset.map(combine_columns)

# -----------------------------
# 3. Load tokenizer and model
# -----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# 4. LoRA config
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Move model to device
# -----------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")
model.to(device)


# -----------------------------
# 6. Tokenize dataset
# -----------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    # Labels for causal LM
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["title", "body", "text"],  # <- REMOVE all string columns!
)

# -----------------------------
# 7. Training args
# -----------------------------
training_args = TrainingArguments(
    output_dir="./llama-finetuned-reddit",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    learning_rate=2e-4,
    fp16=False,
    push_to_hub=False,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
)

# -----------------------------
# 8. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# -----------------------------
# 9. Train and save model
# -----------------------------
trainer.train()
trainer.save_model("./reddit-custom-model")

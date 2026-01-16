from src.utils.training_utils import (
    LoRATrainingConfig,
    TrainingConfig,
    setup_model_with_lora,
    create_trainer,
    train_and_save,
    print_training_info,
    get_optimal_device,
)
from src.utils.tokenizer_utils import load_tokenizer
from src.utils.data_loader import (
    load_json_dataset,
    print_dataset_info,
)
import src.utils.data_loader as data_loader

# ========================
# CONFIGURATION (Matching train-2.ipynb)
# ========================

# Base model - Meta-Llama-3-8B
MODEL_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3-8B",
    "output_dir": "models/llama-3-conan-chat",
}

# Dataset path - Conan jokes (matching train-2 line 241)
DATASET_PATH = "data/training-data/jokes-all.json"

# Settings matching train-2.ipynb exactly
MAX_LENGTH = 512  # Increased for chat format
MAX_ENTRIES = 5000  # Line 324
MIN_SENTENCES = 2  # Line 335
MAX_SENTENCES = 4  # Line 335
EXCLUDE_IDS = {88572, 99457, 99483}  # Line 225

# System prompt (matching train-2 line 325)
SYSTEM_PROMPT = (
    "You are a creative and hilarious comedy writer that loves to craft jokes"
)

# LoRA configuration for Llama-3
lora_config = LoRATrainingConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # Llama-3 attention modules
    lora_dropout=0.05,
    bias="none",
)

# Training configuration
# GPT-3.5-turbo fine-tuning typically uses:
# - 3-4 epochs
# - Small learning rate (1e-5 to 5e-5)
# - Small batch size
training_config = TrainingConfig(
    output_dir=MODEL_CONFIG["output_dir"],
    num_train_epochs=3,  # Standard for GPT-3.5-turbo fine-tuning
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,  # GPT-3.5-turbo default fine-tuning LR
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    fp16=False,
    eval_strategy="no",
    weight_decay=0.01,
)


def main():
    """Main training function - matches train-2.ipynb approach"""

    print("\n" + "=" * 60)
    print("CONAN JOKES CHAT-STYLE FINE-TUNING")
    print("(Llama-3-8B)")
    print("=" * 60)
    print(f"\nModel: {MODEL_CONFIG['model_name']}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"Max entries: {MAX_ENTRIES}")
    print(f"Sentence range: {MIN_SENTENCES}-{MAX_SENTENCES}")
    print(f"Exclude IDs: {EXCLUDE_IDS}")
    print("=" * 60)

    # 1. Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_CONFIG["model_name"])

    # Add chat-specific special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token")

    # 2. Load and prepare chat-formatted dataset
    print("\n[2/5] Loading and preparing jokes (chat format)...")
    chat_dataset = data_loader.load_and_prepare_chat_dataset(
        dataset_path=DATASET_PATH,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_length=MAX_LENGTH,
        max_entries=MAX_ENTRIES,
        min_sentences=MIN_SENTENCES,
        max_sentences=MAX_SENTENCES,
        exclude_ids=EXCLUDE_IDS,
    )

    print(f"  ✓ Prepared {len(chat_dataset)} training examples")
    print_dataset_info(chat_dataset, "Jokes Dataset")

    # 3. Setup model with LoRA
    print("\n[3/5] Setting up model with LoRA...")
    device = get_optimal_device()
    model = setup_model_with_lora(MODEL_CONFIG["model_name"], lora_config, device)

    # 4. Create trainer
    print("\n[4/5] Creating trainer...")
    print_training_info(
        MODEL_CONFIG["model_name"], len(chat_dataset), MAX_LENGTH, training_config
    )
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=chat_dataset,
        training_config=training_config,
    )

    # 5. Train and save
    print("\n[5/5] Training and saving model...")
    train_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=MODEL_CONFIG["output_dir"],
    )

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Model saved to: {MODEL_CONFIG['output_dir']}")
    print("=" * 60)

    print("\nTo test the model, use:")
    print(f"  Model: {MODEL_CONFIG['output_dir']}")
    print(f"  Format: Chat with system prompt")
    print(f"  System: {SYSTEM_PROMPT}")


if __name__ == "__main__":
    main()

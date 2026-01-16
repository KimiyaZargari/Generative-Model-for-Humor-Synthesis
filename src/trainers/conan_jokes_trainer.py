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
    prepare_dataset_completion_style,
    print_dataset_info,
)

# ========================
# CONFIGURATION
# ========================

# Single base model (GPT-2 as Curie alternative)
MODEL_CONFIG = {
    "model_name": "gpt2",  # or "gpt2-medium" for closer to Curie size
    "output_dir": "models/gpt2-conan-jokes",
}

# Single dataset (Conan jokes)
DATASET_PATH = "data/training-data/jokes-all.json"

# Match ipynb settings
MAX_LENGTH = 256  # Accommodate prompt + completion
DATA_LIMIT = 9000  # Match MAX_ENTRIES from ipynb
MIN_SENTENCES = 2  # Only use jokes with 2+ sentences

# LoRA configuration (lighter for GPT-2)
lora_config = LoRATrainingConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
    lora_dropout=0.05,
    bias="none",
)

# Training configuration matching ipynb
training_config = TrainingConfig(
    output_dir=MODEL_CONFIG["output_dir"],
    num_train_epochs=2,  # Match n_epochs from ipynb
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,  # Match learning_rate_multiplier * base_lr
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    fp16=False,
    eval_strategy="no",
)


def main():
    """Main training function - matches ipynb approach"""

    print("\n" + "=" * 60)
    print("CONAN JOKES FINE-TUNING (matching ipynb)")
    print("=" * 60)
    print(f"\nModel: {MODEL_CONFIG['model_name']}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max entries: {DATA_LIMIT}")
    print(f"Min sentences per joke: {MIN_SENTENCES}")
    print("=" * 60)

    # 1. Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_CONFIG["model_name"])

    # Add special tokens for prompt/completion format
    special_tokens = {"pad_token": "[PAD]", "sep_token": "###", "eos_token": " END"}
    tokenizer.add_special_tokens(special_tokens)

    # 2. Load dataset with completion-style formatting
    print("\n[2/5] Loading dataset (completion-style)...")
    dataset = load_json_dataset(DATASET_PATH, limit=DATA_LIMIT)

    # Filter for jokes with 2+ sentences (matching ipynb logic)
    print(f"  Filtering for {MIN_SENTENCES}+ sentence jokes...")
    tokenized_dataset = prepare_dataset_completion_style(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        min_sentences=MIN_SENTENCES,
    )

    print(f"  ✓ Prepared {len(tokenized_dataset)} training examples")

    # 3. Setup model with LoRA
    print("\n[3/5] Setting up model with LoRA...")
    device = get_optimal_device()
    model = setup_model_with_lora(
        MODEL_CONFIG["model_name"],
        lora_config,
        device,
        resize_embeddings=len(tokenizer),  # Resize for new special tokens
    )

    # 4. Create trainer
    print("\n[4/5] Creating trainer...")
    print_training_info(
        MODEL_CONFIG["model_name"], len(tokenized_dataset), MAX_LENGTH, training_config
    )
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
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


if __name__ == "__main__":
    main()

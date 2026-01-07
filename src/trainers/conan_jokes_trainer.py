"""
Trainer for Conan jokes dataset using body-only format (no instruction format)
"""

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
    prepare_dataset,
    create_text_formatter,
    print_dataset_info,
)


# ========================
# CONFIGURATION
# ========================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/training-data/conan_jokes.json"
OUTPUT_DIR = "models/conan-jokes"

# Dataset settings
MAX_LENGTH = 128
DATA_LIMIT = None  # Use full dataset

# LoRA configuration
lora_config = LoRATrainingConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Training configuration
training_config = TrainingConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,  # Fewer warmup steps for body-only format
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    fp16=False,
    eval_strategy="no",
)


def main():
    """Main training function"""

    # 1. Load tokenizer
    print("=" * 60)
    print("CONAN JOKES TRAINER - Body-Only Format")
    print("=" * 60)
    print("\n[1/6] Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_NAME)

    # 2. Load and prepare dataset
    print(f"\n[2/6] Loading dataset from {DATA_PATH}...")
    dataset = load_json_dataset(DATA_PATH, limit=DATA_LIMIT)

    print("\n[3/6] Preparing dataset...")
    # Use body-only format (no instruction format)
    text_formatter = create_text_formatter("body_only")

    tokenized_dataset = prepare_dataset(
        dataset=dataset,
        text_formatter=text_formatter,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print_dataset_info(dataset, num_examples=2)

    # 3. Setup model with LoRA
    print("\n[4/6] Setting up model with LoRA...")
    device = get_optimal_device()
    model = setup_model_with_lora(MODEL_NAME, lora_config, device)

    # 4. Create trainer
    print("\n[5/6] Creating trainer...")
    print_training_info(MODEL_NAME, len(tokenized_dataset), MAX_LENGTH, training_config)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        training_config=training_config,
    )

    # 5. Train and save
    print("[6/6] Training and saving model...")
    train_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

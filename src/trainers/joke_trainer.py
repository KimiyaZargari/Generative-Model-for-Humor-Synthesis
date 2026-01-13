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
from datasets import concatenate_datasets

# ========================
# CONFIGURATION
# ========================

# Define three base models
MODEL_CONFIGS = [
    {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "output_dir": "models/Llama-2-7b-hf-combined",
    },
    {
        "model_name": "deepseek-ai/deepseek-llm-7b-base",
        "output_dir": "models/deepseek-llm-7b-base-combined",
    },
    {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "output_dir": "models/Meta-Llama-3-8B-combined",
    },
]

# Define three datasets
DATASET_PATHS = [
    "data/training-data/conan_jokes.json",
    "data/training-data/reddit_jokes_modified.json",
    "data/training-data/general_jokes.json",
]

# Shared settings
MAX_LENGTH = 128
DATA_LIMIT = None  # Use full dataset
TEXT_FORMAT = "body_only"

# LoRA configuration (same for all models)
lora_config = LoRATrainingConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # ✓ All attention modules
    lora_dropout=0.05,
    bias="none",
)

# Training configuration (same for all models)
training_config = TrainingConfig(
    output_dir="",  # Will be set per model
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    fp16=False,
    eval_strategy="no",
)


def load_and_combine_datasets(dataset_paths, tokenizer):
    """Load all datasets and combine them"""

    print(f"\nLoading {len(dataset_paths)} datasets...")

    tokenized_datasets = []
    text_formatter = create_text_formatter(TEXT_FORMAT)

    for idx, data_path in enumerate(dataset_paths, 1):
        print(f"\n  [{idx}/{len(dataset_paths)}] Loading {data_path}...")

        # Load dataset
        dataset = load_json_dataset(data_path, limit=DATA_LIMIT)
        print(f"      Loaded {len(dataset)} examples")

        # Tokenize dataset
        tokenized_dataset = prepare_dataset(
            dataset=dataset,
            text_formatter=text_formatter,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )

        tokenized_datasets.append(tokenized_dataset)

    # Combine all datasets
    print("\n  Combining datasets...")
    combined_dataset = concatenate_datasets(tokenized_datasets)
    print(f"  ✓ Combined dataset: {len(combined_dataset)} total examples")

    return combined_dataset


def train_single_model(model_config, dataset_paths, lora_cfg, train_cfg):
    """Train a single model on all combined datasets"""

    model_name = model_config["model_name"]
    output_dir = model_config["output_dir"]

    print("=" * 60)
    print(f"TRAINING: {model_name}")
    print(f"DATASETS: {len(dataset_paths)} datasets combined")
    print("=" * 60)

    # 1. Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)

    # 2. Load and combine all datasets
    print("\n[2/6] Loading and combining datasets...")
    combined_dataset = load_and_combine_datasets(dataset_paths, tokenizer)

    # 3. Setup model with LoRA
    print("\n[3/6] Setting up model with LoRA...")
    device = get_optimal_device()
    model = setup_model_with_lora(model_name, lora_cfg, device)

    # 4. Create trainer
    print("\n[4/6] Creating trainer...")
    train_cfg.output_dir = output_dir
    print_training_info(model_name, len(combined_dataset), MAX_LENGTH, train_cfg)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        training_config=train_cfg,
    )

    # 5. Train and save
    print("[5/6] Training and saving model...")
    train_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print(f"✓ TRAINING COMPLETE: {model_name}")
    print(f"✓ Model saved to: {output_dir}")
    print("=" * 60 + "\n\n")


def main():
    """Main training function - trains all models on all datasets"""

    print("\n" + "=" * 60)
    print("MULTI-MODEL TRAINING PIPELINE")
    print(f"Training {len(MODEL_CONFIGS)} models")
    print(f"Each on {len(DATASET_PATHS)} combined datasets")
    print("=" * 60)

    print("\nDatasets to be used:")
    for idx, path in enumerate(DATASET_PATHS, 1):
        print(f"  {idx}. {path}")
    print()

    for idx, model_config in enumerate(MODEL_CONFIGS, 1):
        print(f"\n{'#' * 60}")
        print(f"MODEL {idx}/{len(MODEL_CONFIGS)}")
        print(f"{'#' * 60}\n")

        try:
            train_single_model(
                model_config, DATASET_PATHS, lora_config, training_config
            )
        except Exception as e:
            print(f"\n✗ ERROR training {model_config['model_name']}: {e}")
            print("Continuing to next model...\n")
            continue

    print("\n" + "=" * 60)
    print("✓ ALL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nTrained models:")
    for config in MODEL_CONFIGS:
        print(f"  • {config['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

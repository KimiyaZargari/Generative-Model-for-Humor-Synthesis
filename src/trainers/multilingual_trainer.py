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
from src.utils.data_loader import print_dataset_info
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import login
import random

# Login with your HuggingFace token
login(token="llama-nlp")

# ========================
# CONFIGURATION - Multilingual (EN + ES + ZH)
# ========================
# Base model - Qwen2.5-3B (Small, fast, great for Chinese)
MODEL_CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B",
    "output_dir": "models/qwen2.5-3b-jokes-multilingual",
}

# LoRA config (same as before)
lora_config = LoRATrainingConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Training config - can increase batch size with smaller model
training_config = TrainingConfig(
    output_dir=MODEL_CONFIG["output_dir"],
    num_train_epochs=3,
    per_device_train_batch_size=8,  # ← Increased from 4
    gradient_accumulation_steps=1,  # ← Reduced from 2
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    fp16=False,  # Can enable on GPU for even more speed
    eval_strategy="no",
    weight_decay=0.01,
)

# Dataset configurations
DATASET_CONFIGS = {
    "english": {
        "type": "local",
        "path": "data/training-data/jokes-all.json",
        "text_field": "body",
        "exclude_ids": {88572, 99457, 99483},
    },
    "spanish": {
        "type": "huggingface",
        "path": "mrm8488/CHISTES_spanish_jokes",
        "text_field": "joke",  # Check the actual field name in the dataset
    },
    "chinese": {
        "type": "huggingface",
        "path": "notsobad9527/chinese-joke",
        "text_field": "text",  # Check the actual field name in the dataset
    },
}

# Settings
MAX_LENGTH = 512
MAX_ENTRIES_PER_LANG = 5000  # Per language
MIN_SENTENCES = 2
MAX_SENTENCES = 4

# System prompts (multilingual)
SYSTEM_PROMPTS = {
    "english": "You are a creative and hilarious comedy writer that loves to craft jokes",
    "spanish": "Eres un escritor de comedia creativo e hilarante que ama crear chistes",
    "chinese": "你是一位富有创意且幽默风趣的喜剧作家，热爱创作笑话",
}


# ========================
# DATA LOADING FUNCTIONS
# ========================


def load_local_jokes(path, text_field="body", exclude_ids=None):
    """Load jokes from local JSON file"""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if exclude_ids:
        jokes = [d[text_field] for d in data if d.get("id") not in exclude_ids]
    else:
        jokes = [d[text_field] for d in data if text_field in d]

    return jokes


def load_huggingface_jokes(dataset_path, text_field="text"):
    """Load jokes from HuggingFace dataset"""
    dataset = load_dataset(dataset_path, split="train")

    # Extract jokes
    if text_field in dataset.column_names:
        jokes = [item[text_field] for item in dataset]
    else:
        # Try to find the right field
        print(f"  Available fields: {dataset.column_names}")
        # Use the first text-like field
        text_field = dataset.column_names[0]
        jokes = [item[text_field] for item in dataset]

    return jokes


def clean_and_split_joke(text, nlp):
    """Clean joke text and split into sentences"""
    import unicodedata

    # Skip jokes mentioning "Conan" for English dataset
    if "Conan" in text:
        return None

    # Clean text
    cleaned = (
        unicodedata.normalize("NFKD", text)
        .replace("  ", " ")
        .replace("—", "--")
        .strip()
    )

    if not cleaned:
        return None

    # Split into sentences
    doc = nlp(cleaned)
    sentences = [sent.text for sent in doc.sents]

    return {"text": cleaned, "sentences": sentences, "sentence_ct": len(sentences)}


def prepare_chat_dataset_multilingual(
    jokes,
    tokenizer,
    system_prompt,
    max_length=512,
    max_entries=5000,
    min_sentences=2,
    max_sentences=4,
    language="english",
):
    """
    Prepare jokes in chat format

    Args:
        jokes: List of joke texts
        tokenizer: Tokenizer
        system_prompt: System prompt for this language
        max_length: Max sequence length
        max_entries: Max number of jokes to process
        min_sentences: Minimum sentences per joke
        max_sentences: Maximum sentences per joke
        language: Language name for logging
    """
    import spacy
    import hashlib

    # Load appropriate spaCy model
    if language == "chinese":
        try:
            nlp = spacy.load("zh_core_web_sm")
        except:
            print(f"  Installing Chinese spaCy model...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "zh_core_web_sm"])
            nlp = spacy.load("zh_core_web_sm")
    elif language == "spanish":
        try:
            nlp = spacy.load("es_core_news_sm")
        except:
            print(f"  Installing Spanish spaCy model...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
            nlp = spacy.load("es_core_news_sm")
    else:  # English
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            print(f"  Installing English spaCy model...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

    # Shuffle jokes
    random.shuffle(jokes)

    # Clean and process
    jokes_cleaned = []
    for joke in jokes[: max_entries * 2]:  # Process more than needed
        cleaned = clean_and_split_joke(joke, nlp)
        if cleaned:
            jokes_cleaned.append(cleaned)

    print(f"    Cleaned {len(jokes_cleaned)} jokes")

    # Format for training
    processed_examples = []
    seen = set()

    for joke in jokes_cleaned:
        if len(processed_examples) >= max_entries:
            break

        # Filter by sentence count
        ct = joke["sentence_ct"]
        if ct < min_sentences or ct > max_sentences:
            continue

        # Split into setup and punchline
        setup = joke["sentences"][0]
        punchline = " ".join(joke["sentences"][1:])

        # Deduplication
        hash_id = hashlib.shake_128(punchline.encode()).hexdigest(4)
        if hash_id in seen:
            continue
        seen.add(hash_id)

        # Format in chat style
        formatted_text = f"<|system|>{system_prompt}<|user|>{setup}<|assistant|>{punchline}<|endoftext|>"

        # Tokenize
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        processed_examples.append(tokenized)

    print(f"    ✓ Created {len(processed_examples)} training examples")

    # Convert to Dataset
    return Dataset.from_dict(
        {
            key: [example[key] for example in processed_examples]
            for key in processed_examples[0].keys()
        }
    )


# ========================
# MAIN TRAINING FUNCTION
# ========================


def main():
    """Main training function - multilingual (EN + ES + ZH)"""

    print("\n" + "=" * 70)
    print("MULTILINGUAL JOKES FINE-TUNING")
    print("(English + Spanish + Chinese)")
    print("=" * 70)
    print(f"\nModel: {MODEL_CONFIG['model_name']}")
    print(f"Languages: {list(DATASET_CONFIGS.keys())}")
    print(f"Max entries per language: {MAX_ENTRIES_PER_LANG}")
    print(f"Sentence range: {MIN_SENTENCES}-{MAX_SENTENCES}")
    print("=" * 70)

    # 1. Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = load_tokenizer(MODEL_CONFIG["model_name"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token")

    # 2. Load and prepare datasets for each language
    print("\n[2/5] Loading and preparing multilingual datasets...")
    all_datasets = []

    for lang_name, config in DATASET_CONFIGS.items():
        print(f"\n  [{lang_name.upper()}]")

        try:
            # Load jokes based on type
            if config["type"] == "local":
                print(f"    Loading from local file: {config['path']}")
                jokes = load_local_jokes(
                    config["path"], config["text_field"], config.get("exclude_ids")
                )
            else:  # huggingface
                print(f"    Loading from HuggingFace: {config['path']}")
                jokes = load_huggingface_jokes(config["path"], config["text_field"])

            print(f"    Loaded {len(jokes)} raw jokes")

            # Prepare dataset
            system_prompt = SYSTEM_PROMPTS[lang_name]
            lang_dataset = prepare_chat_dataset_multilingual(
                jokes=jokes,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                max_length=MAX_LENGTH,
                max_entries=MAX_ENTRIES_PER_LANG,
                min_sentences=MIN_SENTENCES,
                max_sentences=MAX_SENTENCES,
                language=lang_name,
            )

            all_datasets.append(lang_dataset)
            print(f"    ✓ Added {len(lang_dataset)} examples")

        except Exception as e:
            print(f"    ✗ Error loading {lang_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_datasets:
        print("\n✗ No datasets loaded!")
        return

    # Combine all datasets
    print("\n  Combining all language datasets...")
    combined_dataset = concatenate_datasets(all_datasets)

    # Shuffle combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42)

    print(f"\n  ✓ Total training examples: {len(combined_dataset)}")
    print(f"  Distribution:")
    for i, (lang, dataset) in enumerate(zip(DATASET_CONFIGS.keys(), all_datasets)):
        print(f"    {lang}: {len(dataset)} examples")

    print_dataset_info(combined_dataset, "Multilingual Jokes Dataset")

    # 3. Setup model with LoRA
    print("\n[3/5] Setting up model with LoRA...")
    device = get_optimal_device()
    model = setup_model_with_lora(MODEL_CONFIG["model_name"], lora_config, device)

    # 4. Create trainer
    print("\n[4/5] Creating trainer...")
    print_training_info(
        MODEL_CONFIG["model_name"],
        len(combined_dataset),
        MAX_LENGTH,
        training_config,
    )
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        training_config=training_config,
    )

    # 5. Train and save
    print("\n[5/5] Training and saving model...")
    train_and_save(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=MODEL_CONFIG["output_dir"],
    )

    print("\n" + "=" * 70)
    print("✓ MULTILINGUAL TRAINING COMPLETE!")
    print(f"✓ Model saved to: {MODEL_CONFIG['output_dir']}")
    print("=" * 70)

    print("\nSupported languages:")
    for lang in DATASET_CONFIGS.keys():
        print(f"  • {lang}")

    print("\nSystem prompts used:")
    for lang, prompt in SYSTEM_PROMPTS.items():
        print(f"  [{lang}] {prompt}")


if __name__ == "__main__":
    main()

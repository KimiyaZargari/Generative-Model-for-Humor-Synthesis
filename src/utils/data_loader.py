import json
import hashlib
import spacy
import unicodedata
import random
from datasets import Dataset
from typing import List, Dict, Set, Optional
from collections import defaultdict

# Load spaCy for sentence splitting (matching train-2)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def load_json_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def clean_joke_text(text: str) -> Optional[str]:
    """
    Clean joke text (matching train-2 preprocessing - lines 264-271)
    Returns None if joke should be excluded
    """
    # Skip jokes mentioning "Conan"
    if "Conan" in text:
        return None

    # Normalize and clean (exact matching train-2)
    cleaned = (
        unicodedata.normalize("NFKD", text)
        .replace("  ", " ")
        .replace("—", "--")
        .strip()
    )

    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy (matching train-2 lines 274-275)"""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def create_hash_id(text: str) -> str:
    """Create hash for deduplication (matching train-2 line 345)"""
    return hashlib.shake_128(text.encode()).hexdigest(4)


def load_and_prepare_chat_dataset(
    dataset_path: str,
    tokenizer,
    system_prompt: str,
    max_length: int = 512,
    max_entries: int = 5000,
    min_sentences: int = 2,
    max_sentences: int = 4,
    exclude_ids: Set[int] = None,
) -> Dataset:
    """
    Load and prepare Conan jokes in chat format
    Exactly matching train-2.ipynb logic (lines 323-356)

    Format:
    {
        "messages": [
            {"role": "system", "content": "You are a creative..."},
            {"role": "user", "content": "setup"},
            {"role": "assistant", "content": "punchline"}
        ]
    }
    """

    if exclude_ids is None:
        exclude_ids = set()

    # Load data (matching train-2 lines 194-196, 227-229)
    print(f"  Loading from {dataset_path}...")
    data_conan = load_json_dataset(dataset_path)
    print(f"  Loaded {len(data_conan)} raw jokes from TeamCoco")

    # Extract jokes excluding duplicate IDs (matching train-2 line 227)
    jokes_conan = [d["body"] for d in data_conan if d["id"] not in exclude_ids]
    print(f"  After excluding duplicates: {len(jokes_conan)} jokes")

    # Use only Conan jokes (matching train-2 line 241)
    jokes = jokes_conan

    # Shuffle (matching train-2 line 242)
    random.shuffle(jokes)
    print(f"  Shuffled {len(jokes)} jokes")

    # Clean and process jokes (matching train-2 lines 260-282)
    counts = defaultdict(int)
    jokes_cleaned = []

    for joke in jokes:
        # Skip if mentions "Conan"
        if "Conan" in joke:
            continue

        # Clean text
        joke_clean = (
            unicodedata.normalize("NFKD", joke)
            .replace("  ", " ")
            .replace("—", "--")
            .strip()
        )

        # Split into sentences
        tokens = nlp(joke_clean)
        sentences = [sent.text for sent in tokens.sents]
        sentence_ct = len(sentences)

        # Track and store
        entry = {"text": joke_clean, "sentences": sentences, "sentence_ct": sentence_ct}
        jokes_cleaned.append(entry)
        counts[sentence_ct] += 1

    print(f"  Cleaned {len(jokes_cleaned)} jokes")
    print(f"  Sentence distribution: {dict(counts)}")

    # Format training data (matching train-2 lines 323-356)
    i = 0
    seen = set()
    processed_examples = []

    for joke in jokes_cleaned:
        # Filter by sentence count (lines 334-336)
        ct = joke["sentence_ct"]
        if ct < min_sentences or ct > max_sentences:
            continue

        # Check max entries limit (lines 338-340)
        i += 1
        if i > max_entries:
            break

        # Split into setup and punchline (lines 342-343)
        setup = joke["sentences"][0]
        punchline = " ".join(joke["sentences"][1:])

        # Deduplication by hash (lines 345-348)
        hash_id = create_hash_id(punchline)
        if hash_id in seen:
            continue

        seen.add(hash_id)

        # Create messages in chat format (lines 350-356)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": setup},
            {"role": "assistant", "content": punchline},
        ]

        # Format for GPT-2 (which doesn't have a built-in chat template)
        # We'll create a simple format: <|system|>...<|user|>...<|assistant|>...
        formatted_text = f"<|system|>{system_prompt}<|user|>{setup}<|assistant|>{punchline}<|endoftext|>"

        # Tokenize
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        processed_examples.append(tokenized)

    print(f"  ✓ Created {len(processed_examples)} training examples")
    print(f"  ✓ Filtered out {len(seen) - len(processed_examples)} duplicates")

    # Convert to HuggingFace Dataset
    return Dataset.from_dict(
        {
            key: [example[key] for example in processed_examples]
            for key in processed_examples[0].keys()
        }
    )


def print_dataset_info(dataset: Dataset, name: str = "Dataset"):
    """Print dataset information"""
    print(f"\n{name} Info:")
    print(f"  Size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"  Features: {list(dataset.features.keys())}")
        print(f"  Example input_ids length: {len(dataset[0]['input_ids'])}")
        print(f"  Example (first 100 tokens): {dataset[0]['input_ids'][:100]}")

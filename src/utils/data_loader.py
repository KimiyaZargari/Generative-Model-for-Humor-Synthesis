"""
Utility functions for loading and processing datasets
"""

import pandas as pd
from datasets import Dataset
from typing import Callable, Optional, List


def load_json_dataset(filepath: str, limit: Optional[int] = None) -> Dataset:
    """
    Load a JSON file and convert to Hugging Face Dataset

    Args:
        filepath: Path to JSON file
        limit: Optional limit on number of rows to load

    Returns:
        Hugging Face Dataset object
    """
    df = pd.read_json(filepath)

    if limit is not None:
        df = df.head(limit)

    return Dataset.from_pandas(df)


def prepare_dataset(
    dataset: Dataset,
    text_formatter: Callable,
    tokenizer,
    max_length: int = 128,
    remove_columns: Optional[List[str]] = None,
) -> Dataset:
    """
    Prepare dataset by formatting text and tokenizing

    Args:
        dataset: Input dataset
        text_formatter: Function to format text (takes example dict, returns dict with 'text' key)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        remove_columns: Columns to remove after tokenization (auto-detected if None)

    Returns:
        Processed dataset ready for training
    """
    from src.utils.tokenizer_utils import create_tokenize_function

    # Format text
    print("Formatting dataset text...")
    dataset = dataset.map(text_formatter)

    # Auto-detect columns to remove if not specified
    if remove_columns is None:
        # Keep only input_ids, attention_mask, and labels
        all_columns = dataset.column_names
        keep_columns = {"input_ids", "attention_mask", "labels"}
        remove_columns = [col for col in all_columns if col not in keep_columns]

    # Tokenize
    print("Tokenizing dataset...")
    tokenize_fn = create_tokenize_function(tokenizer, max_length)

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing",
    )

    return tokenized_dataset


def create_text_formatter(format_type: str) -> Callable:
    """
    Create a text formatting function based on format type

    Args:
        format_type: Type of formatting ('instruction', 'body_only', or 'custom')

    Returns:
        Text formatting function
    """
    from src.utils.tokenizer_utils import format_instruction_prompt, format_simple_text

    if format_type == "instruction":

        def formatter(example):
            return {
                "text": format_instruction_prompt(example["title"], example["body"])
            }

        return formatter

    elif format_type == "body_only":

        def formatter(example):
            return {"text": format_simple_text(example["body"])}

        return formatter

    else:
        raise ValueError(
            f"Unknown format_type: {format_type}. Use 'instruction' or 'body_only'"
        )


def print_dataset_info(dataset: Dataset, num_examples: int = 3) -> None:
    """
    Print dataset information and sample examples

    Args:
        dataset: Dataset to inspect
        num_examples: Number of sample examples to show
    """
    print(f"\nDataset Info:")
    print(f"  Size: {len(dataset):,} examples")
    print(f"  Features: {list(dataset.features.keys())}")

    if num_examples > 0 and "text" in dataset.features:
        print(f"\nSample examples:")
        for i in range(min(num_examples, len(dataset))):
            example_text = dataset[i]["text"]
            # Truncate if too long
            if len(example_text) > 200:
                example_text = example_text[:200] + "..."
            print(f"\n  Example {i+1}:")
            print(f"  {example_text}")

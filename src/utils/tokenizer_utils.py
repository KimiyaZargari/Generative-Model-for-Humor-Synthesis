"""
Utility functions for tokenization operations
"""
from transformers import AutoTokenizer
from typing import Dict, Any


def load_tokenizer(model_name: str, pad_token: str = None) -> AutoTokenizer:
    """
    Load tokenizer and set padding token
    
    Args:
        model_name: Name or path of the pretrained model
        pad_token: Token to use for padding (defaults to eos_token)
    
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if pad_token:
        tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_tokenize_function(tokenizer: AutoTokenizer, max_length: int = 128):
    """
    Create a tokenization function for dataset mapping
    
    Args:
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Tokenization function suitable for dataset.map()
    """
    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # For causal LM, labels are the same as input_ids
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    return tokenize_fn


def format_instruction_prompt(title: str, body: str) -> str:
    """
    Format title and body into instruction-following format
    
    Args:
        title: Instruction/title text
        body: Response/body text
    
    Returns:
        Formatted prompt string
    """
    return f"### Instruction:\n{title}\n\n### Response:\n{body}"


def format_simple_text(body: str) -> str:
    """
    Format body text only (no instruction format)
    
    Args:
        body: Text content
    
    Returns:
        Body text as-is
    """
    return body


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """
    Count number of tokens in text
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
    
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def analyze_dataset_tokens(dataset, tokenizer: AutoTokenizer, text_field: str = "text", sample_size: int = 100):
    """
    Analyze token statistics for a dataset
    
    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer to use
        text_field: Field containing text
        sample_size: Number of samples to analyze
    
    Returns:
        Dict with token statistics
    """
    sample_size = min(sample_size, len(dataset))
    token_counts = [count_tokens(dataset[i][text_field], tokenizer) for i in range(sample_size)]
    
    return {
        "mean": sum(token_counts) / len(token_counts),
        "min": min(token_counts),
        "max": max(token_counts),
        "median": sorted(token_counts)[len(token_counts) // 2],
    }
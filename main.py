"""
Main inference script for testing fine-tuned joke generation models
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Optional, Dict, Any


# ========================
# CONFIGURATION
# ========================
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Model paths - uncomment the ones you want to load
MODEL_PATHS = {
    # "reddit": "./models/sample-reddit-model",
    # "general": "./models/general-custom-model",
    # "conan": "./models/conan-model",
    "semi_merged": "./models/merged-reddit-general-jokes-model",
}

# Generation parameters
GENERATION_CONFIG = {
    "instruction": {
        "max_new_tokens": 80,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": True,
    },
    "conan": {
        "max_new_tokens": 80,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
    },
}


# ========================
# UTILITIES
# ========================
def get_device() -> torch.device:
    """Get the optimal device for inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lora_model(
    base_model_name: str, lora_path: str, device: torch.device
) -> PeftModel:
    """
    Load a LoRA fine-tuned model

    Args:
        base_model_name: Name of the base model
        lora_path: Path to the LoRA adapter
        device: Device to load model on

    Returns:
        Loaded and configured PeftModel
    """
    print(f"Loading LoRA model from {lora_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    model = PeftModel.from_pretrained(base_model, lora_path).to(device)
    model.eval()
    return model


def load_all_models(
    base_model_name: str, model_paths: Dict[str, str], device: torch.device
) -> Dict[str, PeftModel]:
    """
    Load all specified models

    Args:
        base_model_name: Name of the base model
        model_paths: Dictionary mapping model names to paths
        device: Device to load models on

    Returns:
        Dictionary of loaded models
    """
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = load_lora_model(base_model_name, path, device)
            print(f"✓ Loaded {name} model")
        except Exception as e:
            print(f"✗ Failed to load {name} model: {e}")
    return models


# ========================
# GENERATION FUNCTIONS
# ========================
def generate_instruction_format(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40,
    do_sample: bool = True,
) -> str:
    """
    Generate text using instruction-following format

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer
        prompt: The instruction/prompt
        device: Device for inference
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling

    Returns:
        Generated text (response only)
    """
    # Format as instruction
    input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and extract response
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_text.split("### Response:")[-1].strip()
    return response


def generate_direct(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
) -> str:
    """
    Generate text directly (no instruction format)

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer
        prompt: The prompt
        device: Device for inference
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling

    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_from_model(
    model_name: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    use_instruction_format: bool = True,
) -> str:
    """
    Generate text from a model with appropriate formatting

    Args:
        model_name: Name of the model (for config lookup)
        model: The model to use
        tokenizer: Tokenizer
        prompt: The prompt
        device: Device for inference
        use_instruction_format: Whether to use instruction format

    Returns:
        Generated text
    """
    # Determine which config to use
    config_key = "conan" if model_name == "conan" else "instruction"
    config = GENERATION_CONFIG[config_key]

    if use_instruction_format and model_name != "conan":
        return generate_instruction_format(model, tokenizer, prompt, device, **config)
    else:
        return generate_direct(model, tokenizer, prompt, device, **config)


# ========================
# TESTING FUNCTIONS
# ========================
def test_single_prompt(
    models: Dict[str, PeftModel],
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> None:
    """Test all models with a single prompt"""
    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt}")
    print("=" * 70)

    for model_name, model in models.items():
        print(f"\n=== {model_name.upper()} ===")
        try:
            use_instruction = model_name != "conan"
            output = generate_from_model(
                model_name, model, tokenizer, prompt, device, use_instruction
            )
            print(output)
        except Exception as e:
            print(f"Error: {e}")


def test_multiple_prompts(
    models: Dict[str, PeftModel],
    tokenizer: AutoTokenizer,
    prompts: list,
    device: torch.device,
) -> None:
    """Test all models with multiple prompts"""
    for prompt in prompts:
        test_single_prompt(models, tokenizer, prompt, device)
        print("\n")


def interactive_mode(
    models: Dict[str, PeftModel],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> None:
    """Interactive mode for testing models"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Available models:", ", ".join(models.keys()))
    print("Commands:")
    print("  - Type a prompt to test all models")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 70 + "\n")

    while True:
        prompt = input("Enter prompt (or 'quit'): ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        if not prompt:
            continue

        test_single_prompt(models, tokenizer, prompt, device)
        print("\n")


# ========================
# MAIN
# ========================
def main():
    """Main function"""
    print("=" * 70)
    print("JOKE GENERATION MODEL TESTER")
    print("=" * 70)

    # Setup
    print("\n[1/3] Setting up...")
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(BASE_MODEL_NAME)
    print(f"Loaded tokenizer: {BASE_MODEL_NAME}")

    # Load models
    print("\n[2/3] Loading models...")
    models = load_all_models(BASE_MODEL_NAME, MODEL_PATHS, device)

    if not models:
        print("No models loaded! Check your MODEL_PATHS configuration.")
        return

    print(f"\nLoaded {len(models)} model(s): {list(models.keys())}")

    # Test
    print("\n[3/3] Testing models...")

    # Single test prompt
    test_prompts = [
        "Generate a joke containing the two words: Rabbit, Apple",
        "Tell me a joke about programming",
        "Why did the chicken cross the road?",
    ]

    # Test with first prompt
    test_multiple_prompts(models, tokenizer, test_prompts, device)

    # Uncomment to test multiple prompts
    # test_multiple_prompts(models, tokenizer, test_prompts, device)

    # Uncomment for interactive mode
    # interactive_mode(models, tokenizer, device)

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

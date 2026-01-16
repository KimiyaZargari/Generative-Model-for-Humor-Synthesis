"""
Main inference script for testing fine-tuned joke generation models
Supports both GPT-2 and Llama-3 models with chat format
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from typing import Optional, Dict, Any, List


# ========================
# CONFIGURATION
# ========================

# Model configurations - add your trained models here
MODEL_CONFIGS = {
    "gpt2-conan": {
        "base_model": "gpt2",
        "lora_path": "./models/gpt2-conan-chat",
        "format": "chat",
    },
    "llama3-conan": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "lora_path": "./models/llama-3-8b-conan-chat",
        "format": "chat",
    },
    "qwen2": {
        "base_model": "Qwen/Qwen2.5-3B",
        "lora_path": "./models/qwen2.5-3b-jokes-multilingual",
        "format": "chat",
    },
    # Add more models as needed
    # "tinyllama-reddit": {
    #     "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "lora_path": "./models/sample-reddit-model",
    #     "format": "instruction",
    # },
}

# System prompt (matching train-2)
SYSTEM_PROMPT = (
    "You are a creative and hilarious comedy writer that loves to craft jokes"
)

# Generation parameters
GENERATION_CONFIG = {
    "chat": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
    },
    "instruction": {
        "max_new_tokens": 80,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "do_sample": True,
    },
}


# ========================
# UTILITIES
# ========================
def get_device() -> torch.device:
    """Get the optimal device for inference"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    return device


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer"""
    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"    Set pad_token to eos_token")
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
    print(f"  Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    print(f"  Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path).to(device)
    model.eval()
    return model


def load_all_models(
    model_configs: Dict[str, Dict[str, str]], device: torch.device
) -> Dict[str, tuple]:
    """
    Load all specified models

    Args:
        model_configs: Dictionary of model configurations
        device: Device to load models on

    Returns:
        Dictionary mapping model names to (model, tokenizer, format) tuples
    """
    models = {}
    for name, config in model_configs.items():
        try:
            print(f"\n[Loading {name}]")
            tokenizer = load_tokenizer(config["base_model"])
            model = load_lora_model(config["base_model"], config["lora_path"], device)
            models[name] = (model, tokenizer, config["format"])
            print(f"  ✓ Successfully loaded {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {name}: {e}")
    return models


# ========================
# GENERATION FUNCTIONS
# ========================
def format_chat_prompt(setup: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Format a prompt in chat style (matching train-2 format)

    Args:
        setup: The joke setup
        system_prompt: System prompt

    Returns:
        Formatted prompt string
    """
    return f"<|system|>{system_prompt}<|user|>{setup}<|assistant|>"


def format_instruction_prompt(prompt: str) -> str:
    """
    Format a prompt in instruction style

    Args:
        prompt: The instruction/prompt

    Returns:
        Formatted prompt string
    """
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def generate_chat_format(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    setup: str,
    device: torch.device,
    system_prompt: str = SYSTEM_PROMPT,
    **generation_kwargs,
) -> str:
    """
    Generate punchline using chat format (matching train-2)

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer
        setup: The joke setup (first sentence)
        device: Device for inference
        system_prompt: System prompt for the model
        **generation_kwargs: Additional generation parameters

    Returns:
        Generated punchline
    """
    # Format prompt
    prompt = format_chat_prompt(setup, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

    # Decode full output
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract punchline (everything after <|assistant|>)
    if "<|assistant|>" in full_text:
        punchline = full_text.split("<|assistant|>")[-1].strip()
        # Remove end tokens
        punchline = punchline.replace("<|endoftext|>", "").strip()
    else:
        punchline = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Try to extract just the generated part
        punchline = punchline.replace(prompt, "").strip()

    return punchline


def generate_instruction_format(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    **generation_kwargs,
) -> str:
    """
    Generate text using instruction-following format

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer
        prompt: The instruction/prompt
        device: Device for inference
        **generation_kwargs: Additional generation parameters

    Returns:
        Generated text (response only)
    """
    # Format as instruction
    input_text = format_instruction_prompt(prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

    # Decode and extract response
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_text.split("### Response:")[-1].strip()
    return response


def generate_from_model(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    format_type: str = "chat",
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Generate text from a model with appropriate formatting

    Args:
        model: The model to use
        tokenizer: Tokenizer
        prompt: The prompt
        device: Device for inference
        format_type: Format type ("chat" or "instruction")
        system_prompt: System prompt (for chat format)

    Returns:
        Generated text
    """
    # Get generation config
    config = GENERATION_CONFIG.get(format_type, GENERATION_CONFIG["chat"])

    if format_type == "chat":
        return generate_chat_format(
            model, tokenizer, prompt, device, system_prompt, **config
        )
    else:
        return generate_instruction_format(model, tokenizer, prompt, device, **config)


# ========================
# TESTING FUNCTIONS
# ========================
def test_single_prompt(
    models: Dict[str, tuple],
    prompt: str,
    num_variations: int = 3,
) -> None:
    """
    Test all models with a single prompt

    Args:
        models: Dictionary of (model, tokenizer, format) tuples
        prompt: The joke setup or prompt
        num_variations: Number of variations to generate per model
    """
    print("\n" + "=" * 70)
    print(f"SETUP: {prompt}")
    print("=" * 70)

    for model_name, (model, tokenizer, format_type) in models.items():
        print(f"\n=== {model_name.upper()} ===")
        try:
            device = next(model.parameters()).device

            for i in range(num_variations):
                output = generate_from_model(
                    model, tokenizer, prompt, device, format_type
                )
                print(f"\n  [{i+1}] {output}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


def test_multiple_prompts(
    models: Dict[str, tuple],
    prompts: List[str],
    num_variations: int = 3,
) -> None:
    """Test all models with multiple prompts"""
    for prompt in prompts:
        test_single_prompt(models, prompt, num_variations)
        print("\n")


def interactive_mode(models: Dict[str, tuple]) -> None:
    """Interactive mode for testing models"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Available models:", ", ".join(models.keys()))
    print("\nCommands:")
    print("  - Type a joke setup to generate punchlines")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 70 + "\n")

    while True:
        prompt = input("\nEnter joke setup (or 'quit'): ").strip()

        if prompt.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        if not prompt:
            continue

        test_single_prompt(models, prompt, num_variations=3)


# ========================
# MAIN
# ========================
def main():
    """Main function"""
    print("=" * 70)
    print("CONAN JOKES MODEL TESTER (Chat Format)")
    print("=" * 70)

    # Setup
    print("\n[1/3] Setting up...")
    device = get_device()

    # Load models
    print("\n[2/3] Loading models...")
    models = load_all_models(MODEL_CONFIGS, device)

    if not models:
        print("\n✗ No models loaded! Check your MODEL_CONFIGS.")
        print("Make sure the paths in MODEL_CONFIGS point to your trained models.")
        return

    print(f"\n✓ Loaded {len(models)} model(s): {list(models.keys())}")

    # Test
    print("\n[3/3] Testing models...")

    # Test prompts (joke setups)
    test_prompts = [
        "A tech billionaire walked into a bar.",
        "Scientists announced they've discovered a new planet.",
        "The robot applied for a job as a therapist.",
        "I tried to teach my dog about cryptocurrency.",
    ]

    # Run tests
    print("\n" + "=" * 70)
    print("TESTING WITH SAMPLE SETUPS")
    print("=" * 70)

    # Test with multiple prompts (3 variations each)
    test_multiple_prompts(models, test_prompts, num_variations=3)

    # Uncomment for interactive mode
    print("\n" + "=" * 70)
    print("Switching to interactive mode...")
    print("=" * 70)
    interactive_mode(models)

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

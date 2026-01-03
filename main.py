from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# -----------------------
# Device
# -----------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

# -----------------------
# Load tokenizer & base model (once)
# -----------------------
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.to(device)
base_model.eval()

# -----------------------
# Load LoRA model (reddit)
# -----------------------
reddit_model = PeftModel.from_pretrained(base_model, "./sample-reddit-model")
reddit_model.to(device)
reddit_model.eval()

# -----------------------
# Load LoRA model (general)
# -----------------------
reddit_model = PeftModel.from_pretrained(base_model, "./general-custom-model")
reddit_model.to(device)
reddit_model.eval()


# -----------------------
# Shared generation helper
# -----------------------
def _generate(model, prompt, max_length=128):
    input_text = f"### Instruction:\n{prompt}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()


# -----------------------
# Public APIs
# -----------------------
def generate_reddit_response(prompt, max_length=128):
    """Uses sample-reddit-model (LoRA adapted)"""
    return _generate(reddit_model, prompt, max_length)


def generate_general_response(prompt, max_length=128):
    """Uses general-custom-model (LoRA adapted)"""
    return _generate(base_model, prompt, max_length)


# -----------------------
# Example usage
# -----------------------
prompt = "Does removing mustache hair lead to thicker hair growth"

print("=== Reddit jokes ===")
print(generate_reddit_response(prompt))

print("\n=== General jokes ===")
print(generate_general_response(prompt))

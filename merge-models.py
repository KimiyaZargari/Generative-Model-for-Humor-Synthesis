import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

general = PeftModel.from_pretrained(base_model, "./general-custom-model")
reddit = PeftModel.from_pretrained(base_model, "./sample-reddit-model")

general_sd = general.state_dict()
reddit_sd = reddit.state_dict()

alpha = 0.5  # 0.7 = more general, 0.3 = more reddit

merged_sd = {}
for k in general_sd:
    merged_sd[k] = alpha * general_sd[k] + (1 - alpha) * reddit_sd[k]

general.load_state_dict(merged_sd)
general.save_pretrained("./merged-reddit-general-jokes-model")

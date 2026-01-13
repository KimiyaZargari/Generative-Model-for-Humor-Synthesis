import ollama

def check_model_installed(model_name: str) -> None:
    # Retrieve the list of local models
    local_models = [m['model'] for m in ollama.list()['models']]

    if model_name not in local_models:
        raise FileNotFoundError(f"The model '{model_name}' is not installed in Ollama. "
                                f"Run 'ollama pull {model_name}' to download it.")
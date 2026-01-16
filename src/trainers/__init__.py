"""
Training modules for different joke datasets
"""

from .conan_jokes_trainer import main as train_conan
from .conan_jokes_trainer_llama import main as train_conan_llama
from .multilingual_trainer import main as multilingual_trainer

# from .general_jokes_trainer import main as train_general
# from .reddit_jokes_trainer import main as train_reddit
# from .joke_trainer import main as train_model

__all__ = [
    "train_conan",
    "train_conan_llama",
    "multilingual_trainer",
    # "train_general",
    # "train_reddit",
    # "train_model",
]

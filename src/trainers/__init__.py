"""
Training modules for different joke datasets
"""

from .conan_jokes_trainer import main as train_conan
from .general_jokes_trainer import main as train_general
from .reddit_jokes_trainer import main as train_reddit

__all__ = [
    "train_conan",
    "train_general",
    "train_reddit",
]

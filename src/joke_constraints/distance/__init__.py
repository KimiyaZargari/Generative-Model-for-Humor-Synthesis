from .util import check_model_installed
from .joke_distance import distances

check_model_installed()

__all__ = [
    "check_model_installed",
    "distances"
]
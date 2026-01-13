import ollama
import numpy as np
from src.config import load_config


def cosine_distances(embeddings: np.ndarray) -> list[float]:
    E = np.asarray(embeddings, dtype=np.float32)  # (1+N, d)

    if E.ndim != 2 or E.shape[0] < 2:
        return []

    # L2-normalize each row so cosine similarity == dot product
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid divide-by-zero
    E = E / norms

    h = E[0]     # (d,)
    J = E[1:]    # (N, d)

    sims = J @ h         # (N,)
    dists = 1.0 - sims   # (N,)
    return dists.tolist()


def distances(headline: str, jokes: list[str]) -> list[float]:
    cfg = load_config()
    resp = ollama.embed(
        model= cfg.model.embeddingModel,
        input=[headline, *jokes]
    )
    embeddings = resp['embeddings']

    return cosine_distances(embeddings)



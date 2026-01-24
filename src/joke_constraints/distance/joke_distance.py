import ollama
import numpy as np
from src.config import load_config, Config
import pandas as pd

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

def get_embedding(input: str) -> list[float]:
    cfg = load_config()
    resp = ollama.embed(
        model=cfg.model.embeddingModel,
        input=input
    )
    return resp['embeddings']


def distances(headline: str, jokes: list[str]) -> list[float]:
    cfg = load_config()
    resp = ollama.embed(
        model= cfg.model.embeddingModel,
        input=[headline, *jokes]
    )
    embeddings = resp['embeddings']

    return cosine_distances(embeddings)



if __name__ == "__main__":
    # Load the file into a DataFrame
    df = pd.read_csv('../../../data/input-data/results-task-a-en.tsv', sep='\t')
    results = []
    for idx, row in df.iterrows():

        if row["headline"] != "-":
            results.append({
                "id" : row["id"],
                "embed_head": get_embedding(row["headline"]),
                "embed_fine" : get_embedding(row["output_finetuned"]),
                "embed_cmp" : get_embedding(row["output_compare"]),
                "dist_head_fine" : distances(row["headline"], [row["output_finetuned"]]),
                "dist_head_cmp" : distances(row["headline"], [row["output_compare"]]),
                "dist_fine_cmp" : distances(row["output_finetuned"], [row["output_compare"]])
            })
            print(row["id"])

    df_new = pd.DataFrame(results)
    df_new.to_csv('embed-headline-task-a-en.tsv', sep='\t', index=False)



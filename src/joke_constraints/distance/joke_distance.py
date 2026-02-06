import ollama
import numpy as np
import pandas as pd

from src.config import load_config, Config

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


def generate_embedding_tsv(
    input_tsv: str,
    output_tsv: str,
    *,
    id_col: str = "id",
    headline_col: str = "headline",
    fine_col: str = "output_finetuned",
    cmp_col: str = "output_compare",
    skip_headline_dash: bool = True,
    print_ids: bool = False,
) -> None:
    df = pd.read_csv(input_tsv, sep="\t")
    results: list[dict[str, object]] = []

    for _, row in df.iterrows():
        if skip_headline_dash and row[headline_col] == "-":
            continue

        if print_ids:
            print(row[id_col])

        results.append({
            "id": row[id_col],
            "embed_head": get_embedding(row[headline_col]),
            "embed_fine": get_embedding(row[fine_col]),
            "embed_cmp": get_embedding(row[cmp_col]),
            "dist_head_fine": distances(row[headline_col], [row[fine_col]]),
            "dist_head_cmp": distances(row[headline_col], [row[cmp_col]]),
            "dist_fine_cmp": distances(row[fine_col], [row[cmp_col]]),
        })

    df_new = pd.DataFrame(results)
    df_new.to_csv(output_tsv, sep="\t", index=False)



if __name__ == "__main__":
    generate_embedding_tsv(
        "../../../data/input-data/results-task-a-en.tsv",
        "embed-headline-task-a-en.tsv",
        print_ids=True,
    )

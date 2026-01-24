#!/usr/bin/env python3
import argparse
import ast
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def parse_embedding(cell) -> np.ndarray:
    """
    Parse an embedding stored in a TSV cell.

    Supports:
      - Python/JSON-like lists: "[0.1, 0.2, ...]"
      - Comma-separated: "0.1,0.2,0.3"
      - Space-separated: "0.1 0.2 0.3"
      - Mixed delimiters
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        raise ValueError("Empty embedding cell")

    s = str(cell).strip()

    # List-like? Try safe parsing first.
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("Embedding is not a 1D vector")
            return arr
        except (ValueError, SyntaxError):
            # fall through to numeric extraction
            pass

    # Fallback: extract numbers robustly (handles commas/spaces/extra text)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        raise ValueError(f"Could not parse embedding from: {s[:80]}...")
    return np.asarray([float(x) for x in nums], dtype=np.float64)


def load_embeddings(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N_total, D) stacked embeddings
      labels: (N_total,) which column each row came from (string)
      ids: (N_total,) original id repeated per embedding type
    """
    all_vecs = []
    all_labels = []
    all_ids = []

    # Parse and validate dimensionality
    dim = None
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

        for idx, (row_id, cell) in enumerate(zip(df["id"], df[col])):
            vec = parse_embedding(cell)
            if dim is None:
                dim = vec.shape[0]
            elif vec.shape[0] != dim:
                raise ValueError(
                    f"Embedding dimension mismatch in column '{col}' at row index {idx}: "
                    f"expected {dim}, got {vec.shape[0]}"
                )
            all_vecs.append(vec)
            all_labels.append(col)
            all_ids.append(row_id)

    X = np.vstack(all_vecs) if all_vecs else np.empty((0, 0))
    return X, np.asarray(all_labels), np.asarray(all_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Path to input .tsv")
    ap.add_argument(
        "--cols",
        default="embed_head,embed_fine,embed_cmp",
        help="Comma-separated embedding columns to use",
    )
    ap.add_argument("--sample", type=int, default=0, help="Optional cap on #points (0 = no cap)")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--perplexity-note", action="store_true", help="No-op; placeholder if you later add t-SNE")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    df = pd.read_csv(args.tsv, sep="\t", dtype={"id": str})
    if "id" not in df.columns:
        raise KeyError("TSV must have an 'id' column")

    X, labels, ids = load_embeddings(df, cols)

    # L2-normalize rows to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)

    if X.shape[0] == 0:
        raise ValueError("No embeddings loaded.")

    # Optional sampling to speed up plotting for huge files
    if args.sample and X.shape[0] > args.sample:
        rng = np.random.default_rng(args.random_seed)
        keep = rng.choice(X.shape[0], size=args.sample, replace=False)
        X, labels, ids = X[keep], labels[keep], ids[keep]

    # PCA -> 3D
    pca = PCA(n_components=3, random_state=args.random_seed)
    X3 = pca.fit_transform(X)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = list(dict.fromkeys(labels))  # stable order
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(X3[mask, 0], X3[mask, 1], X3[mask, 2], s=10, alpha=0.75, label=lab)

    ax.set_title("PCA (3 components) of embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="best")

    # Show explained variance
    evr = pca.explained_variance_ratio_
    print("Explained variance ratio:", evr, " (sum:", float(evr.sum()), ")")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# python visualize_embed.py --tsv embed-headline-task-a-en.tsv

#!/usr/bin/env python3
import argparse
import ast
import re
import inspect
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def _parse_legend_items(items: List[str]) -> dict:
    """
    Parse legend mappings like: ["embed_head=Headline", "embed_fine=Fine-tuned"]
    """
    mapping = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError("--legend items must be in the form col=Label")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            raise argparse.ArgumentTypeError("--legend items must be in the form col=Label (non-empty)")
        mapping[k] = v
    return mapping



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

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("Embedding is not a 1D vector")
            return arr
        except (ValueError, SyntaxError):
            pass

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


def _parse_pcs(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--pcs must have 2 or 3 comma-separated integers, e.g. 1,2 or 1,2,3")
    try:
        pcs = tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("--pcs must be integers, e.g. 1,2 or 1,2,3")
    if any(p < 1 for p in pcs):
        raise argparse.ArgumentTypeError("--pcs is 1-indexed; all values must be >= 1")
    if len(set(pcs)) != len(pcs):
        raise argparse.ArgumentTypeError("--pcs values must be distinct, e.g. 1,2,4")
    return pcs  # 1-indexed


def _make_tsne(n_components: int, args) -> TSNE:
    """
    Create TSNE with compatibility across scikit-learn versions where n_iter was renamed to max_iter.
    """
    sig = inspect.signature(TSNE)
    kwargs = dict(
        n_components=n_components,
        perplexity=args.tsne_perplexity,
        learning_rate=args.tsne_learning_rate,
        init=args.tsne_init,
        metric=args.tsne_metric,
        random_state=args.random_seed,
        verbose=args.tsne_verbose,
    )

    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = args.tsne_iters
    else:
        kwargs["n_iter"] = args.tsne_iters

    # Barnes-Hut only supports 2/3 dims; exact supports higher dims.
    # Here we only allow 2/3 dims anyway, so barnes_hut is fine.
    if "method" in sig.parameters:
        kwargs["method"] = args.tsne_method

    return TSNE(**kwargs)


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

    # NEW: projection method
    ap.add_argument(
        "--method",
        choices=["pca", "tsne"],
        default="pca",
        help="Projection method: pca or tsne (default: pca)",
    )

    # pcs: still selects plotted dimensions; for PCA, selects which PCs; for t-SNE, only sets 2D vs 3D
    ap.add_argument(
        "--pcs",
        type=_parse_pcs,
        default=(1, 2, 3),
        help="For PCA: which PCs to plot (1-indexed), e.g. 1,2 or 1,3,5. "
             "For t-SNE: use 2 values for 2D or 3 for 3D (indices ignored).",
    )

    ap.add_argument(
        "--legend",
        action="append",
        default=[],
        help="Legend label mapping. Repeatable: --legend embed_head=Headline --legend embed_fine=Fine",
    )

    # NEW: t-SNE knobs (kept conservative; add more as needed)
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-learning-rate", default="auto", help="Float or 'auto' (sklearn supports 'auto').")
    ap.add_argument("--tsne-iters", type=int, default=1000, help="Iterations (n_iter or max_iter depending on sklearn).")
    ap.add_argument("--tsne-init", choices=["pca", "random"], default="pca")
    ap.add_argument("--tsne-metric", default="euclidean")
    ap.add_argument("--tsne-method", choices=["barnes_hut", "exact"], default="barnes_hut")
    ap.add_argument("--tsne-verbose", type=int, default=0)

    args = ap.parse_args()
    legend_map = _parse_legend_items(args.legend)
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

    unique_labels = list(dict.fromkeys(labels))  # stable order

    pcs_1idx = args.pcs
    dims = len(pcs_1idx)

    if args.method == "pca":
        # Fit enough components to cover the max requested PC index
        max_pc = max(pcs_1idx)
        pca = PCA(n_components=max_pc, random_state=args.random_seed)
        Xk = pca.fit_transform(X)  # (N, max_pc)

        pcs_0idx = [p - 1 for p in pcs_1idx]
        Xproj = Xk[:, pcs_0idx]  # (N, 2) or (N, 3)

        evr = pca.explained_variance_ratio_
        sel_evr = np.asarray([evr[i] for i in pcs_0idx], dtype=float)

        print(f"Explained variance ratio (computed PCs 1..{max_pc}):", evr, " (sum:", float(evr.sum()), ")")
        print(f"Selected PCs {pcs_1idx} EVR:", sel_evr, " (sum selected:", float(sel_evr.sum()), ")")

        title = (
            f"PCA (PC{pcs_1idx[0]}, PC{pcs_1idx[1]}) of embeddings"
            if dims == 2
            else f"PCA (PC{pcs_1idx[0]}, PC{pcs_1idx[1]}, PC{pcs_1idx[2]}) of embeddings"
        )
        axis_labels = [f"PC{p}" for p in pcs_1idx]

    else:  # tsne
        if dims not in (2, 3):
            raise ValueError("t-SNE plotting supports only 2D or 3D (use --pcs with 2 or 3 values).")

        tsne = _make_tsne(n_components=dims, args=args)
        Xproj = tsne.fit_transform(X)  # (N, 2) or (N, 3)

        # Note: t-SNE does not have an explained variance ratio analogous to PCA.
        print(
            f"t-SNE completed: n_components={dims}, perplexity={args.tsne_perplexity}, "
            f"learning_rate={args.tsne_learning_rate}, iters={args.tsne_iters}, init={args.tsne_init}"
        )
        if args.pcs not in ((1, 2), (1, 2, 3)):
            print("Note: for t-SNE, --pcs is only used to choose 2D vs 3D; the indices themselves are ignored.")

        title = "t-SNE ({:d}D) of embeddings".format(dims)
        axis_labels = ["tSNE1", "tSNE2"] if dims == 2 else ["tSNE1", "tSNE2", "tSNE3"]

    # Plot (2D or 3D)
    if dims == 2:
        fig, ax = plt.subplots()
        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(
                Xproj[mask, 0], Xproj[mask, 1],
                s=10, alpha=0.75,
                label=legend_map.get(lab, lab),
            )
        ax.set_title(title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.legend(loc="best")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for lab in unique_labels:
            mask = labels == lab
            ax.scatter(
                Xproj[mask, 0], Xproj[mask, 1], Xproj[mask, 2],
                s=10, alpha=0.75,
                label=legend_map.get(lab, lab),
            )
        ax.set_title(title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


# PCA 2D on PC1/PC2
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv --method pca --pcs 1,2

# PCA 3D on PC1/PC3/PC5
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv --method pca --pcs 1,3,5

# t-SNE 2D
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv --method tsne --pcs 1,2 --tsne-perplexity 35

# t-SNE 3D
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv --method tsne --pcs 1,2,3 --tsne-iters 1500

# PCA with custom legend names
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv \
#  --method pca --pcs 1,2,3 \
#  --legend embed_head=Headline \
#  --legend embed_fine=Fine-tuned \
#  --legend embed_cmp=Comparison


# t-SNE 2D with custom legend names
#python visualize_embed.py --tsv embed-headline-task-a-en.tsv \
#  --method tsne --pcs 1,2 \
#  --legend embed_head=Head \
#  --legend embed_fine=Fine \
#  --legend embed_cmp=Cmp

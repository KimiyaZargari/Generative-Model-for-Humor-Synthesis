#!/usr/bin/env python3
import argparse
import ast
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors


# -------------------- Parsing --------------------

def parse_distance(cell) -> float:
    """
    Parse a cosine-distance cell that may be:
      - a float / int
      - a string float: "0.123"
      - a single-element list string: "[0.123]"
      - occasionally a list/tuple object: [0.123]
    Returns a python float.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.nan

    # If already numeric
    if isinstance(cell, (int, float, np.number)):
        return float(cell)

    s = str(cell).strip()

    # Try literal eval for list-like cases
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                return np.nan
            return float(arr[0])
        except (ValueError, SyntaxError):
            pass

    # Fallback: extract first number from string
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return np.nan
    return float(nums[0])



def parse_embedding(cell) -> np.ndarray:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        raise ValueError("Empty embedding cell")

    s = str(cell).strip()

    # Try literal list parsing first
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("Embedding is not 1D")
            return arr
        except (ValueError, SyntaxError):
            pass

    # Fallback: extract numbers
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        raise ValueError(f"Could not parse embedding from: {s[:80]}...")
    return np.asarray([float(x) for x in nums], dtype=np.float64)


def load_three_embeddings(df: pd.DataFrame,
                         col_head="embed_head",
                         col_fine="embed_fine",
                         col_cmp="embed_cmp") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, F, C = [], [], []
    dim = None

    for i, row in df.iterrows():
        h = parse_embedding(row[col_head])
        f = parse_embedding(row[col_fine])
        c = parse_embedding(row[col_cmp])

        if dim is None:
            dim = h.shape[0]
        for name, v in [("head", h), ("fine", f), ("cmp", c)]:
            if v.shape[0] != dim:
                raise ValueError(f"Dim mismatch at row {i} ({name}): expected {dim}, got {v.shape[0]}")

        H.append(h); F.append(f); C.append(c)

    return np.vstack(H), np.vstack(F), np.vstack(C)


# -------------------- Core cosine metrics --------------------

def cosine_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # assumes A,B are L2-normalized
    return np.sum(A * B, axis=1)


def heron_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # triangle area from side lengths a,b,c
    s = (a + b + c) / 2.0
    # numerical safety
    val = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    return np.sqrt(val)


def odd_one_out(d_hf: np.ndarray, d_hc: np.ndarray, d_fc: np.ndarray) -> np.ndarray:
    # Compare sums of distances from each vertex:
    # head sum = d_hf + d_hc
    # fine sum = d_hf + d_fc
    # cmp  sum = d_hc + d_fc
    head_sum = d_hf + d_hc
    fine_sum = d_hf + d_fc
    cmp_sum  = d_hc + d_fc

    m = np.vstack([head_sum, fine_sum, cmp_sum]).T
    idx = np.argmax(m, axis=1)
    labels = np.array(["head", "fine", "cmp"], dtype=object)
    return labels[idx]


# -------------------- KNN metrics --------------------

def self_retrieval_at_k(query: np.ndarray, index: np.ndarray, k: int) -> float:
    """
    For each i, does index i appear in the top-k neighbors when querying with query i?
    Uses cosine distance via sklearn (1 - cos sim) by passing normalized vectors.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(index)
    neigh = nn.kneighbors(query, return_distance=False)
    hits = (neigh == np.arange(query.shape[0])[:, None]).any(axis=1)
    return float(np.mean(hits))


def neighbor_overlap_jaccard(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """
    Mean Jaccard overlap of top-k neighbor sets between spaces A and B,
    computed within each space (A->A neighbors vs B->B neighbors).
    """
    nnA = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute").fit(A)
    nnB = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute").fit(B)

    neighA = nnA.kneighbors(A, return_distance=False)[:, 1:]  # drop self
    neighB = nnB.kneighbors(B, return_distance=False)[:, 1:]  # drop self

    jacc = []
    for i in range(A.shape[0]):
        sa = set(neighA[i].tolist())
        sb = set(neighB[i].tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc.append(inter / union if union else 0.0)
    return float(np.mean(jacc))


def hubness_k_occurrence(X: np.ndarray, k: int) -> Dict[str, float]:
    """
    Simple hubness indicator: distribution skew of how often each point appears
    in other points' top-k neighbors.
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", algorithm="brute").fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]  # drop self

    counts = np.bincount(neigh.reshape(-1), minlength=X.shape[0]).astype(np.float64)
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    p95 = float(np.percentile(counts, 95))
    p99 = float(np.percentile(counts, 99))
    maxv = float(np.max(counts))
    # “hubness index”: max / mean (rough)
    return {"mean": mean, "std": std, "p95": p95, "p99": p99, "max": maxv, "max_over_mean": (maxv / mean if mean > 0 else np.nan)}


# -------------------- Plot helpers --------------------

def save_hist(series: pd.Series, title: str, outpath: Path):
    plt.figure()
    plt.hist(series.values, bins=60)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    plt.figure()
    plt.scatter(x.values, y.values, s=8, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -------------------- Main --------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Path to input TSV")
    ap.add_argument("--outdir", default="metrics_out", help="Output directory for CSV + plots")
    ap.add_argument("--k", type=int, default=10, help="k for kNN-based metrics")
    ap.add_argument("--skip-knn", action="store_true", help="Skip kNN metrics (faster for large datasets)")
    return ap


def run(args: argparse.Namespace) -> None:

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tsv, sep="\t", dtype={"id": str})
    required_cols = ["id", "embed_head", "embed_fine", "embed_cmp",
                     "dist_head_fine", "dist_head_cmp", "dist_fine_cmp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Load embeddings and normalize for cosine
    H, F, C = load_three_embeddings(df)
    Hn = normalize(H, norm="l2")
    Fn = normalize(F, norm="l2")
    Cn = normalize(C, norm="l2")

    # Cosine sims and distances
    sim_hf = cosine_sim_rows(Hn, Fn)
    sim_hc = cosine_sim_rows(Hn, Cn)
    sim_fc = cosine_sim_rows(Fn, Cn)

    # Use TSV distances as “source of truth” for novelty/disagreement (your pipeline)
    d_hf = df["dist_head_fine"].map(parse_distance).to_numpy(dtype=np.float64)
    d_hc = df["dist_head_cmp"].map(parse_distance).to_numpy(dtype=np.float64)
    d_fc = df["dist_fine_cmp"].map(parse_distance).to_numpy(dtype=np.float64)

    # Core relevance / novelty / disagreement
    out = pd.DataFrame({
        "id": df["id"].astype(str).values,
        "sim_head_fine": sim_hf,
        "sim_head_cmp": sim_hc,
        "sim_fine_cmp": sim_fc,
        "dist_head_fine": d_hf,
        "dist_head_cmp": d_hc,
        "dist_fine_cmp": d_fc,
        "relevance_gap_fine_minus_cmp": (sim_hf - sim_hc),
        "novelty_gap_fine_minus_cmp": (d_hf - d_hc),
        "relevance_mean": (sim_hf + sim_hc) / 2.0,
        "disagreement": d_fc
    })

    # “Diverse but relevant” heuristic:
    # high disagreement *and* high mean relevance (shifted to [0,1] by (sim+1)/2 if you want)
    out["diverse_but_relevant_score"] = out["disagreement"] * out["relevance_mean"]

    # Delta (transformation) metrics: use normalized vectors for direction; raw vectors for magnitude if desired.
    # Here: magnitude in normalized space (bounded), and alignment between deltas.
    dF = Fn - Hn
    dC = Cn - Hn

    dF_norm = np.linalg.norm(dF, axis=1)
    dC_norm = np.linalg.norm(dC, axis=1)

    # delta alignment: cosine between delta vectors (add epsilon for numerical safety)
    eps = 1e-12
    delta_align = np.sum(dF * dC, axis=1) / (np.maximum(dF_norm, eps) * np.maximum(dC_norm, eps))

    out["delta_norm_fine"] = dF_norm
    out["delta_norm_cmp"] = dC_norm
    out["delta_alignment_fine_vs_cmp"] = np.clip(delta_align, -1.0, 1.0)

    # Triangle diagnostics
    out["odd_one_out"] = odd_one_out(d_hf, d_hc, d_fc)
    out["triangle_area"] = heron_area(d_hf, d_hc, d_fc)

    # Thinness ratio: area normalized by squared perimeter (scale-invariant-ish)
    perim = d_hf + d_hc + d_fc
    out["triangle_thinness"] = out["triangle_area"] / np.maximum(perim**2, eps)

    # Summary prints
    def summarize(col: str):
        s = out[col]
        return {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p10": float(np.percentile(s, 10)),
            "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
        }

    print("\nCore summaries:")
    for col in ["sim_head_fine", "sim_head_cmp", "dist_head_fine", "dist_head_cmp",
                "disagreement", "relevance_gap_fine_minus_cmp", "diverse_but_relevant_score",
                "delta_alignment_fine_vs_cmp", "triangle_area"]:
        stats = summarize(col)
        print(f"  {col:32s}  mean={stats['mean']:.4f}  std={stats['std']:.4f}  p10={stats['p10']:.4f}  p50={stats['p50']:.4f}  p90={stats['p90']:.4f}")

    print("\nOdd-one-out counts:")
    print(out["odd_one_out"].value_counts(dropna=False).to_string())

    # Optional KNN metrics
    if not args.skip_knn:
        k = args.k
        print(f"\nKNN metrics (k={k}):")
        # Cross-view self retrieval: headline queries into joke embedding spaces
        sr_fine = self_retrieval_at_k(Hn, Fn, k=k)
        sr_cmp  = self_retrieval_at_k(Hn, Cn, k=k)
        print(f"  self_retrieval@{k}  head->fine: {sr_fine:.4f}")
        print(f"  self_retrieval@{k}  head->cmp : {sr_cmp:.4f}")

        # Neighborhood overlap between joke spaces (structure stability)
        jacc = neighbor_overlap_jaccard(Fn, Cn, k=k)
        print(f"  neighbor_overlap_jaccard@{k} (fine vs cmp within-space): {jacc:.4f}")

        # Hubness indicators (if one space collapses stylistically, hubness often increases)
        hub_f = hubness_k_occurrence(Fn, k=k)
        hub_c = hubness_k_occurrence(Cn, k=k)
        print("  hubness fine:", hub_f)
        print("  hubness cmp :", hub_c)

    # Save per-id metrics
    metrics_csv = outdir / "per_id_metrics.csv"
    out.to_csv(metrics_csv, index=False)
    print(f"\nSaved per-id metrics: {metrics_csv}")

    # Plots
    save_hist(out["sim_head_fine"], "Cosine similarity: headline vs fine joke", outdir / "hist_sim_head_fine.png")
    save_hist(out["sim_head_cmp"],  "Cosine similarity: headline vs cmp joke",  outdir / "hist_sim_head_cmp.png")
    save_hist(out["disagreement"],  "Cosine distance: fine vs cmp (disagreement)", outdir / "hist_disagreement.png")
    save_hist(out["relevance_gap_fine_minus_cmp"], "Relevance gap: fine minus cmp", outdir / "hist_relevance_gap.png")

    save_scatter(
        out["relevance_mean"], out["disagreement"],
        "Disagreement vs mean relevance (diverse-but-relevant quadrant)",
        "mean relevance (cos sim)", "fine-cmp disagreement (1-cos)",
        outdir / "scatter_disagreement_vs_relevance.png"
    )

    save_scatter(
        out["dist_head_fine"], out["dist_head_cmp"],
        "Novelty comparison: fine vs cmp (headline->joke distance)",
        "dist(head,fine) = 1-cos", "dist(head,cmp) = 1-cos",
        outdir / "scatter_novelty_fine_vs_cmp.png"
    )

    print(f"Saved plots to: {outdir}")


def main():
    ap = build_parser()
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()


# python joke_embedding_metrics.py --tsv embed-headline-task-a-en.tsv --outdir metrics_out --verify-dists
# python joke_embedding_metrics.py --tsv embed-headline-task-a-en.tsv --outdir metrics_out --skip-knn

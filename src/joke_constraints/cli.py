#!/usr/bin/env python3
"""Unified command-line tool for joke_constraints utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .two_words import check_two_words, compute_word_constraints_tsv
from .joke_two_words_metrics import run_metrics, DEFAULT_TSV


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_word_constraints_input() -> str:
    return str(_repo_root() / "data" / "input-data" / "results-task-a-en.tsv")


def _default_word_constraints_output() -> str:
    return str(_repo_root() / "src" / "joke_constraints" / "word-constraint-task-a-en.tsv")


def _default_embed_input() -> str:
    return str(_repo_root() / "data" / "input-data" / "results-task-a-en.tsv")


def _default_embed_output() -> str:
    return str(_repo_root() / "src" / "joke_constraints" / "distance" / "embed-headline-task-a-en.tsv")


def _parse_pcs_proxy(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--pcs must have 2 or 3 comma-separated integers, e.g. 1,2 or 1,2,3")
    try:
        pcs = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pcs must be integers, e.g. 1,2 or 1,2,3") from exc
    if any(p < 1 for p in pcs):
        raise argparse.ArgumentTypeError("--pcs is 1-indexed; all values must be >= 1")
    if len(set(pcs)) != len(pcs):
        raise argparse.ArgumentTypeError("--pcs values must be distinct, e.g. 1,2,4")
    return pcs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="joke-constraints",
        description="Unified CLI for joke_constraints utilities",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-two-words", help="Check whether a sentence contains both words")
    p_check.add_argument("--sentence", required=True, help="Sentence to search")
    p_check.add_argument("--word1", required=True, help="First word/term to match")
    p_check.add_argument("--word2", required=True, help="Second word/term to match")

    p_wc = sub.add_parser("word-constraints-tsv", help="Compute word-constraint columns for a TSV")
    p_wc.add_argument("--input-tsv", default=_default_word_constraints_input(), help="Input TSV path")
    p_wc.add_argument("--output-tsv", default=_default_word_constraints_output(), help="Output TSV path")
    p_wc.add_argument("--id-col", default="id", help="Column name for id")
    p_wc.add_argument("--word1-col", default="word1", help="Column name for word1")
    p_wc.add_argument("--word2-col", default="word2", help="Column name for word2")
    p_wc.add_argument("--fine-col", default="output_finetuned", help="Column name for fine-tuned output")
    p_wc.add_argument("--cmp-col", default="output_compare", help="Column name for comparison output")
    p_wc.add_argument(
        "--include-word1-dash",
        action="store_true",
        help="Process rows where word1 is '-' (default: skip them)",
    )
    p_wc.add_argument(
        "--print-ids",
        action="store_true",
        help="Print each processed id to stdout",
    )

    p_wc_metrics = sub.add_parser("word-constraints-metrics", help="Summarize word-constraint TSV metrics")
    p_wc_metrics.add_argument("--tsv", default=DEFAULT_TSV, help="Input TSV path")

    p_embed = sub.add_parser("embed-tsv", help="Generate embeddings and cosine distances TSV")
    p_embed.add_argument("--input-tsv", default=_default_embed_input(), help="Input TSV path")
    p_embed.add_argument("--output-tsv", default=_default_embed_output(), help="Output TSV path")
    p_embed.add_argument("--id-col", default="id", help="Column name for id")
    p_embed.add_argument("--headline-col", default="headline", help="Column name for headline")
    p_embed.add_argument("--fine-col", default="output_finetuned", help="Column name for fine-tuned output")
    p_embed.add_argument("--cmp-col", default="output_compare", help="Column name for comparison output")
    p_embed.add_argument(
        "--include-headline-dash",
        action="store_true",
        help="Process rows where headline is '-' (default: skip them)",
    )
    p_embed.add_argument(
        "--print-ids",
        action="store_true",
        help="Print each processed id to stdout",
    )

    p_embed_metrics = sub.add_parser("embedding-metrics", help="Compute embedding-based metrics + plots")
    p_embed_metrics.add_argument("--tsv", required=True, help="Input TSV path")
    p_embed_metrics.add_argument("--outdir", default="metrics_out", help="Output directory for CSV + plots")
    p_embed_metrics.add_argument("--k", type=int, default=10, help="k for kNN-based metrics")
    p_embed_metrics.add_argument("--skip-knn", action="store_true", help="Skip kNN metrics (faster)")

    p_vis = sub.add_parser("visualize-embeddings", help="Visualize embeddings via PCA or t-SNE")
    p_vis.add_argument("--tsv", required=True, help="Input TSV path")
    p_vis.add_argument("--cols", default="embed_head,embed_fine,embed_cmp", help="Comma-separated columns")
    p_vis.add_argument("--sample", type=int, default=0, help="Optional cap on #points (0 = no cap)")
    p_vis.add_argument("--random-seed", type=int, default=0, help="Random seed for sampling")
    p_vis.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Projection method")
    p_vis.add_argument(
        "--pcs",
        type=_parse_pcs_proxy,
        default=(1, 2, 3),
        help="For PCA: PCs to plot (1-indexed). For t-SNE: choose 2 or 3 values.",
    )
    p_vis.add_argument(
        "--legend",
        action="append",
        default=[],
        help="Legend label mapping. Repeatable: --legend embed_head=Headline",
    )
    p_vis.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p_vis.add_argument("--tsne-learning-rate", default="auto", help="t-SNE learning rate (float or 'auto')")
    p_vis.add_argument("--tsne-iters", type=int, default=1000, help="t-SNE iterations")
    p_vis.add_argument("--tsne-init", choices=["pca", "random"], default="pca", help="t-SNE init")
    p_vis.add_argument("--tsne-metric", default="euclidean", help="t-SNE distance metric")
    p_vis.add_argument("--tsne-method", choices=["barnes_hut", "exact"], default="barnes_hut", help="t-SNE method")
    p_vis.add_argument("--tsne-verbose", type=int, default=0, help="t-SNE verbosity")

    p_dist = sub.add_parser("distances", help="Compute cosine distances from a headline to jokes")
    p_dist.add_argument("--headline", required=True, help="Headline text")
    p_dist.add_argument(
        "--joke",
        action="append",
        required=True,
        help="Joke text (repeatable)",
    )

    p_check_model = sub.add_parser("check-model", help="Verify the Ollama embedding model is installed")
    p_check_model.add_argument(
        "--model",
        default=None,
        help="Model name to check (default: embeddingModel from config)",
    )

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "check-two-words":
        print(check_two_words(args.sentence, args.word1, args.word2))
        return

    if args.command == "word-constraints-tsv":
        compute_word_constraints_tsv(
            args.input_tsv,
            args.output_tsv,
            id_col=args.id_col,
            word1_col=args.word1_col,
            word2_col=args.word2_col,
            fine_col=args.fine_col,
            cmp_col=args.cmp_col,
            skip_word1_dash=not args.include_word1_dash,
            print_ids=args.print_ids,
        )
        return

    if args.command == "word-constraints-metrics":
        run_metrics(args.tsv)
        return

    if args.command == "embed-tsv":
        from .distance.joke_distance import generate_embedding_tsv

        generate_embedding_tsv(
            args.input_tsv,
            args.output_tsv,
            id_col=args.id_col,
            headline_col=args.headline_col,
            fine_col=args.fine_col,
            cmp_col=args.cmp_col,
            skip_headline_dash=not args.include_headline_dash,
            print_ids=args.print_ids,
        )
        return

    if args.command == "embedding-metrics":
        from .distance import joke_embedding_metrics

        joke_embedding_metrics.run(args)
        return

    if args.command == "visualize-embeddings":
        from .distance import visualize_embed

        visualize_embed.run(args)
        return

    if args.command == "distances":
        from .distance.joke_distance import distances

        dists = distances(args.headline, args.joke)
        print(json.dumps(dists))
        return

    if args.command == "check-model":
        from .distance.util import check_model_installed

        if args.model:
            check_model_installed(args.model)
        else:
            check_model_installed()
        print("OK")
        return

    ap.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

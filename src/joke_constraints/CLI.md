# joke-constraints CLI

Unified command-line tool for `src/joke_constraints` utilities.

## Usage

```bash
python -m src.joke_constraints.cli --help
python -m src.joke_constraints.cli <subcommand> [options]
```

## Subcommands and parameters

### check-two-words
Checks whether a sentence contains both terms using the same logic as `check_two_words`.

Parameters:
- `--sentence` (required): Sentence to search.
- `--word1` (required): First word/term to match.
- `--word2` (required): Second word/term to match.

### word-constraints-tsv
Computes `word_constraint_fine` and `word_constraint_cmp` for a TSV.

Parameters:
- `--input-tsv`: Input TSV path. Default is `data/input-data/results-task-a-en.tsv` under the repo root.
- `--output-tsv`: Output TSV path. Default is `src/joke_constraints/word-constraint-task-a-en.tsv` under the repo root.
- `--id-col`: Column name for the id. Default `id`.
- `--word1-col`: Column name for word1. Default `word1`.
- `--word2-col`: Column name for word2. Default `word2`.
- `--fine-col`: Column name for fine-tuned output. Default `output_finetuned`.
- `--cmp-col`: Column name for comparison output. Default `output_compare`.
- `--include-word1-dash`: If set, processes rows where word1 is `-`. Default behavior is to skip them.
- `--print-ids`: If set, prints each processed id to stdout.

### word-constraints-metrics
Summarizes counts and joint distributions from a word-constraints TSV.

Parameters:
- `--tsv`: Input TSV path. Default `word-constraint-task-a-en.tsv`.

### embed-tsv
Generates embeddings and cosine distances for headline/fine/cmp rows.

Parameters:
- `--input-tsv`: Input TSV path. Default is `data/input-data/results-task-a-en.tsv` under the repo root.
- `--output-tsv`: Output TSV path. Default is `src/joke_constraints/distance/embed-headline-task-a-en.tsv` under the repo root.
- `--id-col`: Column name for the id. Default `id`.
- `--headline-col`: Column name for the headline. Default `headline`.
- `--fine-col`: Column name for fine-tuned output. Default `output_finetuned`.
- `--cmp-col`: Column name for comparison output. Default `output_compare`.
- `--include-headline-dash`: If set, processes rows where headline is `-`. Default behavior is to skip them.
- `--print-ids`: If set, prints each processed id to stdout.

### embedding-metrics
Computes embedding-based metrics and saves plots.

Parameters:
- `--tsv` (required): Input TSV path.
- `--outdir`: Output directory for CSV + plots. Default `metrics_out`.
- `--k`: k for kNN-based metrics. Default `10`.
- `--skip-knn`: If set, skips kNN metrics.

### visualize-embeddings
Projects embeddings using PCA or t-SNE and shows a plot.

Parameters:
- `--tsv` (required): Input TSV path.
- `--cols`: Comma-separated embedding columns to use. Default `embed_head,embed_fine,embed_cmp`.
- `--sample`: Optional cap on number of points (0 = no cap). Default `0`.
- `--random-seed`: Random seed for sampling. Default `0`.
- `--method`: Projection method, `pca` or `tsne`. Default `pca`.
- `--pcs`: For PCA, which PCs to plot (1-indexed). For t-SNE, use 2 or 3 values to choose 2D vs 3D. Default `1,2,3`.
- `--legend`: Legend label mapping. Repeatable. Example `--legend embed_head=Headline`.
- `--tsne-perplexity`: t-SNE perplexity. Default `30.0`.
- `--tsne-learning-rate`: t-SNE learning rate (float or `auto`). Default `auto`.
- `--tsne-iters`: t-SNE iterations. Default `1000`.
- `--tsne-init`: t-SNE initialization, `pca` or `random`. Default `pca`.
- `--tsne-metric`: t-SNE distance metric. Default `euclidean`.
- `--tsne-method`: t-SNE method, `barnes_hut` or `exact`. Default `barnes_hut`.
- `--tsne-verbose`: t-SNE verbosity. Default `0`.

### distances
Computes cosine distances from a headline to one or more jokes. Output is a JSON list of floats.

Parameters:
- `--headline` (required): Headline text.
- `--joke` (required, repeatable): Joke text. Provide multiple `--joke` flags for multiple jokes.

### check-model
Verifies the Ollama embedding model is installed.

Parameters:
- `--model`: Model name to check. Default is the configured `embeddingModel` from `src/config.py`.

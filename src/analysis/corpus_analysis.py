#!/usr/bin/env python
"""
Exploratory visualization for humor model datasets:
- Input datasets (TSV files inside data/input-data)
- Training joke datasets (JSON files inside data/training-data)

Improvements:
- Better chart captions with detailed titles
- Automatic saving of all charts with descriptive names
- Output directory management
"""

from __future__ import annotations

import os
import json
import collections
from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from wordcloud import WordCloud


# =============================================================================
# CONFIG
# =============================================================================

TEXT_COLUMN = "text"
TOP_N_WORDS = 30

INPUT_DATA_DIR = Path("data/input-data")
TRAINING_DATA_DIR = Path("data/training-data")
OUTPUT_DIR = Path("charts")

# spaCy models
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
}

CJK_FONT_PATH = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_PROP_ZH = None


# =============================================================================
# SETUP
# =============================================================================


def setup_output_directory():
    """Create output directory for charts if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Charts will be saved to: {OUTPUT_DIR.absolute()}")


def setup_chinese_font():
    global FONT_PROP_ZH

    if os.path.isfile(CJK_FONT_PATH):
        from matplotlib import font_manager

        FONT_PROP_ZH = font_manager.FontProperties(fname=CJK_FONT_PATH)

        try:
            family_name = FONT_PROP_ZH.get_name()
            matplotlib.rcParams["font.family"] = family_name
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
    else:
        print(
            f"[INFO] Chinese font not found. Chinese characters may not display correctly."
        )


# =============================================================================
# LOADERS
# =============================================================================


def load_tsv_files(directory: Path) -> List[pd.DataFrame]:
    """Load all TSV files in input-data directory."""
    dfs = []
    for file in directory.glob("*.tsv"):
        df = pd.read_csv(file, sep="\t")
        if "headline" not in df.columns:
            print(f"[WARN] Skipping {file.name}: no 'headline' column.")
            continue

        df = df[df["headline"].notna()]
        df = df[df["headline"].astype(str).str.strip() != "-"]
        df = df[df["headline"].astype(str).str.strip() != ""]
        df = df.rename(columns={"headline": TEXT_COLUMN})

        dfs.append((file.stem, df))

    return dfs


def load_json_files(directory: Path) -> List[pd.DataFrame]:
    """Load joke training data from JSON files. Each entry has {title, body}."""
    dfs = []
    for file in directory.glob("*.json"):
        with open(file, "r", encoding="utf8") as f:
            data = json.load(f)

        rows = []
        for item in data:
            title = item.get("title", "")
            body = item.get("body", "")
            combined = f"{title}. {body}".strip()
            rows.append({TEXT_COLUMN: combined})

        df = pd.DataFrame(rows)
        df = df[df[TEXT_COLUMN].astype(str).str.strip() != ""]
        dfs.append((file.stem, df))

    return dfs


# =============================================================================
# CORPUS BUILDING
# =============================================================================


def build_corpus(df: pd.DataFrame, nlp) -> Tuple[List[str], List[int]]:
    all_tokens = []
    lengths = []

    for text in df[TEXT_COLUMN]:
        doc = nlp(str(text))
        tokens = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
        lengths.append(len(tokens))
        all_tokens.extend(tokens)

    return all_tokens, lengths


# =============================================================================
# BASIC STATS
# =============================================================================


def print_basic_stats(name: str, tokens: List[str], lengths: List[int]):
    print(f"\n=== BASIC TEXT STATS ({name}) ===")
    print(f"Number of texts          : {len(lengths)}")
    print(f"Total word count         : {len(tokens)}")
    print(f"Vocabulary size          : {len(set(tokens))}")
    print(f"Average words per text   : {sum(lengths) / len(lengths):.2f}")


# =============================================================================
# VISUALIZATIONS WITH IMPROVED CAPTIONS AND SAVING
# =============================================================================


def plot_hist(lengths: List[int], name: str, lang: str, dataset_type: str):
    """
    Plot histogram of sentence/text lengths.

    Args:
        lengths: List of text lengths in words
        name: Dataset name for filename
        lang: Language code (en, es, zh)
        dataset_type: 'input' or 'training'
    """
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=400, color="steelblue", edgecolor="black", alpha=0.7)

    # Improved title based on dataset type
    if dataset_type == "input":
        title = f"Distribution of Constraint Text Lengths ({lang.upper()})"
        ylabel = "Number of Constraints"
    else:
        title = f"Distribution of Joke Lengths ({name})"
        ylabel = "Number of Jokes"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Number of Words per Text", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xlim(0, 800)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save with descriptive filename
    filename = f"length-distribution-{name}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {filepath}")
    plt.close()


def plot_top_words(tokens: List[str], name: str, lang: str, dataset_type: str):
    """
    Plot bar chart of top N most frequent words.

    Args:
        tokens: List of all tokens
        name: Dataset name for filename
        lang: Language code
        dataset_type: 'input' or 'training'
    """
    counter = collections.Counter(tokens)
    words, counts = zip(*counter.most_common(TOP_N_WORDS))

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(words)), counts, color="coral", edgecolor="black", alpha=0.8)
    plt.xticks(range(len(words)), words, rotation=45, ha="right", fontsize=9)

    # Improved title
    if dataset_type == "input":
        title = f"Top {TOP_N_WORDS} Most Frequent Words in Constraints ({lang.upper()})"
    else:
        title = f"Top {TOP_N_WORDS} Most Frequent Words ({name})"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=11)
    plt.xlabel("Words", fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save with descriptive filename
    filename = f"top-{TOP_N_WORDS}-{name}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {filepath}")
    plt.close()


def plot_wordcloud(tokens: List[str], name: str, lang: str):
    """
    Generate and save word cloud visualization.

    Args:
        tokens: List of all tokens
        name: Dataset name for filename
        lang: Language code
    """
    # Configure font for word cloud if Chinese
    wc_kwargs = {
        "width": 800,
        "height": 400,
        "background_color": "white",
        "colormap": "viridis",
        "relative_scaling": 0.5,
    }

    if lang == "zh" and FONT_PROP_ZH:
        wc_kwargs["font_path"] = CJK_FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(collections.Counter(tokens))

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud: {name}", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()

    # Save with descriptive filename
    filename = f"wordcloud-{name}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {filepath}")
    plt.close()


def plot_pos(counter: collections.Counter, name: str, lang: str, dataset_type: str):
    """
    Plot part-of-speech tag frequency distribution.

    Args:
        counter: Counter object with POS tags
        name: Dataset name for filename
        lang: Language code
        dataset_type: 'input' or 'training'
    """
    tags, counts = zip(*counter.most_common())

    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(tags)), counts, color="mediumpurple", edgecolor="black", alpha=0.8
    )
    plt.xticks(range(len(tags)), tags, rotation=45, ha="right", fontsize=9)

    # Improved title
    if dataset_type == "input":
        title = f"Part-of-Speech Tag Distribution in Constraints ({lang.upper()})"
    else:
        title = f"Part-of-Speech Tag Distribution ({name})"

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=11)
    plt.xlabel("POS Tags", fontsize=11)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save with descriptive filename
    filename = f"pos-freq-{name}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {filepath}")
    plt.close()


# =============================================================================
# POS TAGGING
# =============================================================================


def pos_distribution(df: pd.DataFrame, nlp) -> collections.Counter:
    counter = collections.Counter()

    for text in df[TEXT_COLUMN]:
        doc = nlp(str(text))
        for t in doc:
            if not t.is_space and not t.is_punct:
                counter[t.pos_] += 1

    return counter


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def analyze_dataset(name: str, df: pd.DataFrame, lang: str, dataset_type: str):
    """
    Perform complete analysis on a dataset.

    Args:
        name: Dataset name
        df: DataFrame containing text data
        lang: Language code (en, es, zh)
        dataset_type: 'input' or 'training'
    """
    print(f"\n===== Analyzing: {name} ({lang}) =====")

    model_name = SPACY_MODELS.get(lang, "en_core_web_sm")
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"[ERROR] spaCy model '{model_name}' not found. Please install it:")
        print(f"        python -m spacy download {model_name}")
        return

    tokens, lengths = build_corpus(df, nlp)
    print_basic_stats(name, tokens, lengths)

    # Generate all visualizations
    plot_hist(lengths, name, lang, dataset_type)
    plot_top_words(tokens, name, lang, dataset_type)
    plot_wordcloud(tokens, name, lang)

    pos_counts = pos_distribution(df, nlp)
    plot_pos(pos_counts, name, lang, dataset_type)


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================


def detect_language(df: pd.DataFrame) -> str:
    """
    Detect language from DataFrame content.

    Returns:
        Language code: 'en', 'es', or 'zh'
    """
    sample_text = " ".join(df[TEXT_COLUMN].head(50).astype(str))

    # Check for Chinese characters
    if any("\u4e00" <= ch <= "\u9fff" for ch in sample_text):
        return "zh"

    # Check for Spanish-specific characters (rough heuristic)
    spanish_chars = set("áéíóúñü¿¡")
    if any(ch in spanish_chars for ch in sample_text.lower()):
        return "es"

    return "en"


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    setup_output_directory()
    setup_chinese_font()

    print("\n=== Loading Input TSV Data ===")
    tsv_datasets = load_tsv_files(INPUT_DATA_DIR)

    print("\n=== Loading Training JSON Joke Data ===")
    json_datasets = load_json_files(TRAINING_DATA_DIR)

    # Process input datasets
    for name, df in tsv_datasets:
        lang = detect_language(df)
        analyze_dataset(name, df, lang, dataset_type="input")

    # Process training datasets
    for name, df in json_datasets:
        lang = detect_language(df)
        print(name)
        analyze_dataset(name, df, lang, dataset_type="training")

    print(f"\n[INFO] All charts saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()

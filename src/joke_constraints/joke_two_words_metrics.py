import argparse

import pandas as pd

# Default TSV path
DEFAULT_TSV = "word-constraint-task-a-en.tsv"


def run_metrics(tsv_path: str) -> None:
    # Read TSV
    df = pd.read_csv(tsv_path, sep="\t")

    # Convert "True"/"False" strings to actual booleans
    bool_cols = ["word_constraint_fine", "word_constraint_cmp"]
    for c in bool_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    n = len(df)

    # 1) Per-column counts + percentages
    print("Per-column counts + percentages:")
    for c in bool_cols:
        counts = df[c].value_counts(dropna=False)
        perc = (counts / n * 100).round(2)
        out = pd.DataFrame({"count": counts.astype(int), "percent": perc})
        # Ensure stable order: True then False (and NaN if present)
        out = out.reindex([True, False]).dropna(how="all")
        print(f"\n{c}")
        print(out.to_string())

    # 2) Joint counts + percentages across both columns
    print("\nJoint counts + percentages (fine, cmp):")
    joint = (
        df.groupby(bool_cols)
          .size()
          .rename("count")
          .reset_index()
    )
    joint["percent"] = (joint["count"] / n * 100).round(2)
    joint = joint.sort_values(["count", "word_constraint_fine", "word_constraint_cmp"], ascending=[False, True, True])
    print(joint.to_string(index=False))

    # 3) Optional: joint table (matrix) with % of total
    print("\nJoint table (counts):")
    ct_counts = pd.crosstab(df["word_constraint_fine"], df["word_constraint_cmp"])
    print(ct_counts)

    print("\nJoint table (% of total):")
    ct_pct = (ct_counts / n * 100).round(2)
    print(ct_pct)

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=DEFAULT_TSV, help="Path to input TSV")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    run_metrics(args.tsv)


if __name__ == "__main__":
    main()

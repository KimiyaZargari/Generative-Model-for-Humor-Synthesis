import re

import pandas as pd

_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")  # CJK Ext A + Unified + Compatibility

def _is_cjk_term(term: str) -> bool:
    return _CJK_RE.search(term) is not None

def _contains_term(sentence: str, term: str) -> bool:
    term = term.strip()
    if not term:
        return False

    if _is_cjk_term(term):
        # Chinese/Japanese Kanji: match as a literal substring
        return re.search(re.escape(term), sentence) is not None
    else:
        # Latin/etc: match as a whole word, case-insensitive
        pattern = rf"\b{re.escape(term)}\b"
        return re.search(pattern, sentence, flags=re.IGNORECASE) is not None

def check_two_words(sentence: str, word1: str, word2: str) -> bool:
    return _contains_term(sentence, word1) and _contains_term(sentence, word2)


def compute_word_constraints_tsv(
    input_tsv: str,
    output_tsv: str,
    *,
    id_col: str = "id",
    word1_col: str = "word1",
    word2_col: str = "word2",
    fine_col: str = "output_finetuned",
    cmp_col: str = "output_compare",
    skip_word1_dash: bool = True,
    print_ids: bool = False,
) -> None:
    df = pd.read_csv(input_tsv, sep="\t")
    results: list[dict[str, object]] = []

    for _, row in df.iterrows():
        if skip_word1_dash and row[word1_col] == "-":
            continue

        if print_ids:
            print(row[id_col])

        results.append({
            "id": row[id_col],
            "word_constraint_fine": check_two_words(row[fine_col], row[word1_col], row[word2_col]),
            "word_constraint_cmp": check_two_words(row[cmp_col], row[word1_col], row[word2_col]),
        })

    df_new = pd.DataFrame(results)
    df_new.to_csv(output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    compute_word_constraints_tsv(
        "../../data/input-data/results-task-a-en.tsv",
        "word-constraint-task-a-en.tsv",
        print_ids=True,
    )

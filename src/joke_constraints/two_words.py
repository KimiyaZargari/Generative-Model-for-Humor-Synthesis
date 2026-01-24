import re
import pandas as pd

def check_two_words(sentence: str, word1: str, word2: str) -> bool:
    pattern1 = rf"\b{re.escape(word1)}\b"
    pattern2 = rf"\b{re.escape(word2)}\b"

    return (
        re.search(pattern1, sentence, flags=re.IGNORECASE) is not None
        and re.search(pattern2, sentence, flags=re.IGNORECASE) is not None
    )

if __name__ == "__main__":
    # Load the file into a DataFrame
    df = pd.read_csv('../../data/input-data/results-task-a-en.tsv', sep='\t')
    results = []
    for idx, row in df.iterrows():

        if row["word1"] != "-":
            results.append({
                "id" : row["id"],
                "word_constraint_fine": check_two_words(row["output_finetuned"], row["word1"], row["word2"]),
                "word_constraint_cmp": check_two_words(row["output_compare"], row["word1"], row["word2"]),
            })
            print(row["id"])

    df_new = pd.DataFrame(results)
    df_new.to_csv('word-constraint-task-a-en.tsv', sep='\t', index=False)
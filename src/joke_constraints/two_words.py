import re

def check_two_words(sentence: str, word1: str, word2: str) -> bool:
    pattern1 = rf"\b{re.escape(word1)}\b"
    pattern2 = rf"\b{re.escape(word2)}\b"

    return (
        re.search(pattern1, sentence, flags=re.IGNORECASE) is not None
        and re.search(pattern2, sentence, flags=re.IGNORECASE) is not None
    )
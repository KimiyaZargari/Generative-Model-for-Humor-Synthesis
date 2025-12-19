import json
import re


def split_sentences(text):
    """Split text into sentences."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


input_file = "data/training-data/output.json"
output_file = "output.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []

for item in data:
    title = item["title"].strip()
    body = item["body"].strip()

    sentences = split_sentences(body)

    # Check conditions
    if body.startswith(title) and len(sentences) > 1:
        new_title = sentences[0]
        new_body = " ".join(sentences[1:])
    else:
        new_title = title
        new_body = body

    processed.append({"title": new_title, "body": new_body})

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=4, ensure_ascii=False)

print("Done. Cleaned JSON written to", output_file)

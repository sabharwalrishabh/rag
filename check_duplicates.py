import json
from collections import defaultdict

with open("hotpot_dev_distractor_v1.json", "r") as f:
    data = json.load(f)

# title -> list of sentence-lists seen across all datapoints
title_to_sentences = defaultdict(list)

for example in data:
    for title, sentences in example["context"]:
        title_to_sentences[title].append(sentences)

repeated_titles = {t: s for t, s in title_to_sentences.items() if len(s) > 1}
print(f"Total unique titles: {len(title_to_sentences)}")
print(f"Titles appearing in more than one datapoint: {len(repeated_titles)}")

same_count = 0
diff_count = 0
diff_examples = []

for title, all_sentence_lists in repeated_titles.items():
    first = all_sentence_lists[0]
    if all(sl == first for sl in all_sentence_lists[1:]):
        same_count += 1
    else:
        diff_count += 1
        diff_examples.append((title, all_sentence_lists))

print(f"\nOf repeated titles:")
print(f"  Always identical sentences: {same_count}")
print(f"  Different sentences across datapoints: {diff_count}")

if diff_examples:
    print(f"\n--- Example of differing sentence lists ---")
    title, variants = diff_examples[0]
    print(f"Title: {title}")
    for i, variant in enumerate(variants):
        print(f"  Variant {i+1}: {variant}")
else:
    print("\nAll repeated titles have identical sentence lists across datapoints.")

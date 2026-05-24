import json
from part1 import retrieve, dense_retrieve


with open("hotpot_dev_distractor_v1.json", "r") as f:
        data = json.load(f)

for datapoint in data[:100]:
    question = datapoint["question"]
    bm25_hits = retrieve(question, k=10)
    dense_hits = dense_retrieve(question, k=10)
    print(bm25_hits)
    print("-"*100)
    print(dense_hits)
    break

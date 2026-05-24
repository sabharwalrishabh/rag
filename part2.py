import json
from part1 import retrieve, dense_retrieve

with open("hotpot_dev_distractor_v1.json", "r") as f:
        data = json.load(f)

for datapoint in data[:100]:
    question = datapoint["question"]
    gold_labels = datapoint["supporting_facts"]
    bm25_result = retrieve(question, k=1)
    bm25_output = [(title, sentence_idx) for title, sentence_idx, text in bm25_result]
    dense_result = dense_retrieve(question, k=1)
    dense_output = [(title, sentence_idx) for title, sentence_idx, text in dense_result]
    
    

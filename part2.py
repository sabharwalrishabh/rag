import json
from part1 import retrieve, dense_retrieve

def compute_metrics(gold, retrieved):
	hits = 0
	sum_precisions = 0
	for i, item in enumerate(retrieved):
		if item in gold:
			hits += 1
			sum_precisions += hits / (i + 1)

	recall_score = hits / len(gold)
	precision_score = hits / len(retrieved)
	map_score = sum_precisions / len(gold) 

	if recall_score + precision_score == 0:
		f1_score = 0
	else:
		f1_score = 2 * recall_score * precision_score / (recall_score + precision_score)

	em_score = 1 if hits == len(gold) else 0

	
	return recall_score, precision_score, f1_score, em_score, map_score


with open("hotpot_dev_distractor_v1.json", "r") as f:
	data = json.load(f)

NUM_DATA = 100
k_values = [1, 5, 10, 15]
bm25_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}
dense_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}

for datapoint in data[:NUM_DATA]:
	question = datapoint["question"]
	gold_labels = datapoint["supporting_facts"]

	bm25_result = retrieve(question, k=15)
	bm25_output = [[title, sentence_idx] for title, sentence_idx, text in bm25_result]

	dense_result = dense_retrieve(question, k=15)
	dense_output = [[title, sentence_idx] for title, sentence_idx, text in dense_result]

	for k in k_values:
		recall, precision, f1, em, map_score = compute_metrics(gold_labels, bm25_output[:k])
		bm25_total_score[k]["recall"] += recall
		bm25_total_score[k]["precision"] += precision
		bm25_total_score[k]["f1"] += f1
		bm25_total_score[k]["em"] += em
		bm25_total_score[k]["map_score"] += map_score

		recall, precision, f1, em, map_score = compute_metrics(gold_labels, dense_output[:k])
		dense_total_score[k]["recall"] += recall
		dense_total_score[k]["precision"] += precision
		dense_total_score[k]["f1"] += f1
		dense_total_score[k]["em"] += em
		dense_total_score[k]["map_score"] += map_score

print("BM25 results:")
for k in k_values:
	avg_recall = round(bm25_total_score[k]["recall"] / NUM_DATA, 4)
	avg_precision = round(bm25_total_score[k]["precision"] / NUM_DATA, 4)
	avg_f1 = round(bm25_total_score[k]["f1"] / NUM_DATA, 4)
	avg_em = round(bm25_total_score[k]["em"] / NUM_DATA, 4)
	avg_map = round(bm25_total_score[k]["map_score"] / NUM_DATA, 4)
	print(f"for top-{k}: Recall={avg_recall}; Precision={avg_precision}; F1={avg_f1}; EM={avg_em}; MAP={avg_map};")

print("\nDense results:")
for k in k_values:
	avg_recall = round(dense_total_score[k]["recall"] / NUM_DATA, 4)
	avg_precision = round(dense_total_score[k]["precision"] / NUM_DATA, 4)
	avg_f1 = round(dense_total_score[k]["f1"] / NUM_DATA, 4)
	avg_em = round(dense_total_score[k]["em"] / NUM_DATA, 4)
	avg_map = round(dense_total_score[k]["map_score"] / NUM_DATA, 4)
	print(f"for top-{k}: Recall={avg_recall}; Precision={avg_precision}; F1={avg_f1}; EM={avg_em}; MAP={avg_map};")

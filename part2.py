import json
from part1 import retrieve, dense_retrieve
from sentence_transformers import CrossEncoder

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


def hybrid_retrieve(question, k=15, rrf_k=60, pool_size=100):
    bm25_results = retrieve(question, k=pool_size)
    dense_results = dense_retrieve(question, k=pool_size)
    
    # Build rank dictionaries: (title, idx) -> rank
    bm25_ranks = {}
    for rank, (title, idx, text) in enumerate(bm25_results):
        bm25_ranks[(title, idx)] = rank + 1  # 1-indexed
    
    dense_ranks = {}
    for rank, (title, idx, text) in enumerate(dense_results):
        dense_ranks[(title, idx)] = rank + 1
    
    # Collect all unique sentences from both lists
    all_sentences = {}
    for title, idx, text in bm25_results + dense_results:
        all_sentences[(title, idx)] = text
    
    # Compute RRF scores
    default_rank = 10000
    rrf_scores = {}
    for key in all_sentences:
        bm25_rank = bm25_ranks.get(key, default_rank)
        dense_rank = dense_ranks.get(key, default_rank)
        rrf_scores[key] = 0.3/(bm25_rank + rrf_k) + 0.7/(dense_rank + rrf_k)
    
    # Sort by RRF score descending, take top-k
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(title, idx, all_sentences[(title, idx)]) for (title, idx), score in sorted_results[:k]]


# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = CrossEncoder('BAAI/bge-reranker-base')

def rerank_retrieve(question, k=15, pool_size=100):
    # Step 1: Use your fast bi-encoder to get a broad candidate pool
    candidates = dense_retrieve(question, k=pool_size)
    
    # Step 2: Create (question, sentence) pairs for each candidate
    pairs = [(question, f"{title}: {text}") for title, idx, text in candidates]
    
    # Step 3: Cross-encoder scores each pair — this is the slow part
    # It encodes question and sentence TOGETHER through the transformer
    # allowing full attention between query and document tokens
    scores = reranker.predict(pairs)
    
    # Step 4: Sort by cross-encoder scores and take top-k
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [candidate for candidate, score in scored[:k]]


with open("hotpot_dev_distractor_v1.json", "r") as f:
	data = json.load(f)

NUM_DATA = 100
k_values = [1, 5, 10, 15]
bm25_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}
dense_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}
bm25_only_hits = {k: 0 for k in k_values}
hybrid_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}
cross_enc_total_score = {k: {"recall": 0, "precision": 0, "f1": 0, "em": 0, "map_score": 0} for k in k_values}


for datapoint in data[:NUM_DATA]:
	question = datapoint["question"]
	# question = "Represent this sentence for searching relevant passages: " + question
	gold_labels = datapoint["supporting_facts"]

	bm25_result = retrieve(question, k=15)
	bm25_output = [[title, sentence_idx] for title, sentence_idx, text in bm25_result]

	dense_result = dense_retrieve(question, k=15)
	dense_output = [[title, sentence_idx] for title, sentence_idx, text in dense_result]

	hybrid_result = hybrid_retrieve(question, k=15)
	hybrid_output = [[title, sentence_idx] for title, sentence_idx, text in hybrid_result]

	cross_enc_result = rerank_retrieve(question, k=15)
	cross_enc_output = [[title, sentence_idx] for title, sentence_idx, text in cross_enc_result]

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

		# for item in gold_labels:
		# 	if item in bm25_output[:k] and item not in dense_output[:k]:
		# 		bm25_only_hits[k] += 1

		recall, precision, f1, em, map_score = compute_metrics(gold_labels, hybrid_output[:k])
		hybrid_total_score[k]["recall"] += recall
		hybrid_total_score[k]["precision"] += precision
		hybrid_total_score[k]["f1"] += f1
		hybrid_total_score[k]["em"] += em
		hybrid_total_score[k]["map_score"] += map_score

		recall, precision, f1, em, map_score = compute_metrics(gold_labels, cross_enc_output[:k])
		cross_enc_total_score[k]["recall"] += recall
		cross_enc_total_score[k]["precision"] += precision
		cross_enc_total_score[k]["f1"] += f1
		cross_enc_total_score[k]["em"] += em
		cross_enc_total_score[k]["map_score"] += map_score


print("Question Modified + 'Title' prepended to the embedding sentences")
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

# print("\nGold facts BM25 found but dense missed (avg per question):")
# for k in k_values:
# 	print(f"for top-{k}: {round(bm25_only_hits[k] / NUM_DATA, 4)}")

print("\n Hybrid results:")
for k in k_values:
	avg_recall = round(hybrid_total_score[k]["recall"] / NUM_DATA, 4)
	avg_precision = round(hybrid_total_score[k]["precision"] / NUM_DATA, 4)
	avg_f1 = round(hybrid_total_score[k]["f1"] / NUM_DATA, 4)
	avg_em = round(hybrid_total_score[k]["em"] / NUM_DATA, 4)
	avg_map = round(hybrid_total_score[k]["map_score"] / NUM_DATA, 4)
	print(f"for top-{k}: Recall={avg_recall}; Precision={avg_precision}; F1={avg_f1}; EM={avg_em}; MAP={avg_map};")

print("\n Cross-Encoder results:")
for k in k_values:
	avg_recall = round(cross_enc_total_score[k]["recall"] / NUM_DATA, 4)
	avg_precision = round(cross_enc_total_score[k]["precision"] / NUM_DATA, 4)
	avg_f1 = round(cross_enc_total_score[k]["f1"] / NUM_DATA, 4)
	avg_em = round(cross_enc_total_score[k]["em"] / NUM_DATA, 4)
	avg_map = round(cross_enc_total_score[k]["map_score"] / NUM_DATA, 4)
	print(f"for top-{k}: Recall={avg_recall}; Precision={avg_precision}; F1={avg_f1}; EM={avg_em}; MAP={avg_map};")


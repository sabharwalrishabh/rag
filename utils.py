import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import faiss
import numpy as np
import bm25s
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
import string
from collections import Counter

def load_corpus():
    if os.path.exists('corpus.json'):
        with open('corpus.json') as f:
            return [tuple(entry) for entry in json.load(f)]

    with open('hotpot_dev_distractor_v1.json') as f:
        data = json.load(f)

    #de-duplicate
    unique_dict = {}
    for datapoint in data:
        for title, sentences in datapoint['context']:
            if title not in unique_dict:
                unique_dict[title] = {}
            for idx, text in enumerate(sentences):
                if idx not in unique_dict[title]:
                    unique_dict[title][idx] = text.strip()

    corpus = []
    for title, sentence_data in unique_dict.items():
        for idx, text in sentence_data.items():
            corpus.append((title, idx, text))

    with open('corpus.json', 'w') as f:
        json.dump(corpus, f)

    return corpus


class BM25():
    def __init__(self, corpus, sentences):
        self.corpus = corpus
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(sentences))

    def retrieve(self, question, k=5):
        results, scores = self.retriever.retrieve(bm25s.tokenize(question), k=k)
        return [self.corpus[i] for i in results[0]]

class DenseRetriever():
    def __init__(self, corpus, sentences):
        self.corpus = corpus
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')

        cache_path = 'cache/embeddings.npy'
        if os.path.exists(cache_path):
            embeddings = np.load(cache_path)
        else:
            embeddings = self.model.encode(sentences, normalize_embeddings=True, batch_size=512, show_progress_bar=True)
            os.makedirs('cache', exist_ok=True)
            np.save(cache_path, embeddings)

        embeddings = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, question, k=5):
        question = "Represent this sentence for searching relevant passages: " + question
        question_emb = np.array(self.model.encode([question], normalize_embeddings=True), dtype=np.float32)
        scores, indices = self.index.search(question_emb, k)
        return [self.corpus[i] for i in indices[0]]

#directly taken from https://github.com/hotpotqa/hotpot/blob/master/util.py
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def hybrid_retrieve(bm25, dense, question, k=15, rrf_k=60, pool_size=100):
    bm25_results = bm25.retrieve(question, k=pool_size)
    dense_results = dense.retrieve(question, k=pool_size)
    sparse_weight = 0.3
    dense_weight = 0.7

    bm25_ranks = {}
    for rank, (title, idx, text) in enumerate(bm25_results):
        bm25_ranks[(title, idx)] = rank + 1

    dense_ranks = {}
    for rank, (title, idx, text) in enumerate(dense_results):
        dense_ranks[(title, idx)] = rank + 1

    all_sentences = {}
    for title, idx, text in bm25_results + dense_results:
        all_sentences[(title, idx)] = text

    default_rank = 10000
    rrf_scores = {}
    for key in all_sentences:
        bm25_rank = bm25_ranks.get(key, default_rank)
        dense_rank = dense_ranks.get(key, default_rank)
        rrf_scores[key] = sparse_weight / (bm25_rank + rrf_k) + dense_weight / (dense_rank + rrf_k)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(title, idx, all_sentences[(title, idx)]) for (title, idx), score in sorted_results[:k]]


reranker = CrossEncoder('BAAI/bge-reranker-base', device='cpu')

def rerank_retrieve(dense, question, k=15, pool_size=100):
    candidates = dense.retrieve(question, k=pool_size)
    pairs = [(question, f"{title}: {text}") for title, idx, text in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [candidate for candidate, score in scored[:k]]


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# OLD_RAG_PROMPT = """
# Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.

# Examples:

# Facts:
# - 514th Flight Test Squadron: It is assigned to the Ogden Air Logistics Center (OO-ALC), Air Force Materiel Command, stationed at Hill Air Force Base, Utah.
# - Hill Air Force Base: The base was named in honor of Major Ployer Peter Hill of the U.S. Army Air Corps, who died test-flying a prototype of the B-17 Flying Fortress bomber.

# Q: Which airplane was this Major test-flying after whom the base, that 514th Flight Test Squadron is stated at, is named?
# A: B-17 Flying Fortress bomber

# Facts:
# - Johnny Angel: The movie stars George Raft, Claire Trevor and Signe Hasso, and features Hoagy Carmichael.
# - George Raft: George Raft (born George Ranft; September 26, 1901 – November 24, 1980) was an American film actor and dancer identified with portrayals of gangsters in crime melodramas of the 1930s and 1940s.

# Q: Which American film actor and dancer starred in the 1945 film Johnny Angel?
# A: George Raft

# Facts:
# - Here We Go Round the Mulberry Bush (film): Here We Go Round the Mulberry Bush is a 1967 British film made based on the novel of the same name by Hunter Davies.
# - Hunter Davies: Edward Hunter Davies, OBE (born 7 January 1936) is a British author, journalist and broadcaster.

# Q: When was the British author who wrote the novel on which "Here We Go Round the Mulberry Bush" was based born? 
# A: 7 January 1936

# Facts:
# - Daryl Hall: Daryl Franklin Hohl (born October 11, 1946), known professionally as Daryl Hall, is an American rock, R&B, and soul singer; keyboardist, guitarist, songwriter, and producer, best known as the co-founder and lead vocalist of Hall & Oates (with guitarist and songwriter John Oates).
# - Gerry Marsden: Gerard Marsden MBE (born 24 September 1942) is an English musician and television personality, best known for being leader of the British Merseybeat band Gerry and the Pacemakers.

# Q: Are Daryl Hall and Gerry Marsden both musicians?
# A: yes

# Facts:
# - Skin Yard: Skin Yard was an American grunge band from Seattle, Washington, who were active from 1985 to 1993.
# - Ostava: Ostava are an alternative rock band from Bulgaria.

# Q: Were the bands Skin Yard and Ostava from the U.S.?
# A: no

# Facts:
# - Boss Bailey: He was originally drafted by the Detroit Lions in the second round of the 2003 NFL Draft.
# - Boss Bailey: He is the brother of former NFL cornerback Champ Bailey.
# - Champ Bailey: He played college football for Georgia, where he earned consensus All-American honors, and was drafted by the Washington Redskins in the first round of the 1999 NFL Draft.
# - Champ Bailey: He is the brother of former NFL linebacker Boss Bailey.

# Q: What year was the brother of this first round draft pick by the Washington Redskins drafted?
# A: 2003

# Facts:
# {facts}

# Q: {question}
# A:
# """
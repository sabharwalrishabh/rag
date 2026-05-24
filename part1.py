import json
import os
import bm25s
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

if os.path.exists('corpus.json'):
    with open('corpus.json') as f:
        corpus = [tuple(entry) for entry in json.load(f)]
else:
    with open("hotpot_dev_distractor_v1.json", "r") as f:
        data = json.load(f)

    # global de-dup becoz we need to index all data (context)
    unique_dict = {}
    for datapoint in data:
        context = datapoint["context"]
        for title, sentences in context:
            if title not in unique_dict:
                unique_dict[title] = {}
            for sentence_idx, text in enumerate(sentences):
                if sentence_idx not in unique_dict[title]:
                    unique_dict[title][sentence_idx] = text.strip()

    corpus = []
    for title, sentence_data in unique_dict.items():
        for sentence_idx, text in sentence_data.items():
            corpus.append((title, sentence_idx, text.strip()))

    with open('corpus.json', 'w') as f:
        json.dump(corpus, f)

# to store all sentences which will be used for encoding later
sentences = [f"{title}: {text}" for title, _, text in corpus]
# sentences = [text for _, _, text in corpus]
# print(sentences[0])


retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(sentences))

question="Were Scott Derrickson and Ed Wood of the same nationality?"

def retrieve(question, k=5):
    results, scores = retriever.retrieve(bm25s.tokenize(question), k=k)
    return [corpus[i] for i in results[0]]

# print(retrieve(question))


#dense retrieval
model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')

if os.path.exists('embeddings.npy'):
    embeddings = np.load('embeddings.npy')
else:
    embeddings = model.encode(sentences, normalize_embeddings=True, batch_size=512, show_progress_bar=True)
    np.save('embeddings.npy', embeddings)

faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

def dense_retrieve(question, k=5):
    question = "Represent this sentence for searching relevant passages: " + question
    question_emb = model.encode([question], normalize_embeddings=True)
    scores, indices = faiss_index.search(question_emb, k)
    return [corpus[i] for i in indices[0]]

# print(dense_retrieve(question))

if __name__ == "__main__":
    print(retrieve(question))
    print(dense_retrieve(question))


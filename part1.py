import argparse
from utils import BM25, DenseRetriever, load_corpus

corpus = load_corpus()
sentences = [f"{title}: {text}" for title, _, text in corpus]

bm25 = BM25(corpus, sentences)
dense = DenseRetriever(corpus, sentences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    print(bm25.retrieve(question, k=args.k))
    print(dense.retrieve(question, k=args.k))


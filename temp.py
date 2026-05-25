import json
from dotenv import load_dotenv
import openai
from utils import f1_score, exact_match_score

load_dotenv()

RAG_PROMPT = """
Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.

Example answers for some random questions: B-17 Flying Fortress bomber; George Raft; 7 January 1936; yes; no; 2003

Facts:
{facts}

Q: {question}
A:
"""

def llm_judge(question, predicted, gold):
    prompt = f"""You are a judge comparing two answers to a question. 
                Determine if the predicted answer is semantically correct, even if phrased differently.
                Question: {question}\n
                Gold answer: {gold}\n
                Predicted answer: {predicted}\n

                Reply with EXACTLY one of:
                - CORRECT (same meaning, different format)
                - INCORRECT (wrong answer)
                - PARTIAL (partially correct but missing or extra information)"""

    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

with open("hotpot_dev_distractor_v1.json", "r") as f:
    data = json.load(f)

with open("corpus.json") as f:
    corpus = [tuple(entry) for entry in json.load(f)]

# (title, sentence_idx) -> text
corpus_lookup = {(title, sentence_idx): text for title, sentence_idx, text in corpus}

NUM_DATA = 100
total_em, total_f1 = 0, 0

for datapoint in data[:NUM_DATA]:
    question = datapoint["question"]
    supporting_facts = datapoint["supporting_facts"]  # list of [title, sentence_idx]

    facts_list = []
    for title, sentence_idx in supporting_facts:
        text = corpus_lookup.get((title, sentence_idx), "")
        if text:
            facts_list.append(f"{title}: {text}")
        else:
            print(f"WARNING: ({title!r}, {sentence_idx}) not found in corpus")

    facts = "\n".join(f"- {fact}" for fact in facts_list)
    prompt = RAG_PROMPT.format(facts=facts, question=question)

    response = openai.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    ground_truth = datapoint["answer"]
    em = exact_match_score(answer, ground_truth)
    f1, _, _ = f1_score(answer, ground_truth)

    total_em += em
    total_f1 += f1

    if em == 0:
        verdict = llm_judge(question, answer, ground_truth)
        print(f"Q: {question}")
        print(f"Predicted: {answer} | Gold: {ground_truth}")
        print(f"Verdict: {verdict}")
        print("---")

print("Gold-label RAG (oracle upper bound)")
print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")

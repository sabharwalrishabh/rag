import json
import os
import re
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import openai
from utils import f1_score, exact_match_score, DenseRetriever, load_corpus
from sentence_transformers import CrossEncoder


def extract_answer(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


NUM_DATA = 100
with open("hotpot_dev_distractor_v1.json", "r") as f:
	data = json.load(f)

corpus = load_corpus()
sentences = [f"{title}: {text}" for title, _, text in corpus]
dense = DenseRetriever(corpus, sentences)

#-------NO-RETRIEVAL QA------------#
NO_RETRIEVAL_PROMPT = """
Your task is to answer the given question concisely. Your answer should be a short phrase, name, number, date or yes/no. Do not explain your reasoning.
Example answers for some random questions: B-17 Flying Fortress bomber; George Raft; 7 January 1936; yes; no; 2003.

Q: {question}
A:
"""


def no_rag_baseline(model_name, client):
    total_em, total_f1 = 0, 0
    print("QA begins")
    for datapoint in data[:NUM_DATA]:
        question = datapoint["question"]
        prompt = NO_RETRIEVAL_PROMPT.format(question=question)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = extract_answer(response.choices[0].message.content)
        # print(f"Q: {question}")
        # print(f"A: {answer}")

        ground_truth = datapoint["answer"]
        em = exact_match_score(answer, ground_truth)
        f1, _, _ = f1_score(answer, ground_truth)

        total_em += em
        total_f1 += f1

    print("No Retrieval QA")
    print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
    print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")


#-------RAG-based QA------------#
reranker = CrossEncoder('BAAI/bge-reranker-base')
def rerank_retrieve(question, k=15, pool_size=100):
	#bi-encoder to get a 100 candidates
	candidates = dense.retrieve(question, k=pool_size)

	#cross-encoding
	pairs = [(question, f"{title}: {text}") for title, idx, text in candidates]
	scores = reranker.predict(pairs)

	scored = list(zip(candidates, scores))
	scored.sort(key=lambda x: x[1], reverse=True)
	return [candidate for candidate, score in scored[:k]]


#simplified prompt; old prompt in utils.py
RAG_PROMPT = """
Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.
Example answers for some random questions: B-17 Flying Fortress bomber; George Raft; 7 January 1936; yes; no; 2003

Facts:
{facts}

Q: {question}
A:
"""


def rag_baseline(model_name, client):
    total_em, total_f1 = 0, 0
    for datapoint in data[:NUM_DATA]:
        question = datapoint["question"]
        cross_enc_result = rerank_retrieve(question, k=5)

        facts_list = [f"{title}: {text}" for title, idx, text in cross_enc_result]
        facts = "\n".join(f"- {fact}" for fact in facts_list)
        prompt = RAG_PROMPT.format(facts=facts, question=question)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = extract_answer(response.choices[0].message.content)
        ground_truth = datapoint["answer"]
        em = exact_match_score(answer, ground_truth)
        f1, _, _ = f1_score(answer, ground_truth)

        total_em += em
        total_f1 += f1

    print("RAG-based QA")
    print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
    print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")

#---------Agentic RAG--------------#
tools = [
    {
        "type": "function",
        "function": {
            "name": "find_facts",
            "description": "Search a database for relevant facts. Use this tool to find specific facts needed to answer the question. If a question required multiple pieces of information, search each piece separately",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short specific search query (upto 10 words). Focus on key entities and relationships."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def agentic_rag(model_name, client, question):
    #Get initial facts
    cross_enc_result = rerank_retrieve(question, k=5)
    all_results = list(cross_enc_result)
    all_facts = [f"{title}: {text}" for title, idx, text in cross_enc_result]

    messages = [
        {"role": "system", "content": "You are a research agent. Your job is to determine if the given facts are sufficient to answer the question. If the facts are sufficient, reply with DONE. If not, use the find_facts tool to search for the missing information. Do not answer the question yourself — only gather facts and reply DONE when you have sufficient facts."},
        {"role": "user", "content": f"Question: {question}\n\nFacts so far:\n" + "\n".join(f"- {f}" for f in all_facts)}
    ]

    for _ in range(2):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break  

        messages.append(msg)
        for tool_call in msg.tool_calls:
            query = json.loads(tool_call.function.arguments)["query"]
            results = rerank_retrieve(query, k=5)
            new_facts = [f"{title}: {text}" for title, idx, text in results]
            all_facts.extend(new_facts)
            all_results.extend(results)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "\n".join(new_facts)
            })

    #Dedup facts
    all_facts = list(dict.fromkeys(all_facts))

    #answer-generation agent
    facts_str = "\n".join(f"- {f}" for f in all_facts)
    prompt = RAG_PROMPT.format(facts=facts_str, question=question)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_answer(response.choices[0].message.content), all_results


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

    response = eval_client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_answer(response.choices[0].message.content)


def agentic_rag_baseline(model_name, client, use_judge=False):
    total_em, total_f1 = 0, 0
    for datapoint in data[:NUM_DATA]:
        question = datapoint["question"]
        answer, all_facts = agentic_rag(model_name, client, question)  
        ground_truth = datapoint["answer"]
        
        em = exact_match_score(answer, ground_truth)
        f1, _, _ = f1_score(answer, ground_truth)
        
        total_em += em
        total_f1 += f1

        if use_judge:
            if em == 0:
                gold_sf = set((title, idx) for title, idx in datapoint["supporting_facts"])
                # Check if gold facts were retrieved
                retrieved_ids = set((title, idx) for title, idx, text in all_facts)
                found_sf = gold_sf & retrieved_ids
                retrieval_success = gold_sf.issubset(retrieved_ids)
                
                verdict = llm_judge(question, answer, ground_truth)
            
                print(f"Q: {question}")
                print(f"Predicted: {answer} | Gold: {ground_truth}")
                print(f"Gold facts found: {len(found_sf)}/{len(gold_sf)}")
                print(f"Retrieval complete: {retrieval_success}")
                print(f"Verdict: {verdict}")
                print("---")

    print("Agentic RAG-based QA")
    print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
    print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_rag_baseline", action="store_true", help="whether to run the 'no rag' baseline or not")
    parser.add_argument("--rag_baseline", action="store_true", help="whether to run the 'rag'")
    parser.add_argument("--agentic_rag_baseline", action="store_true", help="whether to run the 'agentic rag' or not")
    parser.add_argument("--use_judge", action="store_true", help="whether to use an LLM as judge or not")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14",
                    choices=["gpt-4.1-mini-2025-04-14", "qwen3-8b"],
                    help="which LLM backend to use")
    args = parser.parse_args()

    load_dotenv()

    if args.use_judge and not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("--use_judge requires OPENAI_API_KEY to be set in the environment or .env file")

    if args.model == "qwen3-8b":
        client = OpenAI(base_url="http://localhost:8002/v1", api_key="not-required")
        model_name = "Qwen/Qwen3-8B-AWQ"
    else:
        client = OpenAI()
        model_name = "gpt-4.1-mini-2025-04-14"
    
    if args.use_judge:
        
        eval_client = OpenAI()

    if args.no_rag_baseline:
        print("Running the no RAG baseline")
        no_rag_baseline(model_name, client)
    if args.rag_baseline:
        print("Running the RAG baseline")
        rag_baseline(model_name, client)
    if args.agentic_rag_baseline:
        print("Running the agentic RAG baseline")
        agentic_rag_baseline(model_name, client, args.use_judge)
    

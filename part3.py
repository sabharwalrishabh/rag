import json
from dotenv import load_dotenv
from openai import OpenAI
import openai
from utils import f1_score, exact_match_score
from sentence_transformers import CrossEncoder
from part1 import dense_retrieve

load_dotenv()
client = OpenAI()

NO_RETRIEVAL_PROMPT = """
Your task is to answer the given question concisely. Your answer should be a short phrase, name, number, date or yes/no. Do not explain your reasoning.

Examples:
Q: Which airplane was this Major test-flying after whom the base, that 514th Flight Test Squadron is stated at, is named?
A: B-17 Flying Fortress bomber

Q: Which American film actor and dancer starred in the 1945 film Johnny Angel?
A: George Raft

Q: When was the British author who wrote the novel on which "Here We Go Round the Mulberry Bush" was based born? 
A: 7 January 1936

Q: Are Daryl Hall and Gerry Marsden both musicians?
A: yes

Q: Were the bands Skin Yard and Ostava from the U.S.?
A: no

Q: What year was the brother of this first round draft pick by the Washington Redskins drafted?
A: 2003

Q: {question}
A:
"""

with open("hotpot_dev_distractor_v1.json", "r") as f:
	data = json.load(f)

NUM_DATA = 100
# total_em, total_f1 = 0, 0
# print("QA begins")
# for datapoint in data[:NUM_DATA]:
#     question = datapoint["question"]
#     prompt = NO_RETRIEVAL_PROMPT.format(question=question)

#     response = client.responses.create(
#         model="gpt-4.1-mini-2025-04-14",
#         input=prompt
#     )

#     answer = response.output_text.strip()
#     # print(f"Q: {question}")
#     # print(f"A: {answer}")

#     ground_truth = datapoint["answer"]
#     em = exact_match_score(answer, ground_truth)
#     f1, _, _ = f1_score(answer, ground_truth)

#     total_em += em
#     total_f1 += f1

# print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
# print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")

# RAG-based QA
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

# RAG_PROMPT = """
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

RAG_PROMPT = """
Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.

Example answers for some random questions: B-17 Flying Fortress bomber; George Raft; 7 January 1936; yes; no; 2003

Facts:
{facts}

Q: {question}
A:
"""

total_em, total_f1 = 0, 0
for datapoint in data[:NUM_DATA]:
    question = datapoint["question"]
    cross_enc_result = rerank_retrieve(question, k=5)
    # cross_enc_output = [[title, sentence_idx] for title, sentence_idx, text in cross_enc_result]
    facts_list = [f"{title}: {text}" for title, idx, text in cross_enc_result]
    facts = "\n".join(f"- {fact}" for fact in facts_list)
    prompt = RAG_PROMPT.format(facts=facts, question=question)

    # response = client.responses.create(
    #     model="gpt-4.1-mini-2025-04-14",
    #     input=prompt
    # )

    # answer = response.output_text.strip()
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

print("RAG-based QA")
print(f"Avg EM score: {round(total_em / NUM_DATA, 4)}")
print(f"Avg F1 score: {round(total_f1 / NUM_DATA, 4)}")

# Agentic RAG
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

# def agentic_rag(question):
#     cross_enc_result = rerank_retrieve(question, k=5)
#     facts_list = [f"{title}: {text}" for title, idx, text in cross_enc_result]
#     facts = "\n".join(f"- {f}" for f in facts_list)

#     messages = [
#         {
#             "role": "system",
#             "content": """ You are an expert question-answering agent. 
#                             Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.
#                             Example answers for some random questions: B-17 Flying Fortress bomber; George Raft; 7 January 1936; yes; no; 2003.
#                             Additionally, you can use find_facts tool to find relevant facts. 
#                             However, use the tool only if the current facts are not sufficient and you need more information."""
#         },
#         {
#             "role": "user", 
#             "content": f"Facts:\n{facts} \nQuestion: {question}\n"
            
#         }
#     ]

#     additional_searches = 0
#     for _ in range(2):
#         response = openai.chat.completions.create(
#             model="gpt-4.1-mini-2025-04-14",
#             messages=messages,
#             tools=tools
#         )

#         model_output = response.choices[0].message
#         if model_output.tool_calls is None:
#             print(f"Answered after {additional_searches} additional searches")
#             return model_output.content.strip()
        
#         messages.append(model_output)
#         for tool_call in model_output.tool_calls:
#             query = json.loads(tool_call.function.arguments)["query"]
#             additional_searches+=1
#             print(f"  Search {additional_searches}: '{query}'")
#             results = rerank_retrieve(query, k=5)
#             new_facts = [f"{title}: {text}" for title, idx, text in results]
#             print(f"old facts: {facts}\n")
#             print(f"new facts: {new_facts}\n")
#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": "\n".join(new_facts)
#             })
            
    
#     # incase the initial + 2 extra iterations completeled and still no answer
#     print(f"Forced answer after {additional_searches} searches")
#     messages.append({"role": "user", "content": "Based on all the facts retrieved, give your final concise answer now."})
#     response = openai.chat.completions.create(
#         model="gpt-4.1-mini-2025-04-14",
#         messages=messages,
#     )
#     return response.choices[0].message.content.strip()


def agentic_rag(question):
    # Get initial facts
    cross_enc_result = rerank_retrieve(question, k=5)
    all_results = list(cross_enc_result)
    all_facts = [f"{title}: {text}" for title, idx, text in cross_enc_result]

    messages = [
        {"role": "system", "content": "You are a research agent. Your job is to determine if the given facts are sufficient to answer the question. If the facts are sufficient, reply with DONE. If not, use the find_facts tool to search for the missing information. Do not answer the question yourself — only gather facts and reply DONE when you have sufficient facts."},
        {"role": "user", "content": f"Question: {question}\n\nFacts so far:\n" + "\n".join(f"- {f}" for f in all_facts)}
    ]

    for _ in range(2):
        response = openai.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=messages,
            tools=tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            break  # model said DONE or gave an answer — either way, stop searching

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

    # Deduplicate facts
    all_facts = list(dict.fromkeys(all_facts))

    # Now use your WORKING single-step RAG prompt for the final answer
    facts_str = "\n".join(f"- {f}" for f in all_facts)
    prompt = RAG_PROMPT.format(facts=facts_str, question=question)

    response = openai.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip(), all_results

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


total_em, total_f1 = 0, 0
for datapoint in data[:NUM_DATA]:
    question = datapoint["question"]
    answer, all_facts = agentic_rag(question)  
    ground_truth = datapoint["answer"]

    gold_sf = set((t, i) for t, i in datapoint["supporting_facts"])
    
    em = exact_match_score(answer, ground_truth)
    f1, _, _ = f1_score(answer, ground_truth)
    
    total_em += em
    total_f1 += f1
    if em == 0:
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

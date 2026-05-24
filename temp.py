import json

with open("hotpot_dev_distractor_v1.json", "r") as f:
	data = json.load(f)

with open('corpus.json') as f:
    corpus = [tuple(entry) for entry in json.load(f)]

corpus_lookup = {(title, sentence_idx): text for title, sentence_idx, text in corpus}

NUM_DATA = 100
cnt = NUM_DATA+1
for datapoint in data[NUM_DATA+1:NUM_DATA+21]:
    facts = []
    if cnt in [109, 105, 112, 119, 118, 102]:
        print(cnt)
        print(datapoint["question"])
        print(datapoint["supporting_facts"])
        supp_facts = datapoint["supporting_facts"]
        for title, fact_idx in supp_facts:
            text = corpus_lookup.get((title, fact_idx), "")
            facts.append(f"{title}: {text}")

        print(facts)
        print(datapoint["answer"])
        print()
    cnt+=1


RAG_PROMPT = """
Answer the question using ONLY the provided facts. Extract the answer directly from the facts. Your answer should be a short phrase, name, number, date, or yes/no. Do not explain your reasoning.

Examples:

Facts:
- 514th Flight Test Squadron: It is assigned to the Ogden Air Logistics Center (OO-ALC), Air Force Materiel Command, stationed at Hill Air Force Base, Utah.
- Hill Air Force Base: The base was named in honor of Major Ployer Peter Hill of the U.S. Army Air Corps, who died test-flying a prototype of the B-17 Flying Fortress bomber.

Q: Which airplane was this Major test-flying after whom the base, that 514th Flight Test Squadron is stated at, is named?
A: B-17 Flying Fortress bomber

Facts:
- Johnny Angel: The movie stars George Raft, Claire Trevor and Signe Hasso, and features Hoagy Carmichael.
- George Raft: George Raft (born George Ranft; September 26, 1901 – November 24, 1980) was an American film actor and dancer identified with portrayals of gangsters in crime melodramas of the 1930s and 1940s.

Q: Which American film actor and dancer starred in the 1945 film Johnny Angel?
A: George Raft

Facts:
- Here We Go Round the Mulberry Bush (film): Here We Go Round the Mulberry Bush is a 1967 British film made based on the novel of the same name by Hunter Davies.
- Hunter Davies: Edward Hunter Davies, OBE (born 7 January 1936) is a British author, journalist and broadcaster.

Q: When was the British author who wrote the novel on which "Here We Go Round the Mulberry Bush" was based born? 
A: 7 January 1936

Facts:
- Daryl Hall: Daryl Franklin Hohl (born October 11, 1946), known professionally as Daryl Hall, is an American rock, R&B, and soul singer; keyboardist, guitarist, songwriter, and producer, best known as the co-founder and lead vocalist of Hall & Oates (with guitarist and songwriter John Oates).
- Gerry Marsden: Gerard Marsden MBE (born 24 September 1942) is an English musician and television personality, best known for being leader of the British Merseybeat band Gerry and the Pacemakers.

Q: Are Daryl Hall and Gerry Marsden both musicians?
A: yes

Facts:
- Skin Yard: Skin Yard was an American grunge band from Seattle, Washington, who were active from 1985 to 1993.
- Ostava: Ostava are an alternative rock band from Bulgaria.

Q: Were the bands Skin Yard and Ostava from the U.S.?
A: no

Facts:
- Boss Bailey: He was originally drafted by the Detroit Lions in the second round of the 2003 NFL Draft.
- Boss Bailey: He is the brother of former NFL cornerback Champ Bailey.
- Champ Bailey: He played college football for Georgia, where he earned consensus All-American honors, and was drafted by the Washington Redskins in the first round of the 1999 NFL Draft.
- Champ Bailey: He is the brother of former NFL linebacker Boss Bailey.

Q: What year was the brother of this first round draft pick by the Washington Redskins drafted?
A: 2003

Facts:
{facts}

Q: {question}
A:
"""

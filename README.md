# RAG Pipeline

Question answering over the [HotpotQA](https://hotpotqa.github.io/) dataset using BM25, dense retrieval, cross-encoder reranking, and an LLM backend (GPT-4.1-mini or local Qwen3-8B via vLLM).

---

## Quickstart via Docker

The easiest way to run the API. No Python setup required.

### 1. Pull the image

```bash
docker pull sabharwalrishabh/rag-api:latest
```

### 2. Create a `.env` file

```bash
# For OpenAI backend (default)
OPENAI_API_KEY=your_key_here

# For local vLLM backend (optional)
# OPENAI_BASE_URL=http://host.docker.internal:8002/v1
# MODEL_NAME=Qwen/Qwen3-8B-AWQ
```

### 3. Start the API

```bash
docker compose up
```

The API will be available at `http://localhost:8000`.

> **First run note:** On first startup the container downloads the retrieval models (~630MB) and encodes the entire corpus into embeddings. This takes 20–40 minutes. Both are cached in Docker volumes so every restart after that is instant.

---

## API Endpoints

### Health check
```
GET /health
```

### Retrieval

| Endpoint | Method | Description |
|---|---|---|
| `/retrieve/hybrid` | POST | BM25 + dense retrieval fused with RRF |
| `/retrieve/rerank` | POST | Dense retrieval re-ranked by a cross-encoder |

Request body:
```json
{ "question": "Who directed Inception?", "k": 5 }
```

### Question Answering

| Endpoint | Method | Description |
|---|---|---|
| `/qa/no_rag` | POST | LLM answers directly from parametric knowledge |
| `/qa/rag` | POST | Single-shot RAG — retrieves top-k facts then answers |
| `/qa/agentic` | POST | Agentic RAG — LLM iteratively calls a search tool to gather evidence |

Request body:
```json
{ "question": "Who directed Inception?", "k": 5 }
```

Interactive API docs are available at `http://localhost:8000/docs`.

---

## LLM Backends

### Option A — OpenAI (default)

Set `OPENAI_API_KEY` in `.env`. The API uses `gpt-4.1-mini-2025-04-14` by default.

### Option B — Local Qwen3-8B via vLLM

Requires a machine with an NVIDIA GPU. Run the vLLM server separately:

```bash
vllm serve Qwen/Qwen3-8B-AWQ --port 8002 --max-model-len 2048 \
    --gpu-memory-utilization 0.5 --enable-auto-tool-choice --tool-call-parser hermes
```

Then set these in `.env`:
```bash
OPENAI_BASE_URL=http://host.docker.internal:8002/v1
MODEL_NAME=Qwen/Qwen3-8B-AWQ
```

---

## Local Setup (without Docker)

Requires Python 3.11 and a CUDA 12.8 driver.

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch (cu126)

```bash
pip install torch==2.7.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install vLLM (optional — only needed for Qwen3-8B)

```bash
pip install https://github.com/vllm-project/vllm/releases/download/v0.9.1/vllm-0.9.1%2Bcu126-cp38-abi3-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

Patch vLLM for transformers 4.x compatibility — open `venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/ovis.py` and change:

```python
# Before
AutoConfig.register("aimv2", AIMv2Config)

# After
AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

```bash
OPENAI_API_KEY=your_key_here
```

### 6. Run the evaluation scripts

```bash
python part3.py --agentic_rag_baseline
```

#### Flags

| Flag | Description |
|---|---|
| `--no_rag_baseline` | LLM answers directly from parametric knowledge |
| `--rag_baseline` | Retrieves top-5 facts via dense retrieval + reranking, feeds to LLM |
| `--agentic_rag_baseline` | LLM iteratively calls a `find_facts` tool to gather evidence before answering |
| `--model` | `gpt-4.1-mini-2025-04-14` (default) or `qwen3-8b` (requires local vLLM server on port 8002) |
| `--use_judge` | Uses GPT-4.1 to judge semantic correctness of wrong answers. Requires `OPENAI_API_KEY` |

Multiple flags can be combined:

```bash
python part3.py --rag_baseline --agentic_rag_baseline --model qwen3-8b
```

To also run the API server locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

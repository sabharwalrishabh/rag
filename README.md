# RAG Pipeline

Question answering over the HotpotQA dataset using BM25, dense retrieval, reranking, and an LLM backend (GPT-4.1-mini or local Qwen3-8B via vLLM).

---

## Environment Setup

Requires Python 3.11 and a CUDA 12.8 driver.

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch (cu126 — compatible with CUDA 12.8)

```bash
pip install torch==2.7.0+cu126 torchaudio==2.7.0+cu126 torchvision==0.22.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install vLLM (cu126 prebuilt wheel)

```bash
pip install https://github.com/vllm-project/vllm/releases/download/v0.9.1/vllm-0.9.1%2Bcu126-cp38-abi3-manylinux1_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Patch vLLM (one-time fix for transformers 4.x compatibility)

Open `venv/lib/python3.11/site-packages/vllm/transformers_utils/configs/ovis.py` and change line 76:

```python
# Before
AutoConfig.register("aimv2", AIMv2Config)

# After
AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)
```

### 6. Set up environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

Only required when using `--model gpt-4.1-mini-2025-04-14` or `--use_judge`.

---

## Running the Pipeline

### Option A — Local Qwen3-8B (vLLM)

First, start the vLLM server in a separate terminal:

```bash
vllm serve Qwen/Qwen3-8B-AWQ --port 8002 --max-model-len 2048 \
    --gpu-memory-utilization 0.5 --enable-auto-tool-choice --tool-call-parser hermes
```

Wait until you see `Uvicorn running on http://0.0.0.0:8002`, then run:

```bash
python part3.py --agentic_rag_baseline --model qwen3-8b
```

### Option B — GPT-4.1-mini (OpenAI API)

No server needed. Requires `OPENAI_API_KEY` in `.env`:

```bash
python part3.py --agentic_rag_baseline
```

---

## Flags

| Flag | Description |
|------|-------------|
| `--no_rag_baseline` | Run the no-retrieval baseline — LLM answers directly from its parametric knowledge |
| `--rag_baseline` | Run the RAG baseline — retrieves top-5 facts via dense retrieval + reranking, feeds them to the LLM |
| `--agentic_rag_baseline` | Run the agentic RAG pipeline — the LLM iteratively calls a `find_facts` tool to gather evidence before answering |
| `--model` | LLM backend: `gpt-4.1-mini-2025-04-14` (default, requires API key) or `qwen3-8b` (local vLLM server on port 8002) |
| `--use_judge` | After each wrong answer, calls GPT-4.1 to judge whether the predicted answer is semantically correct (CORRECT / PARTIAL / INCORRECT). Requires `OPENAI_API_KEY` |

Multiple baselines can be combined in one run:

```bash
python part3.py --rag_baseline --agentic_rag_baseline --model qwen3-8b
```

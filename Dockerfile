FROM python:3.11-slim

# Prevents OpenMP conflicts between PyTorch and FAISS (same fix as in code)
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# STEP 1 from requirements.txt: install PyTorch cu126 wheels first
RUN pip install --no-cache-dir \
    torch==2.7.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# STEP 3 from requirements.txt: install everything else
# --extra-index-url lets pip verify the cu126 version pins without re-downloading torch
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

COPY api.py utils.py part3.py ./
COPY hotpot_dev_distractor_v1.json ./
RUN mkdir /app/cache

# corpus.json is auto-generated from the dataset on first run.
# embeddings.npy is auto-generated and persisted in the rag-cache named volume.

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

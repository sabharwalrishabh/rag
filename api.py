import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_corpus, BM25, DenseRetriever, hybrid_retrieve, rerank_retrieve
from part3 import agentic_rag, NO_RETRIEVAL_PROMPT, RAG_PROMPT

load_dotenv()


# ----- Request / Response models -----

class QuestionRequest(BaseModel):
    question: str
    k: int = 5

class QuestionOnlyRequest(BaseModel):
    question: str

class RetrievalResult(BaseModel):
    title: str
    idx: int
    text: str

class RetrievalResponse(BaseModel):
    results: list[RetrievalResult]

class QAResponse(BaseModel):
    answer: str

class AgenticQAResponse(BaseModel):
    answer: str
    retrieved_facts: list[str]


# ----- Lifespan: load models once at startup -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Corpus + retrievers
    corpus = load_corpus()
    sentences = [f"{title}: {text}" for title, _, text in corpus]
    app.state.bm25 = BM25(corpus, sentences)
    app.state.dense = DenseRetriever(corpus, sentences)

    # LLM client — reads MODEL_NAME and OPENAI_BASE_URL from .env
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini-2025-04-14")
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        # local vLLM / Qwen backend
        app.state.client = OpenAI(base_url=base_url, api_key="not-required")
    else:
        app.state.client = OpenAI()
    app.state.model_name = model_name

    yield  # server is live and handling requests here

    # shutdown cleanup (nothing needed for this pipeline)


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)


# ----- Health check -----

@app.get("/health")
def health():
    return {"status": "ok"}


# ----- Retrieval endpoints -----

@app.post("/retrieve/hybrid", response_model=RetrievalResponse)
def retrieve_hybrid(req: QuestionRequest, request: Request):
    """Hybrid BM25 + dense retrieval with RRF fusion."""
    state = request.app.state
    results = hybrid_retrieve(state.bm25, state.dense, req.question, k=req.k)
    return RetrievalResponse(results=[
        RetrievalResult(title=title, idx=idx, text=text)
        for title, idx, text in results
    ])


@app.post("/retrieve/rerank", response_model=RetrievalResponse)
def retrieve_rerank(req: QuestionRequest, request: Request):
    """Dense retrieval re-ranked by a cross-encoder."""
    state = request.app.state
    results = rerank_retrieve(state.dense, req.question, k=req.k)
    return RetrievalResponse(results=[
        RetrievalResult(title=title, idx=idx, text=text)
        for title, idx, text in results
    ])


# ----- QA endpoints -----

@app.post("/qa/no_rag", response_model=QAResponse)
def qa_no_rag(req: QuestionOnlyRequest, request: Request):
    """Answer directly from the LLM with no retrieval."""
    state = request.app.state
    prompt = NO_RETRIEVAL_PROMPT.format(question=req.question)
    response = state.client.chat.completions.create(
        model=state.model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return QAResponse(answer=response.choices[0].message.content.strip())


@app.post("/qa/rag", response_model=QAResponse)
def qa_rag(req: QuestionRequest, request: Request):
    """Single-shot RAG: retrieve top-k facts with cross-encoder, then answer."""
    state = request.app.state
    results = rerank_retrieve(state.dense, req.question, k=req.k)
    facts = "\n".join(f"- {title}: {text}" for title, idx, text in results)
    prompt = RAG_PROMPT.format(facts=facts, question=req.question)
    response = state.client.chat.completions.create(
        model=state.model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return QAResponse(answer=response.choices[0].message.content.strip())


@app.post("/qa/agentic", response_model=AgenticQAResponse)
def qa_agentic(req: QuestionRequest, request: Request):
    """Agentic RAG: iteratively retrieves facts until sufficient, then answers."""
    state = request.app.state
    answer, all_results = agentic_rag(state.dense, state.model_name, state.client, req.question)
    retrieved_facts = list(dict.fromkeys(
        f"{title}: {text}" for title, idx, text in all_results
    ))
    return AgenticQAResponse(answer=answer, retrieved_facts=retrieved_facts)

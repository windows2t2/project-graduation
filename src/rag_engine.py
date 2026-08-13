"""
src/rag_engine.py
RAG (Retrieval-Augmented Generation) engine powered by DeepSeek via LangChain.

Builds a vector store from job descriptions + salary data, then answers
user questions with real data-backed responses.
"""

from typing import List, Optional
import pandas as pd

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI as OpenAIClient

from src.utils import get_deepseek_config, DATA_PROCESSED, PROJECT_ROOT, logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cloud DeepSeek — used as a fallback when the configured (e.g. local) endpoint is down.
CLOUD_BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = "deepseek-chat"


# ---------------------------------------------------------------------------
# DeepSeek client (raw OpenAI-compatible, no LangChain wrapper)
# ---------------------------------------------------------------------------
def _get_deepseek_client(base_url: Optional[str] = None, max_retries: int = 1) -> OpenAIClient:
    cfg = get_deepseek_config()
    return OpenAIClient(
        api_key=cfg["api_key"],
        base_url=base_url or cfg["base_url"],
        max_retries=max_retries,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(df, text_columns=None, collection_name="job_market"):
    if text_columns is None:
        text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    logger.info("Building vector store from columns: %s", text_columns)

    documents = []
    for idx, row in df.iterrows():
        parts = []
        for col in text_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {val}")
        if parts:
            documents.append(Document(page_content="\n".join(parts), metadata={"row_id": int(idx)}))

    if not documents:
        raise ValueError("No text content found in the DataFrame.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))

    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name=collection_name, persist_directory=str(CHROMA_DIR),
    )
    logger.info("Vector store built — %d documents", vector_store._collection.count())
    return vector_store


def load_vector_store(collection_name="job_market"):
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=collection_name, embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


# ---------------------------------------------------------------------------
# RAG — manual retrieval + DeepSeek
# ---------------------------------------------------------------------------
def create_rag_chain(vector_store=None, collection_name="job_market"):
    """Return a dict with 'retriever' and 'config' for ask()."""
    if vector_store is None:
        vector_store = load_vector_store(collection_name)
    cfg = get_deepseek_config()
    return {
        "retriever": vector_store.as_retriever(search_kwargs={"k": 5}),
        "client": _get_deepseek_client(),  # kept for backward compatibility
        "config": cfg,
        "model": cfg["model"],
    }


def ask(chain: dict, question: str) -> dict:
    """Retrieve relevant docs, construct prompt, call DeepSeek.

    Tries the configured endpoint first (e.g. local llama.cpp on :8080) and
    falls back to the cloud DeepSeek API if that one is unreachable, so a dead
    backend never crashes the app.
    """
    retriever = chain["retriever"]
    cfg = chain.get("config") or get_deepseek_config()

    # Retrieve
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content[:400] for d in docs[:5])

    # Build prompt
    system_prompt = (
        "You are a helpful career advisor. Answer the user's question using ONLY "
        "the provided job market data below. Be concise. If the data doesn't contain "
        "the answer, say so honestly.\n\n"
        f"JOB MARKET DATA:\n{context}"
    )

    # Endpoint candidates: configured first, cloud DeepSeek as fallback.
    base_url = (cfg.get("base_url") or CLOUD_BASE_URL).rstrip("/")
    endpoints = [{"base_url": base_url, "model": cfg.get("model") or CLOUD_MODEL}]
    if base_url != CLOUD_BASE_URL:
        endpoints.append({"base_url": CLOUD_BASE_URL, "model": CLOUD_MODEL})

    last_err: Optional[Exception] = None
    for ep in endpoints:
        try:
            client = _get_deepseek_client(base_url=ep["base_url"], max_retries=1)
            response = client.chat.completions.create(
                model=ep["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            sources = [d.page_content[:300] for d in docs]
            return {"answer": answer, "sources": sources, "endpoint": ep["base_url"]}
        except Exception as exc:  # noqa: BLE001 — try the next endpoint
            last_err = exc
            logger.warning("LLM endpoint %s failed (%s): %s", ep["base_url"], type(exc).__name__, exc)

    raise RuntimeError(
        "Could not reach any AI backend. Tried "
        f"{[e['base_url'] for e in endpoints]}. Last error: "
        f"{type(last_err).__name__}: {last_err}"
    ) from last_err

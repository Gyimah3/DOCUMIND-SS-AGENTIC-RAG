"""Singleton FlashRank re-ranker service for the RAG pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from langchain_core.documents import Document
from loguru import logger

# Module-level singleton — loaded once on first call
_ranker = None
_model_name: str = "ms-marco-MiniLM-L-12-v2"


def _get_ranker():
    """Lazy-load the FlashRank Ranker (singleton)."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        logger.info("Loading FlashRank model: {}", _model_name)
        start = time.perf_counter()
        _ranker = Ranker(model_name=_model_name)
        logger.info(
            "FlashRank model loaded in {:.0f}ms",
            (time.perf_counter() - start) * 1000,
        )
    return _ranker


def rerank_sync(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Re-rank documents against *query* using FlashRank (synchronous).

    Each returned Document keeps its original metadata and gains a
    ``rerank_score`` key.
    """
    if not documents:
        return []

    from flashrank import RerankRequest

    ranker = _get_ranker()

    # FlashRank expects a list of dicts with "text" (and optional "meta")
    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(documents)
    ]

    start = time.perf_counter()
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "FlashRank re-ranked {} candidates in {:.0f}ms",
        len(documents),
        elapsed_ms,
    )

    # Map re-ranked results back to Document objects
    reranked: List[Document] = []
    for r in results:
        idx = r["id"]
        original = documents[idx]
        meta = {**original.metadata, "rerank_score": float(r["score"])}
        reranked.append(Document(page_content=original.page_content, metadata=meta))

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked


async def rerank(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Async wrapper — runs FlashRank inference in a thread so the event loop stays free."""
    return await asyncio.to_thread(rerank_sync, query, documents, top_k)

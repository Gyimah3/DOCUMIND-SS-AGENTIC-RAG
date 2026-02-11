# Re-ranking in the RAG Pipeline

## What is Re-ranking?

Re-ranking is a second-stage relevance scoring step that sits between vector retrieval and LLM generation. The vector store returns candidates via cosine similarity (a rough, embedding-level match). A cross-encoder re-ranker then scores each candidate *against the actual query* to produce a much more accurate relevance ordering.

**Pipeline flow:**

```
User query
  -> Embedding (OpenAI)
  -> Redis KNN retrieval (k * 3 candidates)
  -> FlashRank cross-encoder re-ranking
  -> Top-k documents
  -> LLM generation
```

## Why It Matters

Cosine similarity operates on pre-computed embeddings — it has no direct awareness of the query at scoring time. A cross-encoder jointly encodes (query, passage) and can capture nuances that embedding distance misses. In practice, adding re-ranking typically improves answer quality noticeably without changing any other part of the pipeline.

## Implementation: FlashRank

We use **FlashRank**, a lightweight ONNX-based re-ranker that runs locally with no API key. It adds ~10-50ms per re-rank call.

### How It Works

1. The retriever over-fetches `k * candidates_multiplier` (default 3x) documents from Redis.
2. `services/reranker.py` passes all candidates through the FlashRank cross-encoder model.
3. Results are sorted by re-rank score and the top `k` are returned.
4. Each returned document's metadata includes the original `vector_score` plus a new `rerank_score`.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Singleton model** | FlashRank model loads once (~200ms), reused across all requests. Module-level `_ranker` variable. |
| **`asyncio.to_thread()`** | ONNX inference is CPU-bound; running in a thread keeps the async event loop responsive. |
| **3x over-retrieval** | Fetching 3x candidates gives the re-ranker enough diversity without excessive Redis load. Configurable via `candidates_multiplier`. |
| **Backward compatible** | Setting `rerank=False` on `InformationLookupTool` skips re-ranking entirely — exact previous behavior. |

### Configuration

In `services/reranker.py`, the model is set at module level:

```python
_model_name: str = "ms-marco-MiniLM-L-12-v2"
```

Other FlashRank models can be swapped in by changing this value (e.g., `ms-marco-MultiBERT-L-12` for multilingual).

The over-retrieval factor is configurable per retriever:

```python
store.as_retriever(..., candidates_multiplier=4)  # fetch 4x instead of 3x
```

### Disabling Re-ranking

Pass `rerank=False` when constructing the tool:

```python
tool = InformationLookupTool(
    embedding_model="text-embedding-3-small",
    index_name="my_index",
    rerank=False,  # pure cosine similarity, no re-ranking
)
```

## Alternative: LangChain ContextualCompressionRetriever

LangChain provides a built-in integration via `FlashrankRerank` and `ContextualCompressionRetriever`. This project uses a custom integration for more control, but here is the LangChain-native approach for reference:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

compressor = FlashrankRerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,  # your existing retriever
)

# Use compression_retriever in place of base_retriever
results = await compression_retriever.ainvoke("your query")
```

The custom approach was chosen because:
- It gives explicit control over the over-retrieval factor
- It keeps the singleton model loading pattern (LangChain's compressor creates a new Ranker each time by default)
- It adds `rerank_score` to metadata for observability
- It integrates cleanly with the existing `RedisVectorStoreRetriever`

## Performance

| Metric | Typical Value |
|--------|--------------|
| Model load (first call) | ~200ms |
| Re-rank 15 passages | ~10-50ms |
| Model size on disk | ~25MB |
| Memory overhead | ~50MB |

## Verifying Re-ranking

After enabling, check the application logs for:

```
FlashRank model loaded in 185ms
Re-ranking: 15 candidates -> top 5
FlashRank re-ranked 15 candidates in 23ms
```

The `used_docs` in the NDJSON stream will include `rerank_score` in each document's metadata.

# RAG / RediSearch: Empty Content in Retrieval Results

## Problem Summary

**Observed:** Data exists in Redis (confirmed via Redis Insight): hash documents have non-empty `content`, `content_vector`, `filename` (e.g. "Manus"), `page_number`, `type`, etc. Despite this, RAG retrieval returns documents with **empty `page_content`** (and effectively empty results to the user).

- **Redis Insight:** Hash fields show real values (e.g. content: "LoRA Fine-tuning python trainer.py ...", content_vector: binary blob, filename: "Manus").
- **FT.SEARCH / redis-py:** Query returns the correct number of documents, but the parsed `Document` objects have empty `content` and `content_vector` when read via `getattr(d, "content", None)` in our code.
- **Index:** The index in use (e.g. "zazi") may have a **prefix** that does not match the key prefix under which documents were stored (e.g. index prefix `za` vs keys `zazi:*`), so either docs are not found at all, or they are found but field values are not returned correctly.

## Technical Context

- **Stack:** FastAPI app (DocuMindSS), LangChain, custom `RedisVectorStore` (`services/redis_vectorstore.py`), RediSearch (FT.SEARCH with vector KNN).
- **Index schema:** HASH index with `$.content`, `$.content_vector`, etc. and `as_name="content"`, `as_name="content_vector"`.
- **Search:** Vector search uses `Query("(*)=>[KNN k @content_vector $query_vector AS vector_score]").return_fields("filename", "content", ...).dialect(2)`.
- **Result mapping:** We build LangChain `Document` from `results.docs` using `getattr(d, "content", None)` for page_content.

## Root cause (in codebase)

**HASH schema used JSON path `$.field` instead of plain field names.**  
RediSearch HASH indexes expect schema field names to match the hash keys exactly (e.g. `content`, `filename`). The code used `$.content`, `$.filename`, etc., which is JSON path syntax for JSON documents. That mismatch can cause indexing or RETURN to not bind to the actual hash fields, so FT.SEARCH returns docs but with empty content.

**Fix applied in codebase:**
- `services/redis_vectorstore.py`:
  - **create_index:** Schema changed from `name="$.content"` (and same for other fields) to **plain names** `name="content"`, `name="filename"`, etc., and removed `as_name` (not needed when name is the hash key).
  - **search:** Added `_get_doc_field()` so we read result fields from redis-py docs using both string and bytes keys and from `payload`/`vars(doc)`, and we normalize `page_number` to int.

**Important:** Existing indexes were created with the old schema. To get correct retrieval you must either:
1. **Recreate the index** (drop without deleting keys, then create again so it picks up the new schema), then re-index, or  
2. **Drop and re-upload** documents so the index is created with the new schema.  
Use `python scripts/check_rag_retrieval.py --fix-reindex <index_name>` to drop and recreate the index (keys are kept; you may need to re-add documents if the index was empty or you want a clean state).

## Other possible causes (if issue persists)

1. **Index prefix vs key prefix mismatch**
   - Index "zazi" may have been created with prefix **"za"** (so only keys `za:*` are indexed).
   - If documents were stored under **"zazi:*"**, they are **not** in the index; if under **"za:*"**, they are. Redis Insight shows the hash content but not necessarily which index covers that key.
   - **Check:** Run `FT.INFO <index_name>` and inspect `index_definition` → `prefixes`. Compare with actual key names (e.g. `za:*` vs `zazi:*`).

2. **redis-py mapping of RETURN fields** (mitigated by `_get_doc_field()` in search)
   - RediSearch returns `[total, doc_id, [field1, val1, field2, val2, ...], ...]`. redis-py maps these into a `Document`-like object.
   - If the client uses `decode_responses=False`, field names in the reply may be **bytes** (e.g. `b"content"`). Our code uses `getattr(d, "content", None)`; if the object stores attributes under byte keys or in a `payload` dict, we would get `None`.
   - **Check:** Log `vars(d)` or the raw `FT.SEARCH` reply (e.g. `FT.SEARCH index "*" RETURN 3 content filename content_vector LIMIT 0 1`) and see the exact key names and values returned.

3. **RETURN clause and schema attribute names**
   - Our schema uses **as_name** (e.g. `as_name="content"`). In FT.SEARCH, RETURN should use the **attribute name** (the name used in the index), which is the as_name. So `RETURN ... content ...` is correct for HASH fields indexed as `$.content` AS content.
   - Some sources suggest that for certain Redis/RediSearch versions or JSON path setups, RETURN might need to use the **path** (e.g. `$.content`) instead of the alias. If your Redis version behaves that way, returning by path could fix empty content.

4. **Redis / RediSearch version**
   - **Vector search:** Requires **DIALECT 2** (we already use `.dialect(2)`).
   - **RETURN:** The RETURN clause is standard across recent RediSearch versions; behavior should be consistent for HASH.
   - **Version compatibility:** RediSearch 2.6+ (Redis 6.0.16+) supports vector similarity; 2.8+ (Redis 7.2+) has RESP3 and other changes. In theory, RETURN field names could differ by version or by client (decode_responses, RESP2 vs RESP3). Worth confirming:
     - Redis server version: `redis-cli INFO server`
     - RediSearch module: `redis-cli MODULE LIST`
     - redis-py version and whether `decode_responses` is used when calling `ft().search()`.

## Does this depend on Redis / RediSearch version?

- **Partly.** Online findings suggest:
  - **Vector search** requires **DIALECT 2** (we use it); older dialect can yield no/wrong results.
  - **RETURN** behavior is standard in RediSearch; empty content is more often due to **schema path** (e.g. `$.` vs `$..`), **index prefix vs key prefix**, or **client parsing** (redis-py attribute names / decode_responses) than to server version alone.
  - **RediSearch 2.6+** (Redis 6.0.16+) supports vector search; **2.8+** (Redis 7.2+) has RESP3. Different Redis/redis-py versions could return response keys as bytes vs str, which would break `getattr(d, "content", None)` if the object only has `b"content"`.
- **Recommendation:** Confirm Redis server version (`redis-cli INFO server`), RediSearch module version (`MODULE LIST`), and redis-py version; if RETURN reply uses byte keys, normalize to string when building Documents.

## References (online)

- **redis-py FT.SEARCH not returning results / empty content:** Path/schema issues (e.g. `$..` vs `$.`) can lead to results with empty fields even when data exists in Redis. [SO: redis-py ft().search() not returning results](https://stackoverflow.com/questions/70399096/redis-py-ft-search-not-returning-results).
- **Vector search empty results:** Requires DIALECT 2; vector must be passed as binary in PARAMS; wrong encoding or dialect can yield no or wrong results. [SO: Unable to find results in vector search when using Redis as a vector database](https://stackoverflow.com/questions/78837936/unable-to-find-results-in-vector-search-when-using-redis-as-a-vector-database).
- **RediSearch RETURN:** RETURN uses field identifiers and optional AS alias; for HASH, specify the field name (or the alias used in the index). [Redis FT.SEARCH](https://redis.io/docs/latest/commands/ft.search/).
- **RediSearch version notes:** [RediSearch release notes](https://redis.io/docs/latest/operate/oss_and_stack/stack-with-enterprise/release-notes/redisearch) — ensure Redis/RediSearch versions meet minimum requirements for vector search and that client and server agree on response format.

## What to Do Next

1. **Verify index prefix vs keys**
   - Run: `python scripts/check_rag_retrieval.py <index_name>` and check "Index prefix(es)" and "Key counts by prefix".
   - If docs are under `zazi:*` but index has prefix `za`, either re-index under `za:*` or drop and recreate the index with prefix `zazi` (e.g. `python scripts/check_rag_retrieval.py --fix-reindex zazi`).

2. **Inspect raw FT.SEARCH reply and redis-py Document**
   - In `scripts/check_rag_retrieval.py`, the script already does a raw `FT.SEARCH` and prints field names in the reply. Confirm that "content" (or `b"content"`) is present and has non-empty values.
   - Log the full `vars(d)` for one result doc in `RedisVectorStore.search` to see where redis-py puts `content` (attribute name vs bytes, or inside `payload`).

3. **Fallback: read content by key if RETURN is empty**
   - If FT.SEARCH returns doc ids but empty RETURN fields, as a workaround we could **HGETALL** the document key (the doc id) inside `RedisVectorStore.search` and use that hash to build `page_content` and metadata. This would confirm that the issue is RETURN/parsing rather than missing data.

4. **Record Redis/RediSearch versions**
   - Add to this doc or to runbooks: Redis server version, RediSearch module version, redis-py version, and whether the Redis client uses `decode_responses=True` anywhere when executing search.

## Diagnostic Script

- **Location:** `scripts/check_rag_retrieval.py`
- **Usage:**
  - `python scripts/check_rag_retrieval.py [index_name]` — list indexes, FT.INFO, key counts by prefix, raw FT.SEARCH result, and project RedisVectorStore search.
  - `python scripts/check_rag_retrieval.py --fix-reindex <index_name>` — drop index (keep keys) and recreate with `prefix=index_name` so keys `index_name:*` are indexed.

## File References

- `services/redis_vectorstore.py` — `create_index` (schema with `$.content` / as_name), `search` (vector query, return_fields, mapping to Document).
- `rag/tools.py` — RAG tool using retriever from `load_vector_store` (in `app.utils`).
- `app/utils.py` — `load_vector_store` building RedisVectorStore/retriever for given index.





from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List
import uuid
from loguru import logger
from pydantic import BaseModel, ConfigDict
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, NumericField, VectorField
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
import numpy as np
import yaml
from contextlib import asynccontextmanager
from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple, Union

import redis.asyncio as redis_async


from app.config import settings

try:
    import zstd  # type: ignore

    ZSTD_AVAILABLE = True
except ImportError:
    zstd = None
    ZSTD_AVAILABLE = False

try:
    import orjson  # type: ignore

    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None
    ORJSON_AVAILABLE = False



class RedisVectorStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: redis.Redis
    embedding_func: Embeddings

    @classmethod
    def from_connecting_string(
        cls, redis_url: str, embedding_func: Embeddings
    ) -> RedisVectorStore:
        client: redis.Redis = redis.Redis.from_url(redis_url)  # type: ignore
        return cls(client=client, embedding_func=embedding_func)

    def create_index(self, index_name: str, prefix: str | None = None) -> None:
        prefix = prefix or index_name
        schema = (
            TextField(name=f"$.content", no_stem=True, as_name="content"),
            TagField(name=f"$.filename", as_name="filename"),
            NumericField(name=f"$.created_at", sortable=True, as_name="created_at"),
            NumericField(name=f"$.modified", sortable=True, as_name="modified"),
            TagField(name=f"$.type", as_name="type"),
            NumericField(name=f"$.page_number", sortable=True, as_name="page_number"),
            VectorField(
                name=f"$.content_vector",
                algorithm="FLAT",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="content_vector",
            ),
        )
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)  # type: ignore

        try:
            self.client.ft(index_name).info()  # type: ignore[no-untyped-call]
            logger.info(f"Index {index_name} already exists")
        except redis.exceptions.ResponseError:
            self.client.ft(index_name).create_index(  # type: ignore[no-untyped-call]
                fields=schema, definition=definition
            )

    @classmethod
    def load_schema(cls, schema_path: str = "services/schema.yml") -> Dict[str, Any]:  # type: ignore
        path = os.path.join(os.path.dirname(__file__), schema_path)
        with open(path) as f:
            schema: Dict[str, Any] = yaml.safe_load(f)
        return schema

    def list_indexes(self) -> Any:
        indexes = self.client.execute_command("FT._LIST")  # type: ignore[no-untyped-call]
        return indexes

    def delete_index(self, index_name: str) -> None:
        try:
            if self.client.ft(index_name).info():  # type: ignore[no-untyped-call]
                self.client.execute_command("FT.DROPINDEX", index_name, "DD")  # type: ignore[no-untyped-call]
                logger.info(
                    f"Index {index_name} and all associated documents have been deleted"
                )
            else:
                logger.error(f"Index {index_name} does not exist")
        except Exception as e:
            logger.error(f"An error occurred while deleting index {index_name}: {e}")

    async def index_documents(
        self,
        index_name: str,
        prefix: str | None = None,
        *,
        documents: List[Document],
    ) -> List[str]:
        prefix = prefix or index_name
        try:
            self.client.ft(index_name).info()  # type: ignore[no-untyped-call]
        except redis.exceptions.ResponseError:
            self.create_index(index_name, prefix)

        async def insert(document: Document, vector: np.ndarray) -> str:
            if doc_id := document.id:
                id_ = f"{prefix}:{doc_id}"
            else:
                id_ = f"{prefix}:{uuid.uuid4()}"
            payload = {
                "filename": document.metadata.get("filename", ""),
                "created_at": document.metadata.get(
                    "created_at", datetime.now().timestamp()
                ),
                "modified": document.metadata.get(
                    "modified", datetime.now().timestamp()
                ),
                "type": document.metadata.get("type"),
                "page_number": document.metadata.get("page_number", 0),
                "content": document.page_content,
                "content_vector": vector.tobytes(),
            }
            _ = await asyncio.to_thread(self.client.hset, name=id_, mapping=payload)
            return id_

        embedding_vectors = await self.embedding_func.aembed_documents(
            [doc.page_content for doc in documents]
        )
        embedding_arr = np.array(embedding_vectors, dtype=np.float32)
        return await asyncio.gather(
            *(insert(doc, vector) for doc, vector in zip(documents, embedding_arr))
        )

    async def search(
        self, index_name: str, query: str, k: int = 5, metadata: dict | None = None
    ) -> List[Document]:
        """Vector search; returns list of Document (content + metadata)."""
        metadata = metadata or {}
        query_vector = await self.embedding_func.aembed_query(query)
        r_query = (
            Query(f"(*)=>[KNN {k} @content_vector $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields("filename", "content", "created_at", "modified", "type", "page_number")
            .dialect(2)
        )
        results = await asyncio.to_thread(
            self.client.ft(index_name).search,  # type: ignore[no-untyped-call]
            r_query,
            {"query_vector": np.array(query_vector, dtype=np.float32).tobytes()},
        )
        docs = []
        for d in results.docs:
            content = getattr(d, "content", None) or getattr(d, "content_vector", "") or ""
            meta = {"filename": getattr(d, "filename", ""), "type": getattr(d, "type", ""), "page_number": getattr(d, "page_number", 0)}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def as_retriever(
        self,
        index_name: str,
        key_prefix: str | None = None,
        search_kwargs: dict | None = None,
    ) -> "RedisVectorStoreRetriever":
        """Return a retriever that uses this store's search (for RAG)."""
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 5)
        return RedisVectorStoreRetriever(
            vectorstore=self,
            index_name=index_name,
            key_prefix=key_prefix or index_name,
            k=k,
        )


class RedisVectorStoreRetriever(BaseRetriever):
    """Retriever that uses RedisVectorStore.search (async)."""

    vectorstore: RedisVectorStore
    index_name: str
    key_prefix: str
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        # Sync fallback: run async search in event loop
        return asyncio.run(self.vectorstore.search(self.index_name, query, k=self.k))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return await self.vectorstore.search(self.index_name, query, k=self.k)



class RedisSerializer:
    """Optimized serialization/deserialization service with multiple formats."""

    @staticmethod
    def serialize_json(data: Any) -> bytes:
        """Serialize data using optimized JSON."""
        if ORJSON_AVAILABLE and orjson:
            return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS)
        else:
            return json.dumps(data).encode("utf-8")

    @staticmethod
    def deserialize_json(data: bytes) -> Any:
        """Deserialize JSON data."""
        if ORJSON_AVAILABLE and orjson:
            return orjson.loads(data)
        else:
            return json.loads(data.decode("utf-8"))

    @staticmethod
    def compress_zstd(data: bytes, level: int = 3) -> bytes:
        """Compress data using ZSTD."""
        if ZSTD_AVAILABLE and zstd:
            return zstd.compress(data, level)
        else:
            return data

    @staticmethod
    def decompress_zstd(data: bytes) -> bytes:
        """Decompress ZSTD data."""
        if ZSTD_AVAILABLE and zstd:
            return zstd.decompress(data)
        else:
            return data

    @staticmethod
    def serialize_compressed(
        data: Any, method: str = "json", compression: str = "zstd"
    ) -> bytes:
        """Serialize and compress data."""
        if method == "json":
            serialized = RedisSerializer.serialize_json(data)
        else:
            raise ValueError(f"Unsupported serialization method: {method}")

        if compression == "zstd":
            return RedisSerializer.compress_zstd(serialized)
        else:
            return serialized

    @staticmethod
    def deserialize_compressed(
        data: bytes, method: str = "json", compression: str = "zstd"
    ) -> Any:
        """Decompress and deserialize data."""
        if compression == "zstd":
            decompressed = RedisSerializer.decompress_zstd(data)
        else:
            decompressed = data

        if method == "json":
            return RedisSerializer.deserialize_json(decompressed)
        else:
            raise ValueError(f"Unsupported serialization method: {method}")


class RedisConnectionPool:
    """Optimized Redis connection pool manager."""

    def __init__(self, redis_url: str, max_connections: int = 20):
        url_parts = redis_url.replace("redis://", "").split("@")

        if len(url_parts) == 2:
            auth_part, host_part = url_parts
            username, password = (
                auth_part.split(":") if ":" in auth_part else (None, auth_part)
            )
            host, port_db = host_part.split(":")
            port, db = port_db.split("/") if "/" in port_db else (port_db, "0")
        else:
            host_port_db = url_parts[0]
            host, port_db = host_port_db.split(":")
            port, db = port_db.split("/") if "/" in port_db else (port_db, "0")
            username = None
            password = None

        self.pool = redis_async.ConnectionPool(
            host=host,
            port=int(port),
            db=int(db),
            username=username,
            password=password,
            max_connections=max_connections,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

    def get_client(self) -> redis_async.Redis:
        """Get an async Redis client from the pool."""
        return redis_async.Redis(connection_pool=self.pool)


class RedisService:
    """High-performance Redis service with serialization, bulk operations, and optimizations."""

    def __init__(
        self,
        redis_url: str,
        max_connections: int = 20,
        enable_serialization: bool = True,
    ):
        self.pool = RedisConnectionPool(redis_url, max_connections)
        self.serializer = RedisSerializer() if enable_serialization else None
        self.enable_serialization = enable_serialization

    def _get_client(self) -> redis_async.Redis:
        """Get async Redis client from pool."""
        return self.pool.get_client()

    @asynccontextmanager
    async def _get_pipeline(self, transaction: bool = False):
        """Get optimized Redis pipeline."""
        client = self._get_client()
        async with client.pipeline(transaction=transaction) as pipe:
            yield pipe

    def _should_serialize(self, value: Any) -> bool:
        """Check if value should be serialized."""
        return self.enable_serialization and not isinstance(
            value, (str, int, float, bool)
        )

    def _serialize_value(
        self, value: Any, method: str = "json", compression: str = "zstd"
    ) -> Union[str, bytes]:
        """Serialize value if needed."""
        if not self._should_serialize(value):
            return str(value) if not isinstance(value, str) else value
        if self.serializer:
            return self.serializer.serialize_compressed(value, method, compression)
        return str(value)

    def _deserialize_value(
        self, value: Union[str, bytes], method: str = "json", compression: str = "zstd"
    ) -> Any:
        """Deserialize value if needed."""
        if isinstance(value, bytes) and self.serializer:
            try:
                return self.serializer.deserialize_compressed(
                    value, method, compression
                )
            except Exception:
                # Fallback to raw value if deserialization fails
                pass
        return value

    # Basic Operations
    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set key with optional serialization and expiration."""
        serialized_value = self._serialize_value(value, serialize_method, compression)

        async with self._get_client() as client:
            try:
                return await client.set(
                    key, serialized_value, ex=ex, px=px, nx=nx, xx=xx
                )
            except Exception as e:
                logger.error(f"Redis SET failed for key {key}: {str(e)}")
                return False

    async def get(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Optional[Any]:
        """Get key with optional deserialization."""
        async with self._get_client() as client:
            try:
                value = await client.get(key)  # type: ignore
                if value is None:
                    return None
                return self._deserialize_value(value, deserialize_method, decompression)
            except Exception as e:
                logger.error(f"Redis GET failed for key {key}: {str(e)}")
                return None

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not keys:
            return 0

        async with self._get_client() as client:
            try:
                return await client.delete(*keys)  # type: ignore
            except Exception as e:
                logger.error(f"Redis DELETE failed: {str(e)}")
                return 0

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not keys:
            return 0

        async with self._get_client() as client:
            try:
                return await client.exists(*keys)  # type: ignore
            except Exception as e:
                logger.error(f"Redis EXISTS failed: {str(e)}")
                return 0

    async def expire(
        self,
        key: str,
        time: int,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Set expiration on key."""
        async with self._get_client() as client:
            try:
                return await client.expire(key, time, nx=nx, xx=xx, gt=gt, lt=lt)  # type: ignore
            except Exception as e:
                logger.error(f"Redis EXPIRE failed for key {key}: {str(e)}")
                return False

    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        async with self._get_client() as client:
            try:
                return await client.ttl(key)  # type: ignore
            except Exception as e:
                logger.error(f"Redis TTL failed for key {key}: {str(e)}")
                return -2

    async def mset(
        self,
        mapping: Dict[str, Any],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set multiple keys at once."""
        if not mapping:
            return True

        serialized_mapping = {
            key: self._serialize_value(value, serialize_method, compression)
            for key, value in mapping.items()
        }

        async with self._get_client() as client:
            try:
                return await client.mset(serialized_mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis MSET failed: {str(e)}")
                return False

    async def mget(
        self,
        keys: List[str],
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> List[Optional[Any]]:
        """Get multiple keys at once."""
        if not keys:
            return []

        async with self._get_client() as client:
            try:
                values = await client.mget(keys)  # type: ignore
                return [
                    self._deserialize_value(value, deserialize_method, decompression)
                    if value is not None
                    else None
                    for value in values
                ]
            except Exception as e:
                logger.error(f"Redis MGET failed: {str(e)}")
                return [None] * len(keys)

    async def bulk_set_with_pipeline(
        self,
        items: Dict[str, Any],
        chunk_size: int = 1000,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Bulk set using optimized pipeline."""
        if not items:
            return True

        try:
            async with self._get_pipeline() as pipe:
                for i, (key, value) in enumerate(items.items()):
                    serialized_value = self._serialize_value(
                        value, serialize_method, compression
                    )
                    pipe.set(key, serialized_value)

                    # Execute in chunks to avoid memory issues
                    if (i + 1) % chunk_size == 0:
                        await pipe.execute()

                # Execute remaining commands
                if len(items) % chunk_size != 0:
                    await pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Redis bulk SET failed: {str(e)}")
            return False

    async def bulk_get_with_pipeline(
        self,
        keys: List[str],
        chunk_size: int = 1000,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Dict[str, Optional[Any]]:
        """Bulk get using optimized pipeline."""
        if not keys:
            return {}

        result = {}
        try:
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i : i + chunk_size]

                async with self._get_pipeline() as pipe:
                    for key in chunk_keys:
                        pipe.get(key)

                    values = await pipe.execute()

                for key, value in zip(chunk_keys, values):
                    result[key] = (
                        self._deserialize_value(
                            value, deserialize_method, decompression
                        )
                        if value is not None
                        else None
                    )

            return result
        except Exception as e:
            logger.error(f"Redis bulk GET failed: {str(e)}")
            return dict.fromkeys(keys)

    # Hash Operations
    async def hset(
        self,
        key: str,
        mapping: Dict[str, Any],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Set hash fields."""
        serialized_mapping = {
            field: self._serialize_value(value, serialize_method, compression)
            for field, value in mapping.items()
        }

        async with self._get_client() as client:
            try:
                return await client.hset(key, mapping=serialized_mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis HSET failed for key {key}: {str(e)}")
                return 0

    async def hget(
        self,
        key: str,
        field: str,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Optional[Any]:
        """Get hash field."""
        async with self._get_client() as client:
            try:
                value = await client.hget(key, field)  # type: ignore
                if value is None:
                    return None
                return self._deserialize_value(value, deserialize_method, decompression)
            except Exception as e:
                logger.error(
                    f"Redis HGET failed for key {key}, field {field}: {str(e)}"
                )
                return None

    async def hgetall(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Dict[str, Any]:
        """Get all hash fields."""
        async with self._get_client() as client:
            try:
                data = await client.hgetall(key)  # type: ignore
                return {
                    field: self._deserialize_value(
                        value, deserialize_method, decompression
                    )
                    for field, value in data.items()
                }
            except Exception as e:
                logger.error(f"Redis HGETALL failed for key {key}: {str(e)}")
                return {}

    # List Operations
    async def lpush(
        self,
        key: str,
        *values: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Push values to list left."""
        serialized_values = [
            self._serialize_value(value, serialize_method, compression)
            for value in values
        ]

        async with self._get_client() as client:
            try:
                return await client.lpush(key, *serialized_values)  # type: ignore
            except Exception as e:
                logger.error(f"Redis LPUSH failed for key {key}: {str(e)}")
                return 0

    async def rpush(
        self,
        key: str,
        *values: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Push values to list right."""
        serialized_values = [
            self._serialize_value(value, serialize_method, compression)
            for value in values
        ]

        async with self._get_client() as client:
            try:
                return await client.rpush(key, *serialized_values)  # type: ignore
            except Exception as e:
                logger.error(f"Redis RPUSH failed for key {key}: {str(e)}")
                return 0

    async def lrange(
        self,
        key: str,
        start: int,
        end: int,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> List[Any]:
        """Get range from list."""
        async with self._get_client() as client:
            try:
                values = await client.lrange(key, start, end)  # type: ignore
                return [
                    self._deserialize_value(value, deserialize_method, decompression)
                    for value in values
                ]
            except Exception as e:
                logger.error(f"Redis LRANGE failed for key {key}: {str(e)}")
                return []

    # Set Operations
    async def sadd(
        self,
        key: str,
        *members: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Add members to set."""
        serialized_members = [
            self._serialize_value(member, serialize_method, compression)
            for member in members
        ]

        async with self._get_client() as client:
            try:
                return await client.sadd(key, *serialized_members)  # type: ignore
            except Exception as e:
                logger.error(f"Redis SADD failed for key {key}: {str(e)}")
                return 0

    async def smembers(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> set:  # type: ignore
        """Get all set members."""
        async with self._get_client() as client:
            try:
                members = await client.smembers(key)  # type: ignore
                return {
                    self._deserialize_value(member, deserialize_method, decompression)
                    for member in members
                }
            except Exception as e:
                logger.error(f"Redis SMEMBERS failed for key {key}: {str(e)}")
                return set()

    async def zadd(
        self,
        key: str,
        mapping: Dict[str, float],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Add members to sorted set."""
        async with self._get_client() as client:
            try:
                return await client.zadd(key, mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis ZADD failed for key {key}: {str(e)}")
                return 0

    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Get range from sorted set."""
        async with self._get_client() as client:
            try:
                result = await client.zrange(key, start, end, withscores=withscores)  # type: ignore
                if withscores:
                    return [
                        (
                            self._deserialize_value(
                                member, deserialize_method, decompression
                            ),
                            score,
                        )
                        for member, score in result
                    ]
                else:
                    return [
                        self._deserialize_value(
                            member, deserialize_method, decompression
                        )
                        for member in result
                    ]
            except Exception as e:
                logger.error(f"Redis ZRANGE failed for key {key}: {str(e)}")
                return []

    # Pub/Sub Operations
    async def publish(
        self,
        channel: str,
        message: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Publish message to channel."""
        serialized_message = self._serialize_value(
            message, serialize_method, compression
        )

        async with self._get_client() as client:
            try:
                return await client.publish(channel, serialized_message)  # type: ignore
            except Exception as e:
                logger.error(f"Redis PUBLISH failed for channel {channel}: {str(e)}")
                return 0

    # Cache Operations with TTL
    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set cache with TTL."""
        return await self.set(
            key,
            value,
            ex=ttl_seconds,
            serialize_method=serialize_method,
            compression=compression,
        )

    async def get_cache(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Optional[Any]:
        """Get cached value."""
        return await self.get(key, deserialize_method, decompression)

    async def set_cache_many(
        self,
        items: Dict[str, Tuple[Any, int]],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set multiple cache items with different TTLs."""
        try:
            async with self._get_pipeline() as pipe:
                for key, (value, ttl) in items.items():
                    serialized_value = self._serialize_value(
                        value, serialize_method, compression
                    )
                    pipe.set(key, serialized_value, ex=ttl)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis set_cache_many failed: {str(e)}")
            return False

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        async with self._get_client() as client:
            try:
                return await client.keys(pattern)  # type: ignore
            except Exception as e:
                logger.error(f"Redis KEYS failed for pattern {pattern}: {str(e)}")
                return []

    async def scan(
        self, cursor: int = 0, match: Optional[str] = None, count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Scan keys with cursor."""
        async with self._get_client() as client:
            try:
                return await client.scan(cursor, match=match, count=count)  # type: ignore
            except Exception as e:
                logger.error(f"Redis SCAN failed: {str(e)}")
                return (0, [])

    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get Redis info."""
        async with self._get_client() as client:
            try:
                info_data = await client.info(section)  # type: ignore
                return dict(info_data) if info_data else {}
            except Exception as e:
                logger.error(f"Redis INFO failed: {str(e)}")
                return {}

    async def flushdb(self, asynchronous: bool = False) -> bool:
        """Flush current database."""
        async with self._get_client() as client:
            try:
                return await client.flushdb(asynchronous=asynchronous)  # type: ignore
            except Exception as e:
                logger.error(f"Redis FLUSHDB failed: {str(e)}")
                return False

    async def check_health(self) -> Dict[str, Any]:
        """Health check with detailed information."""
        try:
            async with self._get_client() as client:
                start_time = time.time()
                pong = await client.ping()  # type: ignore
                ping_time = time.time() - start_time

                if pong:
                    info = await client.info()  # type: ignore
                    return {
                        "status": "healthy",
                        "message": "Redis connection and operations successful",
                        "ping_time_ms": round(ping_time * 1000, 2),
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "total_connections_received": info.get(
                            "total_connections_received", 0
                        ),
                    }
                else:
                    return {"status": "unhealthy", "message": "Redis ping failed"}
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Redis connection failed: {str(e)}",
            }

    async def get_memory_usage(self, key: str) -> Optional[int]:
        """Get memory usage of a key."""
        async with self._get_client() as client:
            try:
                return await client.memory_usage(key)  # type: ignore
            except Exception as e:
                logger.error(f"Redis MEMORY_USAGE failed for key {key}: {str(e)}")
                return None

    async def close(self):
        """Close connection pool."""
        try:
            await self.pool.pool.disconnect()
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


redis_service = RedisService(redis_url=settings.redis_url)


import mimetypes
from typing import IO, List, Union, Type
from loguru import logger
import requests
from starlette import datastructures
from fastapi import File, UploadFile
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from typing_extensions import Any, Dict
import nltk


from .base import BaseDocumentLoader, NamedBytesIO

from ._html import HTMLLoader
from .md import MarkdownLoader
from .ms_docx import DocxLoader
from .ms_excel import MSExcelLoader
from .ms_pptx import PowerPointLoader
from .pdf import PDFLoader
from .text import TextFileLoader

__all__ = [
    "PDFLoader",
    "TextFileLoader",
    "MarkdownLoader",
    "DocxLoader",
    "PowerPointLoader",
    "MSExcelLoader",
]


mime_types_to_loaders: Dict[str, Type[BaseDocumentLoader]] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
    "text/html": HTMLLoader,
    "text/markdown": MarkdownLoader,
    "text/plain": TextFileLoader,
    "text/csv": CSVLoader,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": MSExcelLoader,
    "application/vnd.ms-excel": MSExcelLoader,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PowerPointLoader,
    "application/pdf": PDFLoader,
}


class DocumentLoader(BaseDocumentLoader):
    def __init__(
        self,
        sources: List[UploadFile] = File(...),
        extract_images: bool = False,
        extract_tables: bool = True,
        ):
        self.sources = sources
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        nltk.download("averaged_perceptron_tagger_eng")

    async def detect_document_type(
        self, source: Union[str, IO[bytes], datastructures.UploadFile]
    ) -> str:
        """
        Detect the MIME type of the document based on the file extension or magic number.
        """
        if isinstance(source, str):
            if self.is_url(source):
                response = requests.get(source)
                response.raise_for_status()
                return self._detect_by_magic_number(response.content[:4])
            else:
                # Check the file extension
                mime_type, _ = mimetypes.guess_type(source)
                if mime_type:
                    return mime_type
                else:
                    with open(source, "rb") as file:
                        return self._detect_by_magic_number(file.read(4))
        elif isinstance(source, NamedBytesIO):
            return source.mime_type  # type: ignore
        elif (
            isinstance(source, datastructures.UploadFile)
            and source.content_type is not None
        ):
            return source.content_type
        else:
            raise ValueError("Unsupported source type")

    def _detect_by_magic_number(self, signature: bytes) -> str:
        # Add more file signatures as needed
        signatures = {
            b"\x25\x50\x44\x46": "application/pdf",  # PDF
            b"\x50\x4b\x03\x04": "application/zip",  # ZIP (DOCX, XLSX, PPTX)
        }

        for sig, mime_type in signatures.items():
            if signature.startswith(sig):
                return mime_type
        return "application/octet-stream"

    def metadata(self, file: datastructures.UploadFile) -> Dict[str, Any]:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
        }

    async def load_async(self, source: datastructures.UploadFile) -> List[Document]:
        """
        Load the document asynchronously in async iterator.
        """
        mime_type = await self.detect_document_type(source)
        logger.debug(f"Detected MIME type: {mime_type}")
        if mime_type not in mime_types_to_loaders:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        else:
            loader = mime_types_to_loaders[mime_type]
            source_bytesio = NamedBytesIO(
                initial_bytes=await source.read(),
                name=source.filename,  # type: ignore
                metadata=self.metadata(source),
            )
            return await loader().load_async(source_bytesio)


from typing import Any, Dict, List, Literal, Optional, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores.redis import Redis, RedisTag, RedisFilter
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_core.vectorstores.base import VectorStoreRetriever
from loguru import logger
from app.config import settings
from app.utils import load_vector_store


class FileNameFilter(BaseModel):
    filename: str = Field(
        ...,
        description="The name of the file to filter the search results. The name should be an exact match.",
    )


class CreatedAtFilter(BaseModel):
    """
    A filter for the creation date of the document.
    """

    year: int | None = Field(None, description="The year of the document.")
    month: int | None = Field(None, description="The month of the document.")
    day: int | None = Field(None, description="The day the document.")
    operator: Literal["lt", "lte", "gt", "gte", "eq"] = Field(
        ..., description="The operator to use for comparison."
    )


class ModifiedFilter(CreatedAtFilter):
    """
    A filter for the modification date of the document.
    """

    pass


class PageNumberFilter(BaseModel):
    """
    A filter for the page number of the document.
    """

    page_number: int = Field(..., description="The page number of the document.")
    operator: Literal["lt", "lte", "gt", "gte", "eq"] = Field(
        ..., description="The operator to use for comparison."
    )


class TypeFilter(BaseModel):
    """
    A filter for the type of the document.
    """

    type: Literal["image", "text", "table"] = Field(
        ..., description="The type of the document."
    )


FILTERS = List[
    Union[FileNameFilter, CreatedAtFilter, ModifiedFilter, PageNumberFilter, TypeFilter]
]


class ToolInput(BaseModel):
    query: str = Field(
        ...,
        description="The user query to search for relevant information. Should be more comprehensive and search-friendly while maintaining the original intent.",
    )
    filters: Optional[FILTERS] = Field(
        None,
        description="Filters to apply to the search results. The filters should be used to narrow down the search results based on specific criteria.",
    )


def prepare_filters(filters: FILTERS) -> Any:
    metadata_filters = []
    for filter in filters:
        if isinstance(filter, FileNameFilter):
            metadata_filters.append(RedisFilter.tag("filename") == filter.filename)


class InformationLookupTool(BaseTool):
    name = "information_lookup_tool"
    description = "A tool for searching relevant information about a company based on a user query."
    args_schema: Type[BaseModel] | None = ToolInput
    embedding_model: str
    index_name: str
    key_prefix: str | None = None
    top_k: int = 5
    retriever: VectorStoreRetriever

    @root_validator(pre=True)
    def add_retriever(cls, values: Dict) -> Dict:
        if (embedding_model := values.get("embedding_model")) is None:
            raise ValueError("The 'embedding_model' argument is required.")
        if (index_name := values.get("index_name")) is None:
            raise ValueError("The 'index_name' argument is required.")

        # retriever = Redis.from_existing_index(
        #     redis_url=settings.redis_url,
        #     index_name=index_name,
        #     embedding=embedding,
        #     schema="backend/document/services/schema.yml",
        # ).as_retriever(search_kwargs={"k": values.get("top_k", 5)})
        retriever = load_vector_store(
            redis_url=settings.redis_url,
            index_name=index_name,
            key_prefix=values.get("key_prefix", index_name),
            embedding_model=embedding_model,
            existing=False,
        ).as_retriever(search_kwargs={"k": values.get("top_k", 5)})
        return {**values, "retriever": retriever}

    def _run(
        self,
        query: str,
        filters: Optional[FILTERS] = None,
    ) -> List:
        results = self.retriever.invoke(query)
        return [result.page_content for result in results]

    async def _arun(
        self,
        query: str,
        filters: Optional[FILTERS] = None,
    ) -> List:
        logger.info("Embedding: {}", self.embedding_model)
        results = await self.retriever.ainvoke(query)
        logger.info("Results: {}", results)
        return [result.page_content for result in results]
# def vectorstore(self, index_name: str, embedding: "Embeddings") -> "VectorStore":
#     vs = Redis.from_existing_index(
#         redis_url=settings.redis_url,
#         index_name=index_name,
#         embedding=embedding,
#         schema=INDEX_SECHEMA_PATH,
#     )
#     return vs


import mimetypes
from typing import IO, List, Union, Type
from loguru import logger
import requests
from starlette import datastructures
from fastapi import File, UploadFile
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from typing_extensions import Any, Dict
import nltk


from .base import BaseDocumentLoader, NamedBytesIO

from ._html import HTMLLoader
from .md import MarkdownLoader
from .ms_docx import DocxLoader
from .ms_excel import MSExcelLoader
from .ms_pptx import PowerPointLoader
from .pdf import PDFLoader
from .text import TextFileLoader

__all__ = [
    "PDFLoader",
    "TextFileLoader",
    "MarkdownLoader",
    "DocxLoader",
    "PowerPointLoader",
    "MSExcelLoader",
]


mime_types_to_loaders: Dict[str, Type[BaseDocumentLoader]] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
    "text/html": HTMLLoader,
    "text/markdown": MarkdownLoader,
    "text/plain": TextFileLoader,
    "text/csv": CSVLoader,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": MSExcelLoader,
    "application/vnd.ms-excel": MSExcelLoader,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PowerPointLoader,
    "application/pdf": PDFLoader,
}


class DocumentLoader(BaseDocumentLoader):
    def __init__(
        self,
        sources: List[UploadFile] = File(...),
        extract_images: bool = False,
        extract_tables: bool = True,
        ):
        self.sources = sources
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        nltk.download("averaged_perceptron_tagger_eng")

    async def detect_document_type(
        self, source: Union[str, IO[bytes], datastructures.UploadFile]
    ) -> str:
        """
        Detect the MIME type of the document based on the file extension or magic number.
        """
        if isinstance(source, str):
            if self.is_url(source):
                response = requests.get(source)
                response.raise_for_status()
                return self._detect_by_magic_number(response.content[:4])
            else:
                # Check the file extension
                mime_type, _ = mimetypes.guess_type(source)
                if mime_type:
                    return mime_type
                else:
                    with open(source, "rb") as file:
                        return self._detect_by_magic_number(file.read(4))
        elif isinstance(source, NamedBytesIO):
            return source.mime_type  # type: ignore
        elif (
            isinstance(source, datastructures.UploadFile)
            and source.content_type is not None
        ):
            return source.content_type
        else:
            raise ValueError("Unsupported source type")

    def _detect_by_magic_number(self, signature: bytes) -> str:
        # Add more file signatures as needed
        signatures = {
            b"\x25\x50\x44\x46": "application/pdf",  # PDF
            b"\x50\x4b\x03\x04": "application/zip",  # ZIP (DOCX, XLSX, PPTX)
        }

        for sig, mime_type in signatures.items():
            if signature.startswith(sig):
                return mime_type
        return "application/octet-stream"

    def metadata(self, file: datastructures.UploadFile) -> Dict[str, Any]:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
        }

    async def load_async(self, source: datastructures.UploadFile) -> List[Document]:
        """
        Load the document asynchronously in async iterator.
        """
        mime_type = await self.detect_document_type(source)
        logger.debug(f"Detected MIME type: {mime_type}")
        if mime_type not in mime_types_to_loaders:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        else:
            loader = mime_types_to_loaders[mime_type]
            source_bytesio = NamedBytesIO(
                initial_bytes=await source.read(),
                name=source.filename,  # type: ignore
                metadata=self.metadata(source),
            )
            return await loader().load_async(source_bytesio)

"""Vectorstore router: add/list/delete documents and indexes (from document/api.py)."""

import asyncio
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4
import os
from dateutil.parser import parse as dateutil_parse
from fastapi import Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.routing import APIRouter
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from database.models.data_source import DataSource
from database.models.user import User
from database.models.vectorstore import VectorStore
from database.models.vs_index import VectorIndex
from database.session import db_session_manager
from document.document_loaders import DocumentLoader
from document.document_loaders.base import ChunkType
from middleware.dep import get_current_user


def get_document_loader(
    files: List[UploadFile] = File(..., description="Files to index"),
    extract_images: bool = Form(False, description="Extract images from documents"),
    extract_tables: bool = Form(True, description="Extract tables"),
) -> DocumentLoader:
    """Dependency: inject DocumentLoader with request files and options (old API style: loader = Depends())."""
    return DocumentLoader(
        sources=files,
        extract_images=extract_images,
        extract_tables=extract_tables,
    )
from api.routes.base_di import BaseRouter
from api.decorator import action
from schemas.document.response import (
    AddDocumentResponse,
    DataSourceResponse,
    VectorStoreModel,
)
from services.redis_vectorstore import RedisVectorStore

import redis as redis_sync

_redis_url = settings.redis_url
_redis_client: redis_sync.Redis | None = None


def _get_redis_client() -> redis_sync.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_sync.Redis.from_url(_redis_url)
    return _redis_client


def _get_embedding(model: str = "text-embedding-3-small"):
    """Return embedding function for the given model name."""
    from langchain_openai import OpenAIEmbeddings

    api_key = getattr(settings, "openai_api_key", None) or ""
    if not api_key:
        raise ValueError(
            "OpenAI API key is not set. Add OPENAI_API_KEY to your .env or set openai_api_key in app config."
        )
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAIEmbeddings(model=model)


def _get_vector_store(
    index_name: str,
    key_prefix: str | None,
    embedding_model: str,
) -> RedisVectorStore:
    """Build RedisVectorStore for the given index and embedding model."""
    client = _get_redis_client()
    emb = _get_embedding(embedding_model)
    return RedisVectorStore(client=client, embedding_func=emb)  # type: ignore[arg-type]


class DocumentMetadata(BaseModel):
    """Metadata for a document chunk when adding to vectorstore."""

    filename: str
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    modified: float = Field(default_factory=lambda: datetime.now().timestamp())
    type: str  # ChunkType value
    page_number: int = 1

    @field_validator("created_at", "modified", mode="before")
    @classmethod
    def parse_dates(cls, value: Any) -> float:
        if value is None:
            return datetime.now().timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            if value.startswith("D:"):
                match = re.match(r"D:(\d{14})([+-])(\d{2})'(\d{2})'", value)
                if match:
                    dt_str, sign, hours, minutes = match.groups()
                    dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                    offset = int(hours) * 3600 + int(minutes) * 60
                    if sign == "-":
                        offset = -offset
                    return (dt - datetime.utcfromtimestamp(0)).total_seconds() - offset
            try:
                return dateutil_parse(value).timestamp()
            except ValueError:
                return datetime.now().timestamp()
        return datetime.now().timestamp()

    @field_validator("page_number", mode="before")
    @classmethod
    def parse_page_number(cls, value: Any) -> int:
        if value is None:
            return 1
        return int(value)


class VectorstoreRouter(BaseRouter):
    """Router for vectorstore: add/list/delete documents and indexes."""

    def __init__(self) -> None:
        router = APIRouter(prefix="/vectorstore", tags=["vectorstore"])
        super().__init__(router)

    @action(
        method="POST",
        url_path="add-document",
        response_model=AddDocumentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(get_current_user)],
    )
    async def add_document(
        self,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
        loader: DocumentLoader = Depends(get_document_loader),
        index_name: str = Form(..., description="Name of the vectorstore index"),
        embedding_model: str = Form("text-embedding-3-small", description="Embedding model name"),
        prefix: str | None = Form(None, description="Optional key prefix for Redis"),
        multimodal: bool = Form(False, description="Include image chunks in addition to text/table"),
    ) -> AddDocumentResponse:
        """Upload documents to a vectorstore index. DocumentLoader is injected via get_document_loader (old API style)."""
        try:

            async def get_or_create_index() -> tuple[VectorStore, bool]:
                q = (
                    select(VectorStore)
                    .options(selectinload(VectorStore.data_sources))
                    .where(
                        VectorStore.index_name == index_name,
                        VectorStore.user_id == current_user.id,
                    )
                )
                result = await db.execute(q)
                existing = result.scalar_one_or_none()
                if existing:
                    logger.info("Index %s already exists, using existing index", index_name)
                    return existing, False
                logger.info("Creating new index: %s", index_name)
                new_index = VectorStore(
                    index_name=index_name,
                    user_id=current_user.id,
                    embedding=str(embedding_model),
                    key_prefix=prefix,
                )
                db.add(new_index)
                await db.flush()
                return new_index, True

            async def check_duplicate_files(vs_data: VectorStore, is_new: bool) -> None:
                if is_new:
                    return
                existing_files = {s.title for s in vs_data.data_sources}
                new_files = {getattr(f, "filename", None) or "" for f in loader.sources}
                duplicates = existing_files & new_files
                if duplicates:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Documents already exist in the index: {', '.join(duplicates)}",
                    )

            logger.info("add_document: get_or_create_index for index_name={}", index_name)
            vs_data, is_new = await get_or_create_index()
            await check_duplicate_files(vs_data, is_new)
            logger.info("add_document: index ready, is_new={}, vs_data.id={}", is_new, vs_data.id)

            doc_ids: List[str] = []
            docs: List[List[Document]] = []
            async for doc_chunks in loader.lazy_load_async(loader.sources):
                doc_id = str(uuid4())
                doc_ids.append(doc_id)
                filtered = [
                    doc
                    for doc in doc_chunks
                    if multimodal or doc.metadata.get("type") in (ChunkType.text.value, ChunkType.table.value)
                ]
                chunk_list = []
                for i, doc in enumerate(filtered):
                    meta = dict(doc.metadata)
                    if meta.get("page_number") is None:
                        meta["page_number"] = i + 1
                    chunk_list.append(
                        Document(
                            page_content=doc.page_content,
                            metadata=DocumentMetadata(**meta).model_dump(mode="python"),
                        )
                    )
                docs.append(chunk_list)
            logger.info("add_document: loaded {} doc batches, {} total chunks", len(docs), sum(len(d) for d in docs))

            logger.info("add_document: getting vector store for index={}", vs_data.index_name)
            lc_redis = _get_vector_store(
                vs_data.index_name,
                vs_data.key_prefix,
                vs_data.embedding,
            )
            if is_new:
                try:
                    lc_redis.client.ping()
                except Exception:
                    pass
                lc_redis.create_index(vs_data.index_name, vs_data.key_prefix)
                logger.info("add_document: created Redis index {}", vs_data.index_name)

            logger.info("add_document: indexing {} batches into Redis", len(docs))
            vector_store_ids_nested = await asyncio.gather(
                *(
                    lc_redis.index_documents(
                        vs_data.index_name,
                        vs_data.key_prefix,
                        documents=doc_batch,
                    )
                    for doc_batch in docs
                )
            )
            logger.info("add_document: indexed, got {} chunk id lists", len(vector_store_ids_nested))

            new_data_sources: List[Any] = []
            for source, doc_id, chunk_ids in zip(loader.sources, doc_ids, vector_store_ids_nested):
                data_source = DataSource(
                    id=UUID(doc_id),
                    title=source.filename or "unknown",
                    url=getattr(source, "url", None),
                    mimetype=source.content_type,
                    user_id=current_user.id,
                    vector_store_id=vs_data.id,
                )
                new_data_sources.append(data_source)
                for chunk_id in chunk_ids:
                    part = chunk_id.split(":", 1)[-1]
                    new_data_sources.append(
                        VectorIndex(
                            id=UUID(part),
                            document_id=UUID(doc_id),
                            vector_store_id=vs_data.id,
                        )
                    )
            logger.info("add_document: prepared {} data sources + vector indexes, committing to DB", len(new_data_sources))

            db.add_all(new_data_sources)
            await db.commit()
            logger.info("add_document: commit done")

            flat_ids = [k for sub in vector_store_ids_nested for k in sub]
            return AddDocumentResponse(status="success", ids=flat_ids)

        except HTTPException:
            raise
        except PydanticValidationError as e:
            logger.error("add_document: validation error: {}", e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )
        except Exception as e:
            logger.exception("add_document failed: {}", e)
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="GET",
        url_path="list-documents/{index_name}",
        response_model=List[DataSourceResponse],
        dependencies=[Depends(get_current_user)],
    )
    async def list_documents(
        self,
        index_name: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[DataSourceResponse]:
        """List documents in a vectorstore index."""
        try:
            q = (
                select(DataSource)
                .where(
                    DataSource.vectorstore.has(VectorStore.index_name == index_name),
                    DataSource.user_id == current_user.id,
                )
                .order_by(DataSource.created_at.desc())
            )
            result = await db.execute(q)
            rows = result.scalars().all()
            return [DataSourceResponse.model_validate(r) for r in rows]
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="DELETE",
        url_path="document/{id}",
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(get_current_user)],
    )
    async def delete_document(
        self,
        id: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> Dict[str, str]:
        """Delete a document from the vectorstore."""
        try:
            q = (
                select(DataSource)
                .options(
                    selectinload(DataSource.vector_indexes),
                    selectinload(DataSource.vectorstore),
                )
                .where(DataSource.id == id, DataSource.user_id == current_user.id)
            )
            result = await db.execute(q)
            doc = result.unique().scalar_one_or_none()
            if not doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found",
                )
            prefix = doc.vectorstore.key_prefix or doc.vectorstore.index_name
            keys = [f"{prefix}:{vi.id}" for vi in doc.vector_indexes]
            lc_redis = _get_vector_store(
                doc.vectorstore.index_name,
                doc.vectorstore.key_prefix,
                doc.vectorstore.embedding,
            )
            if keys:
                await asyncio.to_thread(lc_redis.client.delete, *keys)
            await db.delete(doc)
            await db.commit()
            return {"status": "success"}
        except HTTPException:
            raise
        except Exception as e:
            await db.rollback()
            logger.error("Error deleting document: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="GET",
        url_path="list-indexes",
        response_model=List[VectorStoreModel],
        dependencies=[Depends(get_current_user)],
    )
    async def list_indexes(
        self,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[VectorStoreModel]:
        """List vectorstore indexes for the current user."""
        try:
            result = await db.execute(
                select(VectorStore).where(VectorStore.user_id == current_user.id)
            )
            rows = result.scalars().all()
            return [VectorStoreModel.model_validate(r) for r in rows]
        except Exception as e:
            logger.error("Error listing indexes: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="DELETE",
        url_path="delete-index/{id}",
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(get_current_user)],
    )
    async def delete_index(
        self,
        id: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> Dict[str, str]:
        """Delete a vectorstore index."""
        try:
            index = await db.get(VectorStore, UUID(id))
            if not index:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Index not found",
                )
            if index.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not allowed to delete this index",
                )
            index_name = index.index_name
            key_prefix = index.key_prefix
            embedding_model = index.embedding
            try:
                lc_redis = _get_vector_store(index_name, key_prefix, embedding_model)
                lc_redis.delete_index(index_name)
            except Exception as e:
                logger.error("Error deleting index from vector store: %r", e)
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error deleting index from vector store: {e!r}",
                )
            await db.delete(index)
            await db.commit()
            return {"status": "success"}
        except HTTPException:
            raise
        except Exception as e:
            await db.rollback()
            logger.error("Error deleting index: %r", e)
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List
import uuid
from loguru import logger
from pydantic import BaseModel, ConfigDict
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, NumericField, VectorField
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
import numpy as np
import yaml
from contextlib import asynccontextmanager
from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple, Union

import redis.asyncio as redis_async


from app.config import settings

try:
    import zstd  # type: ignore

    ZSTD_AVAILABLE = True
except ImportError:
    zstd = None
    ZSTD_AVAILABLE = False

try:
    import orjson  # type: ignore

    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None
    ORJSON_AVAILABLE = False



class RedisVectorStore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: redis.Redis
    embedding_func: Embeddings

    @classmethod
    def from_connecting_string(
        cls, redis_url: str, embedding_func: Embeddings
    ) -> RedisVectorStore:
        client: redis.Redis = redis.Redis.from_url(redis_url)  # type: ignore
        return cls(client=client, embedding_func=embedding_func)

    def create_index(self, index_name: str, prefix: str | None = None) -> None:
        prefix = prefix or index_name
        schema = (
            TextField(name=f"$.content", no_stem=True, as_name="content"),
            TagField(name=f"$.filename", as_name="filename"),
            NumericField(name=f"$.created_at", sortable=True, as_name="created_at"),
            NumericField(name=f"$.modified", sortable=True, as_name="modified"),
            TagField(name=f"$.type", as_name="type"),
            NumericField(name=f"$.page_number", sortable=True, as_name="page_number"),
            VectorField(
                name=f"$.content_vector",
                algorithm="FLAT",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="content_vector",
            ),
        )
        definition = IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)  # type: ignore

        try:
            self.client.ft(index_name).info()  # type: ignore[no-untyped-call]
            logger.info(f"Index {index_name} already exists")
        except redis.exceptions.ResponseError:
            self.client.ft(index_name).create_index(  # type: ignore[no-untyped-call]
                fields=schema, definition=definition
            )

    @classmethod
    def load_schema(cls, schema_path: str = "services/schema.yml") -> Dict[str, Any]:  # type: ignore
        path = os.path.join(os.path.dirname(__file__), schema_path)
        with open(path) as f:
            schema: Dict[str, Any] = yaml.safe_load(f)
        return schema

    def list_indexes(self) -> Any:
        indexes = self.client.execute_command("FT._LIST")  # type: ignore[no-untyped-call]
        return indexes

    def delete_index(self, index_name: str) -> None:
        try:
            if self.client.ft(index_name).info():  # type: ignore[no-untyped-call]
                self.client.execute_command("FT.DROPINDEX", index_name, "DD")  # type: ignore[no-untyped-call]
                logger.info(
                    f"Index {index_name} and all associated documents have been deleted"
                )
            else:
                logger.error(f"Index {index_name} does not exist")
        except Exception as e:
            logger.error(f"An error occurred while deleting index {index_name}: {e}")

    async def index_documents(
        self,
        index_name: str,
        prefix: str | None = None,
        *,
        documents: List[Document],
    ) -> List[str]:
        prefix = prefix or index_name
        try:
            self.client.ft(index_name).info()  # type: ignore[no-untyped-call]
        except redis.exceptions.ResponseError:
            self.create_index(index_name, prefix)

        async def insert(document: Document, vector: np.ndarray) -> str:
            if doc_id := document.id:
                id_ = f"{prefix}:{doc_id}"
            else:
                id_ = f"{prefix}:{uuid.uuid4()}"
            payload = {
                "filename": document.metadata.get("filename", ""),
                "created_at": document.metadata.get(
                    "created_at", datetime.now().timestamp()
                ),
                "modified": document.metadata.get(
                    "modified", datetime.now().timestamp()
                ),
                "type": document.metadata.get("type"),
                "page_number": document.metadata.get("page_number", 0),
                "content": document.page_content,
                "content_vector": vector.tobytes(),
            }
            _ = await asyncio.to_thread(self.client.hset, name=id_, mapping=payload)
            return id_

        embedding_vectors = await self.embedding_func.aembed_documents(
            [doc.page_content for doc in documents]
        )
        embedding_arr = np.array(embedding_vectors, dtype=np.float32)
        return await asyncio.gather(
            *(insert(doc, vector) for doc, vector in zip(documents, embedding_arr))
        )

    async def search(
        self, index_name: str, query: str, k: int = 5, metadata: dict | None = None
    ) -> List[Document]:
        """Vector search; returns list of Document (content + metadata)."""
        metadata = metadata or {}
        query_vector = await self.embedding_func.aembed_query(query)
        r_query = (
            Query(f"(*)=>[KNN {k} @content_vector $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields("filename", "content", "created_at", "modified", "type", "page_number")
            .dialect(2)
        )
        results = await asyncio.to_thread(
            self.client.ft(index_name).search,  # type: ignore[no-untyped-call]
            r_query,
            {"query_vector": np.array(query_vector, dtype=np.float32).tobytes()},
        )
        docs = []
        for d in results.docs:
            content = getattr(d, "content", None) or getattr(d, "content_vector", "") or ""
            meta = {"filename": getattr(d, "filename", ""), "type": getattr(d, "type", ""), "page_number": getattr(d, "page_number", 0)}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def as_retriever(
        self,
        index_name: str,
        key_prefix: str | None = None,
        search_kwargs: dict | None = None,
    ) -> "RedisVectorStoreRetriever":
        """Return a retriever that uses this store's search (for RAG)."""
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 5)
        return RedisVectorStoreRetriever(
            vectorstore=self,
            index_name=index_name,
            key_prefix=key_prefix or index_name,
            k=k,
        )


class RedisVectorStoreRetriever(BaseRetriever):
    """Retriever that uses RedisVectorStore.search (async)."""

    vectorstore: RedisVectorStore
    index_name: str
    key_prefix: str
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        # Sync fallback: run async search in event loop
        return asyncio.run(self.vectorstore.search(self.index_name, query, k=self.k))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return await self.vectorstore.search(self.index_name, query, k=self.k)



class RedisSerializer:
    """Optimized serialization/deserialization service with multiple formats."""

    @staticmethod
    def serialize_json(data: Any) -> bytes:
        """Serialize data using optimized JSON."""
        if ORJSON_AVAILABLE and orjson:
            return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS)
        else:
            return json.dumps(data).encode("utf-8")

    @staticmethod
    def deserialize_json(data: bytes) -> Any:
        """Deserialize JSON data."""
        if ORJSON_AVAILABLE and orjson:
            return orjson.loads(data)
        else:
            return json.loads(data.decode("utf-8"))

    @staticmethod
    def compress_zstd(data: bytes, level: int = 3) -> bytes:
        """Compress data using ZSTD."""
        if ZSTD_AVAILABLE and zstd:
            return zstd.compress(data, level)
        else:
            return data

    @staticmethod
    def decompress_zstd(data: bytes) -> bytes:
        """Decompress ZSTD data."""
        if ZSTD_AVAILABLE and zstd:
            return zstd.decompress(data)
        else:
            return data

    @staticmethod
    def serialize_compressed(
        data: Any, method: str = "json", compression: str = "zstd"
    ) -> bytes:
        """Serialize and compress data."""
        if method == "json":
            serialized = RedisSerializer.serialize_json(data)
        else:
            raise ValueError(f"Unsupported serialization method: {method}")

        if compression == "zstd":
            return RedisSerializer.compress_zstd(serialized)
        else:
            return serialized

    @staticmethod
    def deserialize_compressed(
        data: bytes, method: str = "json", compression: str = "zstd"
    ) -> Any:
        """Decompress and deserialize data."""
        if compression == "zstd":
            decompressed = RedisSerializer.decompress_zstd(data)
        else:
            decompressed = data

        if method == "json":
            return RedisSerializer.deserialize_json(decompressed)
        else:
            raise ValueError(f"Unsupported serialization method: {method}")


class RedisConnectionPool:
    """Optimized Redis connection pool manager."""

    def __init__(self, redis_url: str, max_connections: int = 20):
        url_parts = redis_url.replace("redis://", "").split("@")

        if len(url_parts) == 2:
            auth_part, host_part = url_parts
            username, password = (
                auth_part.split(":") if ":" in auth_part else (None, auth_part)
            )
            host, port_db = host_part.split(":")
            port, db = port_db.split("/") if "/" in port_db else (port_db, "0")
        else:
            host_port_db = url_parts[0]
            host, port_db = host_port_db.split(":")
            port, db = port_db.split("/") if "/" in port_db else (port_db, "0")
            username = None
            password = None

        self.pool = redis_async.ConnectionPool(
            host=host,
            port=int(port),
            db=int(db),
            username=username,
            password=password,
            max_connections=max_connections,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

    def get_client(self) -> redis_async.Redis:
        """Get an async Redis client from the pool."""
        return redis_async.Redis(connection_pool=self.pool)


class RedisService:
    """High-performance Redis service with serialization, bulk operations, and optimizations."""

    def __init__(
        self,
        redis_url: str,
        max_connections: int = 20,
        enable_serialization: bool = True,
    ):
        self.pool = RedisConnectionPool(redis_url, max_connections)
        self.serializer = RedisSerializer() if enable_serialization else None
        self.enable_serialization = enable_serialization

    def _get_client(self) -> redis_async.Redis:
        """Get async Redis client from pool."""
        return self.pool.get_client()

    @asynccontextmanager
    async def _get_pipeline(self, transaction: bool = False):
        """Get optimized Redis pipeline."""
        client = self._get_client()
        async with client.pipeline(transaction=transaction) as pipe:
            yield pipe

    def _should_serialize(self, value: Any) -> bool:
        """Check if value should be serialized."""
        return self.enable_serialization and not isinstance(
            value, (str, int, float, bool)
        )

    def _serialize_value(
        self, value: Any, method: str = "json", compression: str = "zstd"
    ) -> Union[str, bytes]:
        """Serialize value if needed."""
        if not self._should_serialize(value):
            return str(value) if not isinstance(value, str) else value
        if self.serializer:
            return self.serializer.serialize_compressed(value, method, compression)
        return str(value)

    def _deserialize_value(
        self, value: Union[str, bytes], method: str = "json", compression: str = "zstd"
    ) -> Any:
        """Deserialize value if needed."""
        if isinstance(value, bytes) and self.serializer:
            try:
                return self.serializer.deserialize_compressed(
                    value, method, compression
                )
            except Exception:
                # Fallback to raw value if deserialization fails
                pass
        return value

    # Basic Operations
    async def set(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set key with optional serialization and expiration."""
        serialized_value = self._serialize_value(value, serialize_method, compression)

        async with self._get_client() as client:
            try:
                return await client.set(
                    key, serialized_value, ex=ex, px=px, nx=nx, xx=xx
                )
            except Exception as e:
                logger.error(f"Redis SET failed for key {key}: {str(e)}")
                return False

    async def get(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Optional[Any]:
        """Get key with optional deserialization."""
        async with self._get_client() as client:
            try:
                value = await client.get(key)  # type: ignore
                if value is None:
                    return None
                return self._deserialize_value(value, deserialize_method, decompression)
            except Exception as e:
                logger.error(f"Redis GET failed for key {key}: {str(e)}")
                return None

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not keys:
            return 0

        async with self._get_client() as client:
            try:
                return await client.delete(*keys)  # type: ignore
            except Exception as e:
                logger.error(f"Redis DELETE failed: {str(e)}")
                return 0

    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        if not keys:
            return 0

        async with self._get_client() as client:
            try:
                return await client.exists(*keys)  # type: ignore
            except Exception as e:
                logger.error(f"Redis EXISTS failed: {str(e)}")
                return 0

    async def expire(
        self,
        key: str,
        time: int,
        nx: bool = False,
        xx: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> bool:
        """Set expiration on key."""
        async with self._get_client() as client:
            try:
                return await client.expire(key, time, nx=nx, xx=xx, gt=gt, lt=lt)  # type: ignore
            except Exception as e:
                logger.error(f"Redis EXPIRE failed for key {key}: {str(e)}")
                return False

    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        async with self._get_client() as client:
            try:
                return await client.ttl(key)  # type: ignore
            except Exception as e:
                logger.error(f"Redis TTL failed for key {key}: {str(e)}")
                return -2

    async def mset(
        self,
        mapping: Dict[str, Any],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set multiple keys at once."""
        if not mapping:
            return True

        serialized_mapping = {
            key: self._serialize_value(value, serialize_method, compression)
            for key, value in mapping.items()
        }

        async with self._get_client() as client:
            try:
                return await client.mset(serialized_mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis MSET failed: {str(e)}")
                return False

    async def mget(
        self,
        keys: List[str],
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> List[Optional[Any]]:
        """Get multiple keys at once."""
        if not keys:
            return []

        async with self._get_client() as client:
            try:
                values = await client.mget(keys)  # type: ignore
                return [
                    self._deserialize_value(value, deserialize_method, decompression)
                    if value is not None
                    else None
                    for value in values
                ]
            except Exception as e:
                logger.error(f"Redis MGET failed: {str(e)}")
                return [None] * len(keys)

    async def bulk_set_with_pipeline(
        self,
        items: Dict[str, Any],
        chunk_size: int = 1000,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Bulk set using optimized pipeline."""
        if not items:
            return True

        try:
            async with self._get_pipeline() as pipe:
                for i, (key, value) in enumerate(items.items()):
                    serialized_value = self._serialize_value(
                        value, serialize_method, compression
                    )
                    pipe.set(key, serialized_value)

                    # Execute in chunks to avoid memory issues
                    if (i + 1) % chunk_size == 0:
                        await pipe.execute()

                # Execute remaining commands
                if len(items) % chunk_size != 0:
                    await pipe.execute()

            return True
        except Exception as e:
            logger.error(f"Redis bulk SET failed: {str(e)}")
            return False

    async def bulk_get_with_pipeline(
        self,
        keys: List[str],
        chunk_size: int = 1000,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Dict[str, Optional[Any]]:
        """Bulk get using optimized pipeline."""
        if not keys:
            return {}

        result = {}
        try:
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i : i + chunk_size]

                async with self._get_pipeline() as pipe:
                    for key in chunk_keys:
                        pipe.get(key)

                    values = await pipe.execute()

                for key, value in zip(chunk_keys, values):
                    result[key] = (
                        self._deserialize_value(
                            value, deserialize_method, decompression
                        )
                        if value is not None
                        else None
                    )

            return result
        except Exception as e:
            logger.error(f"Redis bulk GET failed: {str(e)}")
            return dict.fromkeys(keys)

    # Hash Operations
    async def hset(
        self,
        key: str,
        mapping: Dict[str, Any],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Set hash fields."""
        serialized_mapping = {
            field: self._serialize_value(value, serialize_method, compression)
            for field, value in mapping.items()
        }

        async with self._get_client() as client:
            try:
                return await client.hset(key, mapping=serialized_mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis HSET failed for key {key}: {str(e)}")
                return 0

    async def hget(
        self,
        key: str,
        field: str,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Optional[Any]:
        """Get hash field."""
        async with self._get_client() as client:
            try:
                value = await client.hget(key, field)  # type: ignore
                if value is None:
                    return None
                return self._deserialize_value(value, deserialize_method, decompression)
            except Exception as e:
                logger.error(
                    f"Redis HGET failed for key {key}, field {field}: {str(e)}"
                )
                return None

    async def hgetall(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Dict[str, Any]:
        """Get all hash fields."""
        async with self._get_client() as client:
            try:
                data = await client.hgetall(key)  # type: ignore
                return {
                    field: self._deserialize_value(
                        value, deserialize_method, decompression
                    )
                    for field, value in data.items()
                }
            except Exception as e:
                logger.error(f"Redis HGETALL failed for key {key}: {str(e)}")
                return {}

    # List Operations
    async def lpush(
        self,
        key: str,
        *values: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Push values to list left."""
        serialized_values = [
            self._serialize_value(value, serialize_method, compression)
            for value in values
        ]

        async with self._get_client() as client:
            try:
                return await client.lpush(key, *serialized_values)  # type: ignore
            except Exception as e:
                logger.error(f"Redis LPUSH failed for key {key}: {str(e)}")
                return 0

    async def rpush(
        self,
        key: str,
        *values: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Push values to list right."""
        serialized_values = [
            self._serialize_value(value, serialize_method, compression)
            for value in values
        ]

        async with self._get_client() as client:
            try:
                return await client.rpush(key, *serialized_values)  # type: ignore
            except Exception as e:
                logger.error(f"Redis RPUSH failed for key {key}: {str(e)}")
                return 0

    async def lrange(
        self,
        key: str,
        start: int,
        end: int,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> List[Any]:
        """Get range from list."""
        async with self._get_client() as client:
            try:
                values = await client.lrange(key, start, end)  # type: ignore
                return [
                    self._deserialize_value(value, deserialize_method, decompression)
                    for value in values
                ]
            except Exception as e:
                logger.error(f"Redis LRANGE failed for key {key}: {str(e)}")
                return []

    # Set Operations
    async def sadd(
        self,
        key: str,
        *members: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Add members to set."""
        serialized_members = [
            self._serialize_value(member, serialize_method, compression)
            for member in members
        ]

        async with self._get_client() as client:
            try:
                return await client.sadd(key, *serialized_members)  # type: ignore
            except Exception as e:
                logger.error(f"Redis SADD failed for key {key}: {str(e)}")
                return 0

    async def smembers(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> set:  # type: ignore
        """Get all set members."""
        async with self._get_client() as client:
            try:
                members = await client.smembers(key)  # type: ignore
                return {
                    self._deserialize_value(member, deserialize_method, decompression)
                    for member in members
                }
            except Exception as e:
                logger.error(f"Redis SMEMBERS failed for key {key}: {str(e)}")
                return set()

    async def zadd(
        self,
        key: str,
        mapping: Dict[str, float],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Add members to sorted set."""
        async with self._get_client() as client:
            try:
                return await client.zadd(key, mapping)  # type: ignore
            except Exception as e:
                logger.error(f"Redis ZADD failed for key {key}: {str(e)}")
                return 0

    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
        deserialize_method: str = "json",
        decompression: str = "zstd",
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Get range from sorted set."""
        async with self._get_client() as client:
            try:
                result = await client.zrange(key, start, end, withscores=withscores)  # type: ignore
                if withscores:
                    return [
                        (
                            self._deserialize_value(
                                member, deserialize_method, decompression
                            ),
                            score,
                        )
                        for member, score in result
                    ]
                else:
                    return [
                        self._deserialize_value(
                            member, deserialize_method, decompression
                        )
                        for member in result
                    ]
            except Exception as e:
                logger.error(f"Redis ZRANGE failed for key {key}: {str(e)}")
                return []

    # Pub/Sub Operations
    async def publish(
        self,
        channel: str,
        message: Any,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> int:
        """Publish message to channel."""
        serialized_message = self._serialize_value(
            message, serialize_method, compression
        )

        async with self._get_client() as client:
            try:
                return await client.publish(channel, serialized_message)  # type: ignore
            except Exception as e:
                logger.error(f"Redis PUBLISH failed for channel {channel}: {str(e)}")
                return 0

    # Cache Operations with TTL
    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set cache with TTL."""
        return await self.set(
            key,
            value,
            ex=ttl_seconds,
            serialize_method=serialize_method,
            compression=compression,
        )

    async def get_cache(
        self, key: str, deserialize_method: str = "json", decompression: str = "zstd"
    ) -> Optional[Any]:
        """Get cached value."""
        return await self.get(key, deserialize_method, decompression)

    async def set_cache_many(
        self,
        items: Dict[str, Tuple[Any, int]],
        serialize_method: str = "json",
        compression: str = "zstd",
    ) -> bool:
        """Set multiple cache items with different TTLs."""
        try:
            async with self._get_pipeline() as pipe:
                for key, (value, ttl) in items.items():
                    serialized_value = self._serialize_value(
                        value, serialize_method, compression
                    )
                    pipe.set(key, serialized_value, ex=ttl)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis set_cache_many failed: {str(e)}")
            return False

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        async with self._get_client() as client:
            try:
                return await client.keys(pattern)  # type: ignore
            except Exception as e:
                logger.error(f"Redis KEYS failed for pattern {pattern}: {str(e)}")
                return []

    async def scan(
        self, cursor: int = 0, match: Optional[str] = None, count: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Scan keys with cursor."""
        async with self._get_client() as client:
            try:
                return await client.scan(cursor, match=match, count=count)  # type: ignore
            except Exception as e:
                logger.error(f"Redis SCAN failed: {str(e)}")
                return (0, [])

    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get Redis info."""
        async with self._get_client() as client:
            try:
                info_data = await client.info(section)  # type: ignore
                return dict(info_data) if info_data else {}
            except Exception as e:
                logger.error(f"Redis INFO failed: {str(e)}")
                return {}

    async def flushdb(self, asynchronous: bool = False) -> bool:
        """Flush current database."""
        async with self._get_client() as client:
            try:
                return await client.flushdb(asynchronous=asynchronous)  # type: ignore
            except Exception as e:
                logger.error(f"Redis FLUSHDB failed: {str(e)}")
                return False

    async def check_health(self) -> Dict[str, Any]:
        """Health check with detailed information."""
        try:
            async with self._get_client() as client:
                start_time = time.time()
                pong = await client.ping()  # type: ignore
                ping_time = time.time() - start_time

                if pong:
                    info = await client.info()  # type: ignore
                    return {
                        "status": "healthy",
                        "message": "Redis connection and operations successful",
                        "ping_time_ms": round(ping_time * 1000, 2),
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "total_connections_received": info.get(
                            "total_connections_received", 0
                        ),
                    }
                else:
                    return {"status": "unhealthy", "message": "Redis ping failed"}
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Redis connection failed: {str(e)}",
            }

    async def get_memory_usage(self, key: str) -> Optional[int]:
        """Get memory usage of a key."""
        async with self._get_client() as client:
            try:
                return await client.memory_usage(key)  # type: ignore
            except Exception as e:
                logger.error(f"Redis MEMORY_USAGE failed for key {key}: {str(e)}")
                return None

    async def close(self):
        """Close connection pool."""
        try:
            await self.pool.pool.disconnect()
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


redis_service = RedisService(redis_url=settings.redis_url)

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
        """
        Create RediSearch index.

        IMPORTANT: For HASH indexes use plain field names (e.g. "content", "filename").
        JSON path syntax ($.field) is only for IndexType.JSON.
        """
        prefix = prefix or index_name
        # Plain field names for HASH index (must match hash keys exactly)
        schema = (
            TextField(name="content", no_stem=True),
            TagField(name="filename"),
            NumericField(name="created_at", sortable=True),
            NumericField(name="modified", sortable=True),
            TagField(name="type"),
            NumericField(name="page_number", sortable=True),
            TagField(name="sheet_name"),
            NumericField(name="row_number", sortable=True),
            VectorField(
                name="content_vector",
                algorithm="FLAT",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                },
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
                "sheet_name": document.metadata.get("sheet_name", ""),
                "row_number": document.metadata.get("row_number", 0),
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

    @staticmethod
    def _get_doc_field(doc: Any, field: str) -> Any:
        """
        Robust field extraction from RediSearch result document.
        Handles both string and bytes keys, and checks payload/vars(doc).
        Returns None if the field is not found or any error occurs.
        """
        try:
            value = getattr(doc, field, None)
            if value is not None:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            pass
        try:
            field_bytes = field.encode("utf-8") if isinstance(field, str) else field
            value = getattr(doc, field_bytes, None)
            if value is not None:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            pass
        try:
            if hasattr(doc, "payload") and isinstance(doc.payload, dict):
                value = doc.payload.get(field) or doc.payload.get(
                    field.encode("utf-8") if isinstance(field, str) else field
                )
                if value is not None:
                    return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            pass
        try:
            doc_vars = vars(doc)
            value = doc_vars.get(field) or doc_vars.get(
                field.encode("utf-8") if isinstance(field, str) else field
            )
            if value is not None:
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            pass
        return None

    async def search(
        self, index_name: str | None, query: str, k: int = 5, metadata: dict | None = None,
        user_index_names: List[str] | None = None,
    ) -> List[Document]:
        """
        Vector search; returns list of Document (content + metadata).

        Args:
            index_name: Name of the index to search. If None or empty, searches all available indexes.
            query: Search query string.
            k: Number of results to return.
            metadata: Optional metadata filters (currently unused but reserved for future use).
            user_index_names: Explicit list of index names to search when index_name is None.
                              Must be provided for multi-index search to scope results to the current user.

        Returns:
            List of Document objects sorted by relevance (vector_score).
        """
        metadata = metadata or {}

        # If no index_name provided, search across user-scoped indexes
        if not index_name:
            return await self._search_all_indexes(query, k, metadata, user_index_names=user_index_names)

        # Single index search (original behavior)
        return await self._search_single_index(index_name, query, k, metadata)
    
    async def _search_single_index(
        self, index_name: str, query: str, k: int = 5, metadata: dict | None = None
    ) -> List[Document]:
        """Search a single index."""
        query_vector = await self.embedding_func.aembed_query(query)
        r_query = (
            Query(f"(*)=>[KNN {k} @content_vector $query_vector AS vector_score]")
            .sort_by("vector_score")
            .return_fields("filename", "content", "created_at", "modified", "type", "page_number", "sheet_name", "row_number", "vector_score")
            .dialect(2)
        )
        results = await asyncio.to_thread(
            self.client.ft(index_name).search,  # type: ignore[no-untyped-call]
            r_query,
            {"query_vector": np.array(query_vector, dtype=np.float32).tobytes()},
        )
        logger.info(
            f"Search on index '{index_name}': returned {len(results.docs)} docs (total: {results.total})"
        )
        if len(results.docs) == 0 and results.total > 0:
            logger.warning(
                "Search found matches but returned no docs - possible RETURN field mismatch"
            )
        
        return self._parse_search_results(results)
    
    async def _search_all_indexes(
        self, query: str, k: int = 5, metadata: dict | None = None,
        user_index_names: List[str] | None = None,
    ) -> List[Document]:
        """
        Search across user-scoped Redis indexes concurrently.

        Args:
            query: Search query string.
            k: Total number of results to return across all indexes.
            metadata: Optional metadata filters.
            user_index_names: Explicit list of index names belonging to the current user.
                              If None or empty, returns no results (safe default).

        Returns:
            Merged and sorted list of top k documents from the user's indexes.
        """
        try:
            if not user_index_names:
                logger.warning("No user index names provided for multi-index search — returning empty results")
                return []

            available_indexes = user_index_names
            if not available_indexes:
                logger.warning("No indexes available for search")
                return []
            
            logger.info(f"Searching across {len(available_indexes)} indexes: {available_indexes}")
            
            # Calculate k per index to ensure we get enough results for final top-k selection
            # Request more results per index to ensure good coverage after merging
            k_per_index = max(k, 10)  # Request at least 10 from each index
            
            # Search all indexes concurrently
            search_tasks = [
                self._search_single_index(index_name.decode('utf-8') if isinstance(index_name, bytes) else index_name, 
                                         query, k_per_index, metadata)
                for index_name in available_indexes
            ]
            
            # Gather results from all indexes
            all_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Merge results and filter out exceptions
            merged_docs: List[Document] = []
            for idx, result in enumerate(all_results):
                if isinstance(result, Exception):
                    index_name = available_indexes[idx]
                    logger.error(f"Error searching index '{index_name}': {result}")
                    continue
                merged_docs.extend(result)
            
            # Sort by vector_score (lower is better for cosine distance)
            merged_docs.sort(key=lambda doc: doc.metadata.get("vector_score", float('inf')))
            
            # Return top k results
            top_k_docs = merged_docs[:k]
            
            logger.info(
                f"Multi-index search: returned {len(top_k_docs)} docs from {len(available_indexes)} indexes "
                f"(total candidates: {len(merged_docs)})"
            )
            
            return top_k_docs
            
        except Exception as e:
            logger.error(f"Error during multi-index search: {e}")
            return []
    
    def _parse_search_results(self, results: Any) -> List[Document]:
        """
        Parse Redis search results into Document objects.
        
        Args:
            results: Redis search results object.
            
        Returns:
            List of Document objects with parsed metadata.
        """
        docs = []
        for d in results.docs:
            content = self._get_doc_field(d, "content") or ""
            page_num = self._get_doc_field(d, "page_number")
            try:
                page_num = int(page_num) if page_num is not None else 0
            except (ValueError, TypeError):
                page_num = 0
            score_val = self._get_doc_field(d, "vector_score")
            try:
                score = float(score_val) if score_val is not None else 1.0
            except (TypeError, ValueError):
                score = 1.0
            sheet_name = self._get_doc_field(d, "sheet_name") or ""
            row_num = self._get_doc_field(d, "row_number")
            try:
                row_num = int(row_num) if row_num is not None else 0
            except (ValueError, TypeError):
                row_num = 0
            meta = {
                "filename": self._get_doc_field(d, "filename") or "",
                "type": self._get_doc_field(d, "type") or "",
                "page_number": page_num,
                "sheet_name": sheet_name,
                "row_number": row_num,
                "vector_score": score,
            }
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def as_retriever(
        self,
        index_name: str | None,
        key_prefix: str | None = None,
        search_kwargs: dict | None = None,
        user_index_names: List[str] | None = None,
        reranker: Any | None = None,
        candidates_multiplier: int = 3,
    ) -> "RedisVectorStoreRetriever":
        """Return a retriever that uses this store's search (for RAG)."""
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 5)
        return RedisVectorStoreRetriever(
            vectorstore=self,
            index_name=index_name,
            key_prefix=key_prefix or (index_name if index_name else "default"),
            k=k,
            user_index_names=user_index_names,
            reranker=reranker,
            candidates_multiplier=candidates_multiplier,
        )


class RedisVectorStoreRetriever(BaseRetriever):
    """Retriever that uses RedisVectorStore.search (async), with optional re-ranking."""

    vectorstore: RedisVectorStore
    index_name: str | None  # Optional: if None, searches all indexes
    key_prefix: str
    k: int = 5
    user_index_names: List[str] | None = None  # Scoped list of indexes for multi-index search
    reranker: Any | None = None  # Optional reranker module (services.reranker)
    candidates_multiplier: int = 3  # Over-retrieve factor when re-ranking

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        if self.reranker is not None:
            fetch_k = self.k * self.candidates_multiplier
            candidates = asyncio.run(self.vectorstore.search(
                self.index_name, query, k=fetch_k, user_index_names=self.user_index_names
            ))
            logger.info(
                "Re-ranking: {} candidates → top {}",
                len(candidates), self.k,
            )
            return self.reranker.rerank_sync(query, candidates, top_k=self.k)

        return asyncio.run(self.vectorstore.search(
            self.index_name, query, k=self.k, user_index_names=self.user_index_names
        ))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        if self.reranker is not None:
            fetch_k = self.k * self.candidates_multiplier
            candidates = await self.vectorstore.search(
                self.index_name, query, k=fetch_k, user_index_names=self.user_index_names
            )
            logger.info(
                "Re-ranking: {} candidates → top {}",
                len(candidates), self.k,
            )
            return await self.reranker.rerank(query, candidates, top_k=self.k)

        return await self.vectorstore.search(
            self.index_name, query, k=self.k, user_index_names=self.user_index_names
        )



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

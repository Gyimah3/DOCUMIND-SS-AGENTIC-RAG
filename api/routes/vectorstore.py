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
from sqlalchemy.exc import IntegrityError
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
from services.storage import AWSStorageService


# Maximum file size: 10 MB
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def get_document_loader(
    files: List[UploadFile] = File(..., description="Files to index"),
    extract_images: bool = Form(False, description="Extract images from documents"),
    extract_tables: bool = Form(True, description="Extract tables"),
) -> DocumentLoader:
    """Dependency: inject DocumentLoader with request files and options (old API style: loader = Depends())."""
    # Validate file sizes before processing
    for file in files:
        # Read content to check size, then seek back
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset for downstream processing

        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' is {size_mb:.2f} MB, which exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB.",
            )

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
    sheet_name: str = ""
    row_number: int = 0

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
        self.storage = AWSStorageService(settings.aws_config)

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

                # Check if another user already owns this index name
                taken_q = select(VectorStore).where(VectorStore.index_name == index_name)
                taken_result = await db.execute(taken_q)
                if taken_result.scalar_one_or_none():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f'The index name "{index_name}" is already in use. Please choose a different name.',
                    )

                logger.info("Creating new index: %s", index_name)
                new_index = VectorStore(
                    index_name=index_name,
                    user_id=current_user.id,
                    embedding=str(embedding_model),
                    key_prefix=prefix,
                )
                db.add(new_index)
                try:
                    await db.flush()
                except IntegrityError:
                    await db.rollback()
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f'The index name "{index_name}" is already in use. Please choose a different name.',
                    )
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
                # 2. Upload to S3 (moved inside loop for individual tracking)
                # We need to seek back to start if loader read it
                await source.seek(0)
                file_content = await source.read()
                s3_result = await self.storage.upload_file(
                    file_content=file_content,
                    filename=source.filename,
                    content_type=source.content_type,
                    prefix=f"{current_user.id}/{index_name}"
                )

                data_source = DataSource(
                    id=UUID(doc_id),
                    title=source.filename or "unknown",
                    s3_url=s3_result["url"],
                    s3_key=s3_result["key"],
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

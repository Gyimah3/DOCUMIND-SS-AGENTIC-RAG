"""Document router: handles S3 uploads, document listing, and bulk deletion."""
import asyncio
from typing import Any, Dict, List
from uuid import UUID, uuid4

from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from fastapi.routing import APIRouter
from langchain_core.documents import Document as LCDocument
from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.decorator import action
from api.routes.base_di import BaseRouter
from app.config import settings
from database.models import User, VectorStore, DataSource, VectorIndex
from database.session import db_session_manager
from document.document_loaders import DocumentLoader
from document.document_loaders.base import ChunkType
from middleware.dep import get_current_user
from services.redis_vectorstore import RedisVectorStore
from services.storage import AWSStorageService
from schemas.document import (
    DocumentResponse, 
    DocumentUploadResponse, 
    BulkDeleteResponse,
    DocumentUrlResponse,
    GlobalDocumentResponse
)

import redis as redis_sync

_redis_url = settings.redis_url
_redis_client: redis_sync.Redis | None = None


def _get_redis_client() -> redis_sync.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_sync.Redis.from_url(_redis_url)
    return _redis_client


def _get_embedding(model: str = "text-embedding-3-small"):
    from langchain_openai import OpenAIEmbeddings
    api_key = getattr(settings, "openai_api_key", None) or ""
    if not api_key:
        raise ValueError("OpenAI API key is missing")
    return OpenAIEmbeddings(model=model, api_key=api_key)


def _get_vector_store(embedding_model: str) -> RedisVectorStore:
    client = _get_redis_client()
    emb = _get_embedding(embedding_model)
    return RedisVectorStore(client=client, embedding_func=emb)


class DocumentRouter(BaseRouter):
    """Router for document management using DataSource model and S3."""

    def __init__(self) -> None:
        router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
        super().__init__(router)
        self.storage = AWSStorageService(settings.aws_config)

    @action(
        method="POST",
        url_path="{index_name}/upload",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(get_current_user)],
    )
    async def upload_document(
        self,
        index_name: str,
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
        embedding_model: str = Form("text-embedding-3-small"),
        prefix: str | None = Form(None),
        multimodal: bool = Form(False),
        extract_tables: bool = Form(False),
        extract_images: bool = Form(False),
    ) -> DocumentUploadResponse:
        """Upload a file to S3, index it in Redis, and track in DataSource."""
        try:
            # 1. Get or create VectorStore record
            q = select(VectorStore).where(
                VectorStore.index_name == index_name,
                VectorStore.user_id == current_user.id
            )
            result = await db.execute(q)
            vs_data = result.scalar_one_or_none()
            
            if not vs_data:
                vs_data = VectorStore(
                    index_name=index_name,
                    user_id=current_user.id,
                    embedding=embedding_model,
                    key_prefix=prefix or index_name
                )
                db.add(vs_data)
                await db.flush()
            
            # 2. Upload to S3
            file_content = await file.read()
            s3_result = await self.storage.upload_file(
                file_content=file_content,
                filename=file.filename,
                content_type=file.content_type,
                prefix=f"{current_user.id}/{index_name}"
            )
            
            # 3. Load and Chunk
            # Pass extraction flags to DocumentLoader
            loader = DocumentLoader(
                sources=[file],
                extract_images=extract_images or multimodal,
                extract_tables=extract_tables
            )
            chunks = []
            async for chunk_batch in loader.lazy_load_async([file]):
                for chunk in chunk_batch:
                    if multimodal or chunk.metadata.get("type") in (ChunkType.text.value, ChunkType.table.value):
                        chunks.append(LCDocument(
                            page_content=chunk.page_content,
                            metadata={
                                **chunk.metadata,
                                "source_id": str(vs_data.id),
                                "filename": file.filename
                            }
                        ))

            # 4. Index in Redis
            lc_redis = _get_vector_store(embedding_model)
            # Ensure index exists
            try:
                lc_redis.create_index(vs_data.index_name, vs_data.key_prefix)
            except Exception:
                pass # Already exists or other error
            
            vector_ids = await lc_redis.index_documents(
                vs_data.index_name,
                vs_data.key_prefix,
                documents=chunks
            )
            
            # 5. Create DataSource and VectorIndex records
            doc_id = uuid4()
            data_source = DataSource(
                id=doc_id,
                user_id=current_user.id,
                vector_store_id=vs_data.id,
                title=file.filename,
                mimetype=file.content_type,
                s3_key=s3_result["key"],
                s3_url=s3_result["url"]
            )
            db.add(data_source)
            
            for vid in vector_ids:
                part = vid.split(":", 1)[-1]
                v_idx = VectorIndex(
                    id=UUID(part),
                    vector_store_id=vs_data.id,
                    document_id=doc_id
                )
                db.add(v_idx)
            
            await db.commit()
            
            return DocumentUploadResponse(
                document_id=doc_id,
                s3_url=s3_result["url"],
                s3_key=s3_result["key"]
            )
            
        except Exception as e:
            logger.exception("Failed to upload and index document")
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

    @action(
        method="GET",
        url_path="{index_name}/list",
        response_model=List[DocumentResponse],
        dependencies=[Depends(get_current_user)],
    )
    async def list_documents(
        self,
        index_name: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[DocumentResponse]:
        """List all documents for a given index."""
        q = select(DataSource).join(VectorStore).where(
            VectorStore.index_name == index_name,
            VectorStore.user_id == current_user.id
        ).order_by(DataSource.created_at.desc())
        
        result = await db.execute(q)
        docs = result.scalars().all()
        return [DocumentResponse.model_validate(d) for d in docs]

    @action(
        method="GET",
        url_path="{index_name}/url",
        response_model=DocumentUrlResponse,
        dependencies=[Depends(get_current_user)],
    )
    async def get_document_url(
        self,
        index_name: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> DocumentUrlResponse:
        """Get the URL of the most recent document for a given index."""
        q = select(DataSource).join(VectorStore).where(
            VectorStore.index_name == index_name,
            VectorStore.user_id == current_user.id
        ).order_by(DataSource.created_at.desc()).limit(1)
        
        result = await db.execute(q)
        doc = result.scalar_one_or_none()
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document found for index '{index_name}'"
            )
            
        url = doc.s3_url or doc.url
        if not url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document found but no URL is available"
            )
            
        return DocumentUrlResponse(url=url)

    @action(
        method="GET",
        url_path="list-all",
        response_model=List[GlobalDocumentResponse],
        dependencies=[Depends(get_current_user)],
    )
    async def list_all_documents(
        self,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[GlobalDocumentResponse]:
        """List all documents for the current user across all indices."""
        q = select(DataSource, VectorStore.index_name).join(VectorStore).where(
            VectorStore.user_id == current_user.id
        ).order_by(DataSource.created_at.desc())
        
        result = await db.execute(q)
        rows = result.all()
        
        responses = []
        for doc, index_name in rows:
            # We can't use model_validate directly because we need to inject index_name
            # and doc is a row/model
            data = {
                "id": doc.id,
                "title": doc.title,
                "mimetype": doc.mimetype,
                "s3_key": doc.s3_key,
                "s3_url": doc.s3_url,
                "url": doc.url,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at,
                "index_name": index_name
            }
            responses.append(GlobalDocumentResponse(**data))
            
        return responses

    @action(
        method="DELETE",
        url_path="{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(get_current_user)],
    )
    async def delete_document(
        self,
        document_id: UUID,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ):
        """Delete a specific document, its vectors in Redis, and its file in S3."""
        # 1. Fetch document and verify ownership
        q = select(DataSource).options(
            selectinload(DataSource.vectorstore),
            selectinload(DataSource.vector_indexes)
        ).where(
            DataSource.id == document_id,
            DataSource.user_id == current_user.id
        )
        result = await db.execute(q)
        doc = result.scalar_one_or_none()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # 2. Delete from Redis
        if doc.vectorstore and doc.vector_indexes:
            lc_redis = _get_vector_store(doc.vectorstore.embedding)
            key_prefix = doc.vectorstore.key_prefix
            
            # Redis keys are usually {prefix}:{uuid}
            keys_to_delete = [f"{key_prefix}:{vi.id}" for vi in doc.vector_indexes]
            if keys_to_delete:
                try:
                    await asyncio.to_thread(lc_redis.client.delete, *keys_to_delete)
                    logger.info(f"Deleted {len(keys_to_delete)} vectors from Redis for doc {document_id}")
                except Exception as e:
                    logger.error(f"Failed to delete vectors from Redis: {e}")

        # 3. Delete from S3
        if doc.s3_key:
            try:
                await self.storage.delete_file(doc.s3_key)
                logger.info(f"Deleted file {doc.s3_key} from S3")
            except Exception as e:
                logger.error(f"Failed to delete file from S3: {e}")

        # 4. Delete document from DB
        vectorstore = doc.vectorstore
        await db.delete(doc)
        await db.flush()

        # 5. If this was the last document, clean up the empty VectorStore and Redis index
        if vectorstore:
            remaining = await db.scalar(
                select(func.count(DataSource.id)).where(
                    DataSource.vector_store_id == vectorstore.id
                )
            )
            if remaining == 0:
                logger.info(
                    f"No documents remaining in index '{vectorstore.index_name}', deleting empty index"
                )
                try:
                    lc_redis = _get_vector_store(vectorstore.embedding)
                    lc_redis.client.execute_command("FT.DROPINDEX", vectorstore.index_name, "DD")
                except Exception as e:
                    logger.error(f"Failed to delete empty Redis index: {e}")
                await db.delete(vectorstore)

        await db.commit()

        return None

    @action(
        method="DELETE",
        url_path="{index_name}/delete-all",
        response_model=BulkDeleteResponse,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(get_current_user)],
    )
    async def delete_all_in_index(
        self,
        index_name: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> BulkDeleteResponse:
        """Delete ALL documents, vectors, and S3 files for an index."""
        try:
            # 1. Find the VectorStore
            q = select(VectorStore).options(
                selectinload(VectorStore.data_sources)
            ).where(
                VectorStore.index_name == index_name,
                VectorStore.user_id == current_user.id
            )
            result = await db.execute(q)
            vs = result.scalar_one_or_none()
            if not vs:
                raise HTTPException(status_code=404, detail="Index not found")
            
            count = len(vs.data_sources)
            
            # 2. Delete S3 files
            s3_deletions = 0
            for ds in vs.data_sources:
                if ds.s3_key:
                    if await self.storage.delete_file(ds.s3_key):
                        s3_deletions += 1
            
            # 3. Delete from Redis
            # We can just delete the entire index or all keys with prefix
            lc_redis = _get_vector_store(vs.embedding)
            try:
                # Option A: Delete entire index
                # lc_redis.delete_index(vs.index_name) 
                # Wait, the user might want to keep the index but clear records?
                # "delete every records... regarding to that index"
                # If we delete the index, we also delete the schema/metadata in Redis.
                # Usually it's better to recreate it or just delete keys.
                # Let's delete the index for completeness.
                lc_redis.delete_index(vs.index_name)
                redis_deletions = 1 # Representing the index deletion
            except Exception as e:
                logger.warning(f"Error deleting Redis index {index_name}: {e}")
                redis_deletions = 0

            # 4. Delete from DB (Cascade will handle DataSources and VectorIndexes)
            await db.delete(vs)
            await db.commit()
            
            return BulkDeleteResponse(
                deleted_count=count,
                deleted_from_s3=s3_deletions,
                deleted_from_redis=redis_deletions,
                deleted_from_db=count + 1 # +1 for VectorStore itself
            )
            
        except Exception as e:
            logger.exception(f"Bulk deletion failed for index {index_name}")
            await db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

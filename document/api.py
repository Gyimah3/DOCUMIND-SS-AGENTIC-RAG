"""
Legacy vectorstore API (backend.* imports).

These routes have been moved to api/routes/vectorstore.py using the @action-style
router (BaseRouter + @action). Register the new router in your app with:

  from api.routes.vectorstore import VectorstoreRouter
  vs_router = VectorstoreRouter()
  vs_router.setup_routes(app)  # or app.include_router(vs_router.get_router())
"""
import asyncio
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

import redis
from dateutil.parser import parse as dateutil_parse
from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.routing import APIRouter
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from langchain_core.documents import Document

from backend.auth.services.dependencies import AccessBearer, RoleChecker
from backend.document.document_loaders.base import ChunkType
from backend.library.config import settings
from backend.library.utils import load_vector_store
from backend.persistence.db.models.base import get_db
from backend.persistence.db.models.data_source import DataSource
from backend.persistence.db.models.vectorstore import VectorStore
from backend.persistence.db.models.vs_index import VectorIndex
from backend.utilities.embeddings import EMBEDDING_MODELS
from .document_loaders import DocumentLoader
from .schemas import response


role_checker = RoleChecker(roles=["admin", "user"])


class Metadata(BaseModel):
    filename: str
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    modified: float = Field(default_factory=lambda: datetime.now().timestamp())
    type: ChunkType
    page_number: int = Field(default=1)

    @field_validator("created_at", "modified", mode="before")
    def parse_dates(cls, value):
        if not value:
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

    @field_validator("page_number", mode="before")
    def parse_page_number(cls, value):
        if not value:
            return 1
        return value


if settings.REDIS_PASSWORD:
    redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
else:
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

redis_client = redis.Redis.from_url(redis_url)


router = APIRouter(
    prefix="/vectorstore",
    tags=["vectorstore"],
    dependencies=[
        Depends(lambda: redis_client.ping()),
    ],  # Depends(role_checker)],
)

security = AccessBearer()


# @router.post("/create-index", status_code=status.HTTP_201_CREATED)
# async def create_index(
#     index_name: str,
#     key_prefix: str | None = None,
#     *,
#     embedding_model: EMBEDDING_MODELS,
#     token_details: dict = Depends(security),
#     db: AsyncSession = Depends(get_db),
# ) -> Dict[str, str]:
#     token_data = TokenData.model_validate(token_details)
#     try:
#         index = await db.execute(
#             select(VectorStore).filter(
#                 VectorStore.index_name == index_name,
#                 VectorStore.user_id == token_data.id,
#             )
#         )
#         if index.scalar_one_or_none():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Index with this name already exists",
#             )

#         vs_data = VectorStore(
#             index_name=index_name,
#             key_prefix=key_prefix,
#             user_id=token_data.id,
#             embedding=embedding_model,
#         )
#         db.add(vs_data)
#         await db.commit()

#         try:
#             vs = redis_vectorstore.RedisVectorStore(
#                 client=redis_client,
#                 embedding_func=load_embedding_model(model: EMBEDDING_MODELS, provider: Literal["openai", "huggingface"]),
#             )
#             vs.create_index(index_name, key_prefix)
#         except Exception as e:
#             await db.rollback()
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Failed to create index: {str(e)}",
#             )

#         return {"status": "success"}

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An unexpected error occurred: {str(e)}",
#         )


@router.post("/add-document", status_code=status.HTTP_201_CREATED)
async def load_document(
    index_name: str,
    embedding: EMBEDDING_MODELS = "text-embedding-3-small",
    prefix: str | None = None,
    multimodal: bool = False,
    loader: DocumentLoader = Depends(),
    token_data: Dict = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    async def get_or_create_index() -> tuple[VectorStore, bool]:
        query = (
            select(VectorStore)
            .options(selectinload(VectorStore.data_sources))
            .filter(
                VectorStore.index_name == index_name,
                VectorStore.user_id == token_data["id"],
            )
        )
        result = await db.execute(query)
        existing_index = result.scalar_one_or_none()
        if existing_index:
            logger.info("Index %s already exists, using existing index", index_name)
            return existing_index, False

        logger.info("Creating new index: %s" % index_name)
        new_index = VectorStore(
            index_name=index_name,
            user_id=token_data["id"],
            embedding=embedding,
            key_prefix=prefix,
        )
        db.add(new_index)
        await db.flush()
        return new_index, True

    async def check_duplicate_files(vs_data: VectorStore, is_new_index: bool = False):
        if is_new_index:
            return
        existing_files = {source.title for source in vs_data.data_sources}
        new_files = {source.filename for source in loader.sources}
        duplicate_files = existing_files.intersection(new_files)
        if duplicate_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Documents already exist in the index: {', '.join(duplicate_files)}",
            )

    async def process_documents() -> tuple[List[str], List[List[Document]]]:
        doc_ids: List[str] = []
        docs: List[List[Document]] = []
        async for doc_chunks in loader.lazy_load_async(loader.sources):  # type: ignore
            doc_id = str(uuid4())
            doc_ids.append(doc_id)
            docs.append(
                [
                    Document(
                        page_content=doc.page_content,
                        metadata=Metadata(**doc.metadata).model_dump(mode="python"),
                    )
                    for doc in doc_chunks
                    if multimodal
                    or doc.metadata["type"] in [ChunkType.text, ChunkType.table]
                ]
            )
        # documents = await loader.load_batch_async(loader.sources)
        # logger.info("Loaded %d documents" % len(documents))
        # for doc in documents:
        #     doc_id = str(uuid4())
        #     doc_ids.append(doc_id)
        #     docs.append(
        #         [
        #             Document(
        #                 page_content=chunk.page_content,
        #                 metadata=Metadata(**chunk.metadata).dict(),
        #             )
        #             for chunk in doc
        #             if multimodal or chunk.metadata.type in [ChunkType.text, ChunkType.table]
        #         ]
        #     )

        return doc_ids, docs

    async def add_to_vector_store(
        vs_data: VectorStore, docs: List[List[Document]], is_new_index: bool
    ):
        logger.info("Embedding model: %s" % vs_data.embedding)
        lc_redis = load_vector_store(
            redis_url,
            vs_data.index_name,
            vs_data.key_prefix,
            vs_data.embedding,  # type: ignore
            # existing=not is_new_index,
        )
        return await asyncio.gather(*(lc_redis.aadd_documents(doc) for doc in docs))

    async def prepare_data_sources(
        vs_data: VectorStore, source: UploadFile, doc_id: str, chunk_ids: List[str]
    ):
        data_source = DataSource(
            id=doc_id,
            title=source.filename,
            url=getattr(source, "url", None),
            mimetype=source.content_type,
            user_id=token_data["id"],
            vector_store_id=vs_data.id,
        )
        vector_indices = [
            VectorIndex(
                id=str(UUID(chunk_id.split(":", 1)[-1])),
                document_id=doc_id,
                vector_store_id=vs_data.id,
            )
            for chunk_id in chunk_ids
        ]
        return [data_source] + vector_indices

    try:
        vs_data, is_new_index = await get_or_create_index()
        await check_duplicate_files(vs_data, is_new_index)
        doc_ids, docs = await process_documents()
        vector_store_ids = await add_to_vector_store(vs_data, docs, is_new_index)
        new_data_sources = []
        for source, doc_id, chunk_ids in zip(loader.sources, doc_ids, vector_store_ids):
            new_data_sources.extend(
                await prepare_data_sources(vs_data, source, doc_id, chunk_ids)
            )

        db.add_all(new_data_sources)
        await db.commit()

        return {"status": "success", "ids": vector_store_ids}

    except HTTPException:
        raise
    except ValidationError as val_err:
        logger.error("Validation error: %s" % val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err)
        )
    except Exception as e:
        logger.error("Error loading Document: %r", e)
        logger.exception("Traceback:")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )


@router.get("/list-documents/{index_name}")
async def list_documents(
    index_name: str,
    token_data: Dict = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    try:
        query = (
            select(DataSource)
            .filter(
                DataSource.vectorstore.has(index_name=index_name),
                DataSource.user_id == token_data["id"],
            )
            .order_by(DataSource.created_at.desc())
        )
        result = await db.execute(query)
        return result.scalars().all()

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.delete("/document/{id}")
async def delete_document(
    id: str,
    token_data: Dict = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    try:
        query = (
            select(DataSource)
            .options(selectinload(DataSource.vector_indexes))
            .options(selectinload(DataSource.vectorstore))
            .filter(DataSource.id == id, DataSource.user_id == token_data["id"])
        )
        result = await db.execute(query)
        doc = result.unique().scalar_one_or_none()

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        embedding_model = str(doc.vectorstore.embedding)
        index_name = str(doc.vectorstore.index_name)
        prefix = doc.vectorstore.key_prefix or index_name
        doc_ids = [str(index.id.hex) for index in doc.vector_indexes]

        lc_redis = load_vector_store(
            embedding_model=embedding_model,  # type: ignore
            index_name=index_name,
            key_prefix=prefix,
            redis_url=redis_url,
            # existing=True,
        )
        logger.info("Deleting document from index: %s" % doc_ids)
        success = await lc_redis.adelete(doc_ids)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting document from index",
            )
        await db.delete(doc)
        await db.commit()

        return {"status": "success"}

    except HTTPException as e:
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred {str(e)}",
        )


@router.get(
    "/search", deprecated=True, dependencies=[Depends(RoleChecker(roles=["admin"]))]
)
async def search(index_name: str, prefix: str, query: str, k: int = 4) -> Any:
    lc_redis = load_vector_store(
        redis_url, index_name, prefix, "text-embedding-3-small", existing=True
    ).as_retriever(search_kwargs={"k": k})
    docs = await lc_redis.ainvoke(query)
    return [doc.dict() for doc in docs]


@router.get(
    "/list-indexes",
    status_code=status.HTTP_200_OK,
    response_model=List[response.VectorStoreModel],
)
async def list_indexes(
    token_data: Dict = Depends(security), db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        indexes = await db.execute(
            select(VectorStore).filter(VectorStore.user_id == token_data["id"])
        )
        return indexes.scalars().all()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.delete(
    "/delete-index/{id}",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(security)],
)
async def delete_index(
    id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, str]:
    try:
        index = await db.get(VectorStore, id)
        if not index:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index not found",
            )
        index_name = index.index_name
        try:
            await db.delete(index)
            await db.commit()
            try:
                lc_redis = load_vector_store(
                    redis_url,
                    index_name,
                    index.key_prefix,
                    index.embedding,  # type: ignore
                )
                lc_redis.index.delete(drop=True)
            except Exception as e:
                await db.rollback()
                logger.error(f"Error deleting index from vector store: {repr(e)}")
                traceback.print_exc()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error deleting index from vector store: {repr(e)}",
                )
            return {"status": "success"}
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting index from database: {repr(e)}")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting index from database: {repr(e)}",
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {repr(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {repr(e)}",
        )

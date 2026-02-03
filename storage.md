"""File storage service with AWS S3 integration and presigned URL generation."""

import asyncio
import mimetypes
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from loguru import logger

from app.errors import BadRequestError, InternalServerError, NotFoundError
from schemas.document import FileMetadata, PresignedUrlResponse

if TYPE_CHECKING:
    from app.config import AWSConfig


class FileStorage(Protocol):
    async def generate_upload_url(
        self,
        filename: str,
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> PresignedUrlResponse: ...

    async def upload_file(
        self,
        file_content: bytes,
        key: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a file directly to storage."""
        ...

    async def generate_download_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,
        filename: Optional[str] = None,
    ) -> str:
        """Generate a presigned URL for file download."""
        ...

    async def delete_file(
        self,
        key: str,
        bucket: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> bool:
        """Delete a file from storage."""
        ...

    async def get_file_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> Optional[FileMetadata]:
        """Get file metadata."""
        ...

    async def download_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download file content from storage."""
        ...

    async def file_exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if file exists."""
        ...


class AWSStorageService:
    def __init__(self, config: "AWSConfig"):
        self.config = config
        self._client = None
        self._lock = asyncio.Lock()

    @property
    def client(self):
        """Lazy-loaded S3 client."""
        if self._client is None:
            try:
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=self.config.access_key_id,
                    aws_secret_access_key=self.config.secret_access_key,
                    region_name=self.config.region,
                    config=Config(
                        signature_version=self.config.signature_version,
                        retries={"max_attempts": 3, "mode": "standard"},
                    ),
                )
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                raise InternalServerError(
                    detail="Failed to initialize storage service",
                )
        return self._client

    def _generate_unique_key(
        self,
        filename: str,
        prefix: Optional[str] = None,
    ) -> str:
        name_parts = filename.rsplit(".", 1)
        if len(name_parts) == 2:
            base_name, extension = name_parts
            extension = f".{extension}"
        else:
            base_name = filename
            extension = ""

        unique_id = str(uuid4())

        if prefix:
            if not prefix.endswith("/"):
                prefix = f"{prefix}/"
            key = f"{prefix}{base_name}_{unique_id}{extension}"
        else:
            key = f"{base_name}_{unique_id}{extension}"

        return key

    def _validate_content_type(self, content_type: str) -> None:
        all_allowed_types = set(
            self.config.allowed_content_types
            + [
                "image/jpg",
                "image/svg+xml",
                "application/json",
                "text/csv",
                "application/zip",
                "application/x-zip-compressed",
            ]
        )

        if content_type not in all_allowed_types:
            logger.warning(f"Content type {content_type} not in allowed list")

    async def generate_upload_url(
        self,
        filename: str,
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> PresignedUrlResponse:
        """Generate presigned URL for secure file upload.
        
        Note: Uploads still use presigned URLs for security.
        Downloads use direct public URLs since bucket policy allows public read.
        """
        try:
            bucket_name = bucket or self.config.default_bucket

            key = self._generate_unique_key(filename, prefix)

            if not content_type:
                content_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )

            self._validate_content_type(content_type)

            params: Dict[str, Any] = {
                "Bucket": bucket_name,
                "Key": key,
                "ContentType": content_type,
                # Don't specify ACL for public buckets - let bucket policy handle permissions
            }

            if metadata:
                params["Metadata"] = metadata

            async with self._lock:
                url = await asyncio.to_thread(
                    self.client.generate_presigned_url,
                    "put_object",
                    Params=params,
                    ExpiresIn=expires_in,
                )

            return PresignedUrlResponse(
                url=url, key=key, bucket=bucket_name, expires_in=expires_in, fields=None
            )

        except ClientError as e:
            logger.error(f"S3 client error generating upload URL: {e}")
            raise InternalServerError(
                detail="Failed to generate upload URL",
            )

    async def upload_file(
        self,
        file_content: bytes,
        key: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            bucket_name = bucket or self.config.default_bucket

            if len(file_content) > self.config.max_upload_size:
                raise BadRequestError(
                    detail=f"File size exceeds maximum allowed size of {self.config.max_upload_size} bytes",
                )

            self._validate_content_type(content_type)

            upload_params = {
                "Bucket": bucket_name,
                "Key": key,
                "Body": file_content,
                "ContentType": content_type,
                "ACL": "private",  # Private ACL, but bucket policy allows public read
            }

            if metadata:
                upload_params["Metadata"] = metadata

            async with self._lock:
                response = await asyncio.to_thread(
                    self.client.put_object, **upload_params
                )

            file_url = (
                f"https://{bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"
            )

            logger.success(
                f"Successfully uploaded file: {key} to bucket: {bucket_name}"
            )

            return {
                "url": file_url,
                "key": key,
                "bucket": bucket_name,
                "etag": response.get("ETag"),
                "version_id": response.get("VersionId"),
            }

        except ClientError as e:
            logger.error(f"S3 client error uploading file: {e}")
            raise InternalServerError(
                detail="Failed to upload file",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            raise InternalServerError(
                detail="Internal server error",
            )

    async def generate_download_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in: int = 3600,  # Not used for public buckets, kept for compatibility
        filename: Optional[str] = None,
    ) -> str:
        """Generate download URL. Returns direct public URL since bucket is public.
        
        Since the S3 bucket has public read access via bucket policy,
        we return direct public URLs instead of presigned URLs.
        """
        try:
            bucket_name = bucket or self.config.default_bucket

            # Generate direct public URL (bucket is public via policy)
            # Format: https://{bucket}.s3.{region}.amazonaws.com/{key}
            public_url = f"https://{bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"
            
            logger.info(f"Generated public download URL for {key} (bucket is public)")
            return public_url

        except Exception as e:
            logger.error(f"Unexpected error generating download URL: {e}")
            raise InternalServerError(
                detail="Failed to generate download URL",
            )

    async def delete_file(
        self,
        key: str,
        bucket: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> bool:
        try:
            bucket_name = bucket or self.config.default_bucket

            params = {"Bucket": bucket_name, "Key": key}
            if version_id:
                params["VersionId"] = version_id

            async with self._lock:
                await asyncio.to_thread(self.client.delete_object, **params)

            logger.info(f"Deleted file: {key} from bucket: {bucket_name}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                logger.warning(f"File not found for deletion: {key}")
                return False
            logger.error(f"S3 client error deleting file: {e}")
            raise InternalServerError(
                detail="Failed to delete file",
            )
        except Exception as e:
            logger.error(f"Unexpected error deleting file: {e}")
            raise InternalServerError(
                detail="Internal server error",
            )

    async def get_file_metadata(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> Optional[FileMetadata]:
        try:
            bucket_name = bucket or self.config.default_bucket

            async with self._lock:
                response = await asyncio.to_thread(
                    self.client.head_object, Bucket=bucket_name, Key=key
                )

            metadata = FileMetadata(
                filename=response.get("Metadata", {}).get("original-filename", key),
                content_type=response.get("ContentType", "application/octet-stream"),
                size=response.get("ContentLength", 0),
                bucket=bucket_name,
                key=key,
                etag=response.get("ETag"),
                version_id=response.get("VersionId"),
            )

            return metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return None
            logger.error(f"S3 client error getting file metadata: {e}")
            raise InternalServerError(
                detail="Failed to get file metadata",
            )
        except Exception as e:
            logger.error(f"Unexpected error getting file metadata: {e}")
            raise InternalServerError(
                detail="Internal server error",
            )

    async def download_file(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bytes:
        """Download file content from storage.

        Args:
            key: S3 object key
            bucket: Optional bucket name (defaults to configured bucket)

        Returns:
            bytes: File content

        Raises:
            NotFoundError: If file doesn't exist
            InternalServerError: If download fails

        """
        try:
            bucket_name = bucket or self.config.default_bucket

            async with self._lock:
                response = await asyncio.to_thread(
                    self.client.get_object, Bucket=bucket_name, Key=key
                )

            return response["Body"].read()

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise NotFoundError(detail="File not found")
            raise InternalServerError(
                detail="Failed to download file",
            )
        except Exception as e:
            logger.error(f"Unexpected error downloading file: {e}")
            raise InternalServerError(
                detail="Internal server error",
            )

    async def file_exists(
        self,
        key: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if file exists."""
        try:
            metadata = await self.get_file_metadata(key, bucket)
            return metadata is not None
        except HTTPException:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

    async def list_files(
        self, bucket: Optional[str] = None, prefix: Optional[str] = None
    ) -> List[FileMetadata]:
        try:
            bucket_name = bucket or self.config.default_bucket
            params = {"Bucket": bucket_name, "Prefix": prefix}
            async with self._lock:
                response = await asyncio.to_thread(
                    self.client.list_objects_v2, **params
                )
            return [FileMetadata(**file) for file in response.get("Contents", [])]
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                raise NotFoundError(detail="Bucket not found")
            raise InternalServerError(detail="Failed to list files")
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}")
            raise InternalServerError(detail="Internal server error")


import asyncio
from typing import TYPE_CHECKING, Any, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.documents import Document
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing_extensions import Annotated

from agent.controllers.documents import (
    Metadata,
    check_duplicate_files,
    get_or_create_index,
    list_all_data_source,
    process_documents,
    splitter,
)

# from app.container import ServeiceProvider  # Removed to avoid circular import
from agent.document_loaders import EnsembleDocumentLoader, mime_types_to_loaders
from agent.document_loaders.base import ChunkType, NamedBytesIO
from app.config import settings
from database.models.knowledge_base import KnowledgeBase
from database.models.user import User
from database.models.vectorstore import VectorStore
from database.session import get_session_no_tenant
from middleware.dep import get_current_active_user
from services.storage import AWSStorageService
from services.vectorstore import FileDetails, VectorStoreService

from .base_di import BaseRouter
from .decorator import action

if TYPE_CHECKING:
    from services.vectorstore import VectorStoreService


def get_vectorstore_service() -> VectorStoreService:
    """Get VectorStoreService instance."""
    return VectorStoreService(
        redis_url=settings.redis_url,
        embedding_model=settings.embedding_model,
        embedding_model_provider=settings.embedding_provider,
    )


s3_service = AWSStorageService(
    config=settings.aws_config,
)


class KnowledgeBaseRouter(BaseRouter):
    """Knowledge Base and Vector Store management router."""

    def __init__(self):
        router = APIRouter(
            prefix=f"/api/{settings.api_version}/documents", tags=["vectorstore"]
        )
        super().__init__(router)

    @action(
        method="POST",
        url_path="{index_name}/generate-presigned-url",
        status_code=status.HTTP_200_OK,
    )
    async def generate_presigned_url_for_upload(
        self,
        index_name: str,
        file_name: str = Body(...),
        content_type: str = Body(...),
    ):
        """Generate presigned URL for file upload to S3."""
        rag_folder = f"vectorstore_documents/{index_name}"
        try:
            result = s3_service.generate_upload_url(
                file_name=file_name,
                content_type=content_type,
                prefix=rag_folder,
            )
            return result
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error generating presigned URL",
            )

    @action(
        method="POST",
        url_path="{index_name}/add-document-s3",
        status_code=status.HTTP_201_CREATED,
    )
    async def load_document_s3(
        self,
        index_name: str,
        s3_key: str = Body(...),
        filename: str = Body(...),
        content_type: str = Body(...),
        prefix: str | None = None,
        multimodal: bool = False,
        vs: VectorStoreService = Depends(get_vectorstore_service),
        current_user: Annotated[User, Depends(get_current_active_user)] = None,
    ):
        """Load document from S3 into vector store."""
        prefix = prefix or index_name
        try:
            vs_data, is_new_index = await get_or_create_index(index_name, prefix)
            await check_duplicate_files(vs_data, [filename], is_new_index)

            try:
                file_content = await asyncio.to_thread(s3_service.download_file, s3_key)
                if file_content is None:
                    raise ValueError("Empty file content")
            except Exception as e:
                logger.error(f"Error downloading file from S3: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error downloading file from S3: {str(e)}",
                )

            source_bytesio = NamedBytesIO(
                initial_bytes=file_content,
                name=filename,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "size": len(file_content),
                },
            )

            mime_type = (
                await EnsembleDocumentLoader(
                    sources=[], extract_images=False, extract_tables=False
                ).detect_document_type(source_bytesio)
                or content_type
            )
            loader_cls = mime_types_to_loaders.get(mime_type)
            if not loader_cls:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported MIME type: {mime_type}",
                )
            loader = loader_cls()

            docs_pages = await loader.load_async(source_bytesio)
            filtered_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata=Metadata(**doc.metadata).model_dump(mode="python"),
                )
                for doc in docs_pages
                if multimodal
                or doc.metadata.get("type") in [ChunkType.text, ChunkType.table]
            ]
            split_docs = splitter.split_documents(filtered_docs)
            vector_store_ids = await vs.add_to_vector_store(vs_data, [split_docs])

            s3_url = f"https://{s3_service.config.default_bucket}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"
            source = FileDetails(
                filename=filename,
                content_type=content_type,
                s3_key=s3_key,
                s3_url=s3_url,
            )
            doc_id = str(uuid4())
            await vs.add_vectorstore_and_vectorindex(
                [source],
                [doc_id],
                vector_store_ids,
                vs_data,
                current_user.organization_id,
            )

            return {"status": "success", "ids": vector_store_ids}
        except HTTPException:
            raise
        except Exception as e:
            raise e
            logger.error(f"Error processing document from S3: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="POST",
        url_path="{index_name}/add-document",
        status_code=status.HTTP_201_CREATED,
        deprecated=True,
    )
    async def load_document(
        self,
        index_name: str,
        prefix: str | None = None,
        multimodal: bool = False,
        loader: EnsembleDocumentLoader = Depends(),
        vs: VectorStoreService = Depends(get_vectorstore_service),
        current_user: Annotated[User, Depends(get_current_active_user)] = None,
    ):
        """Load document into vector store (deprecated)."""
        prefix = prefix or index_name
        try:
            vs_data, is_new_index = await get_or_create_index(index_name, prefix)
            await check_duplicate_files(
                vs_data,
                [s.filename for s in loader.sources],  # type: ignore
                is_new_index,  # type: ignore
            )
            doc_ids, docs = await process_documents(multimodal, loader)
            vector_store_ids = await vs.add_to_vector_store(vs_data, docs)

            sources = []
            for source in loader.sources:
                try:
                    file_content = None
                    # Log file details for debugging
                    logger.info(
                        f"Processing file: {getattr(source, 'filename', 'unknown')}"
                    )
                    logger.info(f"Source type: {type(source)}")
                    logger.info(
                        f"Has read method: {hasattr(source, 'read') and callable(source.read)}"
                    )

                    if hasattr(source, "read") and callable(source.read):
                        try:
                            file_content = await source.read()
                            logger.info(
                                f"File content read successfully, size: {len(file_content) if file_content else 0} bytes"
                            )
                            if hasattr(source, "seek"):
                                await source.seek(0)
                        except Exception as read_err:
                            logger.error(f"Error reading file content: {read_err}")
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Error reading file content: {str(read_err)}",
                            )

                    if not file_content:
                        logger.warning(
                            f"No file content for {getattr(source, 'filename', 'unknown')}"
                        )
                        # If no file content, just add basic details
                        source_filename = getattr(source, "filename", "unknown")
                        source_content_type = getattr(
                            source, "content_type", "application/octet-stream"
                        )
                        sources.append(
                            FileDetails(
                                filename=source_filename,
                                content_type=source_content_type,
                            )
                        )
                        continue

                    # Generate a unique filename
                    file_extension = ""
                    if hasattr(source, "filename") and source.filename:
                        if "." in source.filename:
                            file_extension = source.filename.split(".")[-1]

                    s3_folder = f"vectorstore_documents/{index_name}"
                    s3_filename = (
                        f"{uuid4()}.{file_extension}"
                        if file_extension
                        else f"{uuid4()}"
                    )
                    s3_key = f"{s3_folder}/{s3_filename}"

                    # Log S3 upload details
                    logger.info(f"Uploading to S3: {s3_key}")
                    logger.info(f"Bucket: {s3_service.config.default_bucket}")
                    logger.info(
                        f"Content type: {getattr(source, 'content_type', 'application/octet-stream')}"
                    )

                    try:
                        content_type = getattr(
                            source, "content_type", "application/octet-stream"
                        )
                        upload_success = await s3_service.upload_file(
                            file_content, s3_key, content_type
                        )

                        if not upload_success:
                            logger.error(f"Failed to upload file to S3: {s3_key}")
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Failed to upload file to S3",
                            )

                        logger.info(f"Successfully uploaded to S3: {s3_key}")
                    except Exception as s3_err:
                        logger.error(f"Error uploading to S3: {s3_err}")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error uploading to S3: {str(s3_err)}",
                        )

                    s3_url = f"https://{s3_service.config.default_bucket}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"

                    # Create FileDetails with S3 information
                    source_filename = getattr(source, "filename", "unknown")
                    source_content_type = getattr(
                        source, "content_type", "application/octet-stream"
                    )
                    sources.append(
                        FileDetails(
                            filename=source_filename,
                            content_type=source_content_type,
                            s3_key=s3_key,
                            s3_url=s3_url,
                        )
                    )
                    logger.info(
                        f"Added file details for {source_filename} with S3 key: {s3_key}"
                    )
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error processing file: {str(e)}",
                    )

            await vs.add_vectorstore_and_vectorindex(
                sources,
                doc_ids,
                vector_store_ids,
                vs_data,
                current_user.organization_id,
            )

            return {"status": "success", "ids": vector_store_ids}

        except HTTPException:
            raise
        except ValidationError as val_err:
            logger.error("Validation error: %s" % val_err)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err)
            )
        except Exception as e:
            raise e
            logger.error("Error loading Document: %r" % e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="POST",
        url_path="{index_name}/add-webpage",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_webpages(
        self,
        index_name: str,
        prefix: str | None = None,
        *,
        urls: List[str],
        vs: VectorStoreService = Depends(get_vectorstore_service),
        current_user: Annotated[User, Depends(get_current_active_user)] = None,
    ):
        """Add webpages to vector store."""
        try:
            vs_data, is_new_index = await get_or_create_index(index_name, prefix)
            await check_duplicate_files(vs_data, urls, is_new_index)

            # Configure WebBaseLoader to skip SSL verification
            loader = WebBaseLoader(
                urls,
                verify_ssl=False,  # Disable SSL verification for development
                bs_get_text_kwargs={"strip": True, "separator": "\n\n"},
                continue_on_failure=True,  # Continue loading even if some URLs fail
            )
            docs = await asyncio.get_event_loop().run_in_executor(None, loader.aload)
            doc_ids = [str(uuid4()) for _ in range(len(docs))]

            documents = []
            for doc, doc_id in zip(docs, doc_ids):
                chunks = splitter.split_text(doc.page_content)
                for chunk in chunks:
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata=Metadata(
                                **{
                                    "id": doc_id,
                                    **doc.metadata,
                                    "filename": doc.metadata.get("source"),
                                }
                            ).model_dump(mode="python"),
                        )
                    )
            vector_store_ids = await vs.add_to_vector_store(vs_data, [documents])
            sources = [
                FileDetails(filename=url, content_type="text/html", url=url)
                for url in urls
            ]
            await vs.add_vectorstore_and_vectorindex(
                sources,
                doc_ids,
                vector_store_ids,
                vs_data,
                current_user.organization_id,
            )

            return ORJSONResponse(
                content={"status": "success", "ids": vector_store_ids},
                status_code=status.HTTP_201_CREATED,
            )
        except HTTPException:
            raise
        except ValidationError as val_err:
            logger.error("Validation error: %s" % val_err)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err)
            )
        except Exception as e:
            logger.error("Error loading Document: %r" % e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred",
            )

    @action(
        method="GET",
        url_path="{index_name}/list-documents",
        status_code=status.HTTP_200_OK,
    )
    async def list_documents(
        self,
        index_name: str,
    ):
        """List all documents in an index."""
        try:
            documents = await list_all_data_source(index_name)
            # Always add view and download URLs for S3 files to all documents
            for doc in documents:
                if isinstance(doc, dict) and doc.get("s3_key"):
                    try:
                        # Generate fresh pre-signed URLs each time (these expire)
                        doc["view_url"] = await s3_service.generate_download_url(
                            doc["s3_key"]
                        )
                        if doc.get("title"):
                            doc[
                                "download_url"
                            ] = await s3_service.generate_download_url(
                                doc["s3_key"], doc["title"]
                            )
                        else:
                            doc["download_url"] = None
                    except Exception as e:
                        logger.error(f"Error generating URLs for {doc['s3_key']}: {e}")
                        doc["view_url"] = None
                        doc["download_url"] = None
            return documents
        except HTTPException as e:
            raise e

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="GET",
        url_path="{index_name}/documents-details",
        status_code=status.HTTP_200_OK,
    )
    async def get_documents_by_index(
        self,
        index_name: str,
    ):
        """Get detailed information about documents in an index."""
        try:
            async for db in get_session_no_tenant():
                # Query all documents for the given index with their relationships
                query = (
                    select(KnowledgeBase)
                    .options(selectinload(KnowledgeBase.vector_indexes))
                    .options(selectinload(KnowledgeBase.vectorstore))
                    .join(VectorStore)
                    .filter(VectorStore.index_name == index_name)
                )

                result = await db.execute(query)
                docs = result.unique().scalars().all()

                if not docs:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No documents found for index {index_name}",
                    )

                # Process each document and add presigned URLs if needed
                documents_data = []
                for doc in docs:
                    doc_data = jsonable_encoder(doc)
                    if doc.s3_key:
                        doc_data["view_url"] = s3_service.generate_download_url(
                            doc.s3_key
                        )
                        doc_data["download_url"] = s3_service.generate_download_url(
                            doc.s3_key, doc.title
                        )
                    documents_data.append(doc_data)

                return documents_data

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting documents for index {index_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="GET",
        url_path="{index_name}/download-document/{id}",
        status_code=status.HTTP_200_OK,
    )
    async def download_document(
        self,
        index_name: str,
        id: str,
    ):
        """Download a specific document."""
        try:
            async for db in get_session_no_tenant():
                # Find both the document and verify it belongs to the specified index
                query = (
                    select(KnowledgeBase)
                    .join(VectorStore)
                    .filter(KnowledgeBase.id == id)
                    .filter(VectorStore.index_name == index_name)
                )
                result = await db.execute(query)
                doc = result.scalars().first()

                if not doc:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document with ID {id} not found in index {index_name}",
                    )

                if not doc.s3_key:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="No downloadable file associated with this document",
                    )

                try:
                    view_url = await s3_service.generate_download_url(doc.s3_key)
                    download_url = await s3_service.generate_download_url(doc.s3_key)

                    return {
                        "view_url": view_url,
                        "download_url": download_url,
                        "filename": doc.title,
                        "mimetype": doc.mimetype,
                    }
                except Exception as e:
                    logger.error(f"Error generating download URLs: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Error generating download URLs",
                    )

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="GET",
        url_path="{index_name}/document/{id}",
        status_code=status.HTTP_200_OK,
    )
    async def get_document_details(
        self,
        index_name: str,
        id: UUID,
    ):
        """Get detailed information about a specific document."""
        try:
            async for db in get_session_no_tenant():
                # Query document with related data
                query = (
                    select(KnowledgeBase)
                    .options(selectinload(KnowledgeBase.vector_indexes))
                    .options(selectinload(KnowledgeBase.vectorstore))
                    .join(VectorStore)
                    .filter(KnowledgeBase.id == id)
                    .filter(VectorStore.index_name == index_name)
                )

                result = await db.execute(query)
                doc = result.unique().scalars().first()

                if not doc:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document with ID {id} not found in index {index_name}",
                    )

                # Generate presigned URLs if document has S3 key
                document_data = jsonable_encoder(doc)
                if doc.s3_key:
                    document_data["view_url"] = await s3_service.generate_download_url(
                        doc.s3_key
                    )
                    document_data[
                        "download_url"
                    ] = await s3_service.generate_download_url(doc.s3_key)

                return document_data

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error getting document details: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="DELETE",
        url_path="{index_name}/document/{id}",
        status_code=status.HTTP_200_OK,
        response_class=ORJSONResponse,
    )
    async def delete_document(
        self,
        index_name: str,
        id: str,
        vs: VectorStoreService = Depends(get_vectorstore_service),
    ):
        """Delete a document from the vector store."""
        try:
            # Find both the document and verify it belongs to the specified index
            query = (
                select(KnowledgeBase)
                .options(selectinload(KnowledgeBase.vector_indexes))
                .options(selectinload(KnowledgeBase.vectorstore))
                .join(VectorStore)
                .filter(KnowledgeBase.id == id)
                .filter(VectorStore.index_name == index_name)
            )

            async for db in get_session_no_tenant():
                result = await db.execute(query)
                doc = result.unique().scalars().first()

                if not doc:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document with ID {id} not found in index {index_name}",
                    )

                # Get the index details from the document
                vs_index_name = str(doc.vectorstore.index_name)
                prefix = doc.vectorstore.key_prefix or vs_index_name
                doc_ids = [str(index.id.hex) for index in doc.vector_indexes]

                # Delete from vector store
                lc_redis = vs.load_vector_store(
                    index_name=vs_index_name,
                    key_prefix=prefix,
                )
                logger.info(f"Deleting document {id} from index: {vs_index_name}")
                success = await lc_redis.adelete(doc_ids)

                if not success:
                    logger.error(
                        f"Error deleting document {id} from index {vs_index_name}"
                    )

                # Delete from S3 if applicable
                if doc.s3_key:
                    s3_deleted = await asyncio.to_thread(
                        s3_service.delete_file, doc.s3_key
                    )
                    if not s3_deleted:
                        logger.error(f"Failed to delete file from S3: {doc.s3_key}")

                # Delete from database
                await db.delete(instance=doc)
                await db.commit()
                logger.info(
                    f"Successfully deleted document {id} from index {index_name}"
                )

            return ORJSONResponse(
                content={"status": "success"}, status_code=status.HTTP_200_OK
            )

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="GET",
        url_path="indexes",
        status_code=status.HTTP_200_OK,
        response_class=ORJSONResponse,
    )
    async def list_indexes(self) -> Any:
        """List all vector store indexes."""
        try:
            async for db in get_session_no_tenant():
                indexes = await db.execute(select(VectorStore))
                models = indexes.scalars().all()
                return ORJSONResponse(
                    content=[
                        jsonable_encoder(
                            model, exclude={"data_sources", "vector_indexes"}
                        )
                        for model in models
                    ],
                    status_code=status.HTTP_200_OK,
                )
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}",
            )

    @action(
        method="DELETE",
        url_path="indexes/{id}",
        status_code=status.HTTP_200_OK,
        response_class=ORJSONResponse,
    )
    async def delete_index(
        self,
        id: UUID,
        vs_service: VectorStoreService = Depends(get_vectorstore_service),
    ):
        """Delete a vector store index and all its documents."""
        try:
            async for db in get_session_no_tenant():
                index = await db.get(VectorStore, id)
                if not index:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Index not found",
                    )

                # Get all data sources to delete S3 files
                query = select(KnowledgeBase).filter(
                    KnowledgeBase.vector_store_id == id
                )
                result = await db.execute(query)
                data_sources = result.scalars().all()

                # Delete all S3 files associated with this index
                for data_source in data_sources:
                    if data_source.s3_key:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda key=data_source.s3_key: s3_service.delete_file(key),
                        )

                index_name = index.index_name
                # Delete from database first (cascade will delete data sources and vector indexes)
                await db.delete(index)
                await db.commit()

                # Delete from vector store
                try:
                    lc_redis = vs_service.load_vector_store(
                        index_name,
                        index.key_prefix,
                    )
                    lc_redis.index.delete(drop=True)
                except Exception as e:
                    logger.error(f"Error deleting index from vector store: {repr(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error deleting index from vector store: {repr(e)}",
                    )
                return ORJSONResponse(
                    content={"status": "success"}, status_code=status.HTTP_200_OK
                )

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {repr(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {repr(e)}",
            )


from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IDMixin

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase


class VectorIndex(Base, IDMixin):
    __tablename__ = "vector_indexes"

    vector_store_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("vectorstores.id"), index=True
    )
    document_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("knowledge_bases.id"), index=True
    )
    knowledge_base: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="vector_indexes"
    )
from typing import TYPE_CHECKING, List

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IDMixin

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeBase


class VectorStore(Base, IDMixin):
    __tablename__ = "vectorstores"
    index_name: Mapped[str] = mapped_column(
        String, nullable=False, index=True, unique=True
    )
    key_prefix: Mapped[str] = mapped_column(String, nullable=True)
    knowledge_bases: Mapped[List["KnowledgeBase"]] = relationship(
        "KnowledgeBase", back_populates="vectorstore", cascade="all, delete-orphan"
    )

    __mapper_args__ = {"eager_defaults": False}


from typing import TYPE_CHECKING, List
from uuid import UUID

import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, IDMixin, TenantIsolatedMixin

if TYPE_CHECKING:
    from .organization import Organization
    from .vectorstore import VectorStore
    from .vs_index import VectorIndex


class KnowledgeBase(Base, IDMixin, TenantIsolatedMixin):
    __tablename__ = "knowledge_bases"

    vector_store_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("vectorstores.id"), index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=True)
    mimetype: Mapped[str] = mapped_column(String, nullable=True)
    s3_key: Mapped[str] = mapped_column(String, nullable=True)
    s3_url: Mapped[str] = mapped_column(String, nullable=True)
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="knowledge_bases"
    )
    vectorstore: Mapped["VectorStore"] = relationship(
        "VectorStore", back_populates="knowledge_bases"
    )
    vector_indexes: Mapped[List["VectorIndex"]] = relationship(
        "VectorIndex", back_populates="knowledge_base", cascade="all, delete-orphan"
    )

    __mapper_args__ = {"eager_defaults": False}

    __table_args__ = (
        UniqueConstraint(
            "title", "vector_store_id", name="unique_user_title_vector_store"
        ),
        UniqueConstraint("url", "vector_store_id", name="unique_user_url_vector_store"),
        UniqueConstraint("title", "organization_id", name="unique_title_organization"),
        UniqueConstraint("url", "organization_id", name="unique_url_organization"),
    )

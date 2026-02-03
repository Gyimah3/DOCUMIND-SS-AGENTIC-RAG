from datetime import datetime
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FileMetadata(BaseModel):
    """Metadata for a file in S3."""
    
    filename: str
    content_type: str
    size: int
    bucket: str
    key: str
    etag: Optional[str] = None
    version_id: Optional[str] = None


class PresignedUrlResponse(BaseModel):
    """Response for presigned URL generation (not used with public bucket)."""
    
    url: str
    key: str
    bucket: str
    expires_in: int
    fields: Optional[dict[str, str]] = None


class DocumentResponse(BaseModel):
    """Response schema for a single document (DataSource)."""

    id: UUID
    user_id: UUID
    vector_store_id: UUID
    title: str
    url: str | None
    mimetype: str | None
    s3_key: str | None = None
    s3_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    document_id: UUID
    s3_url: str
    s3_key: str


class BulkDeleteResponse(BaseModel):
    """Response after bulk delete of documents in an index."""

    deleted_count: int
    deleted_from_s3: int
    deleted_from_redis: int
    deleted_from_db: int


class VectorStoreModel(BaseModel):
    id: UUID = Field(None, title="ID", description="ID")
    index_name: str = Field(..., title="Index Name", description="Name of the index")
    embedding: str = Field(..., title="Embedding", description="Embedding")
    key_prefix: str | None = Field(None, title="Key Prefix", description="Key Prefix")

    model_config = ConfigDict(from_attributes=True)


class DataSourceResponse(BaseModel):
    id: UUID
    user_id: UUID
    vector_store_id: UUID
    title: str
    url: str | None
    mimetype: str | None
    s3_key: str | None = None
    s3_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AddDocumentResponse(BaseModel):
    status: str = "success"
    ids: list[str]


class DocumentUrlResponse(BaseModel):
    """Response for document URL retrieval."""

    url: str


class GlobalDocumentResponse(BaseModel):
    """Response for document metadata with index information (list-all)."""

    id: UUID
    title: str
    mimetype: str | None = None
    s3_key: str | None = None
    s3_url: str | None = None
    url: str | None = None
    created_at: datetime
    updated_at: datetime
    index_name: str

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    
    documents: list[DocumentResponse]
    total: int

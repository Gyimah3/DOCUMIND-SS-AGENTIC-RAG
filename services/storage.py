"""Simplified AWS S3 storage service for direct uploads to public bucket."""
import asyncio
import mimetypes
from typing import Any, Dict
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger

from app.config import AWSConfig
from errors import BadRequestError, InternalServerError


class AWSStorageService:
    """Simplified S3 service for direct uploads (bucket is public)."""

    def __init__(self, config: AWSConfig):
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
        prefix: str | None = None,
    ) -> str:
        """Generate unique S3 key for file."""
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

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str | None = None,
        prefix: str | None = None,
        bucket: str | None = None,
    ) -> Dict[str, Any]:
        """Upload file directly to S3.
        
        Args:
            file_content: File bytes
            filename: Original filename
            content_type: MIME type (auto-detected if None)
            prefix: S3 folder prefix
            bucket: Bucket name (uses default if None)
            
        Returns:
            Dict with url, key, bucket, etag
        """
        try:
            bucket_name = bucket or self.config.default_bucket

            if len(file_content) > self.config.max_upload_size:
                raise BadRequestError(
                    detail=f"File size exceeds maximum allowed size of {self.config.max_upload_size} bytes",
                )

            # Generate unique key
            key = self._generate_unique_key(filename, prefix)

            # Auto-detect content type if not provided
            if not content_type:
                content_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )

            upload_params = {
                "Bucket": bucket_name,
                "Key": key,
                "Body": file_content,
                "ContentType": content_type,
            }

            async with self._lock:
                response = await asyncio.to_thread(
                    self.client.put_object, **upload_params
                )

            # Generate public URL (bucket is public)
            file_url = f"https://{bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"

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
        except BadRequestError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading file: {e}")
            raise InternalServerError(
                detail="Internal server error",
            )

    def get_public_url(
        self,
        key: str,
        bucket: str | None = None,
    ) -> str:
        """Generate public S3 URL (bucket is public, no presigning needed).
        
        Args:
            key: S3 object key
            bucket: Bucket name (uses default if None)
            
        Returns:
            Public URL string
        """
        bucket_name = bucket or self.config.default_bucket
        return f"https://{bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"

    async def delete_file(
        self,
        key: str,
        bucket: str | None = None,
    ) -> bool:
        """Delete file from S3.
        
        Args:
            key: S3 object key
            bucket: Bucket name (uses default if None)
            
        Returns:
            True if deleted, False if not found
        """
        try:
            bucket_name = bucket or self.config.default_bucket

            params = {"Bucket": bucket_name, "Key": key}

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

    async def file_exists(
        self,
        key: str,
        bucket: str | None = None,
    ) -> bool:
        """Check if file exists in S3.
        
        Args:
            key: S3 object key
            bucket: Bucket name (uses default if None)
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket_name = bucket or self.config.default_bucket

            async with self._lock:
                await asyncio.to_thread(
                    self.client.head_object, Bucket=bucket_name, Key=key
                )

            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NoSuchKey", "404"]:
                return False
            logger.error(f"Error checking file existence: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking file existence: {e}")
            return False

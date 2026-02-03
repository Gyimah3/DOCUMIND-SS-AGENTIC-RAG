from __future__ import annotations

import os
from typing import Literal
from dataclasses import dataclass
from dotenv import load_dotenv
from loguru import logger
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class AWSConfig:
    access_key_id: str
    secret_access_key: str
    region: str
    signature_version: str
    default_bucket: str
    allowed_content_types: list[str] = None
    max_upload_size: int = 50 * 1024 * 1024

class BaseConfig(BaseSettings):
    APP_NAME: str = " DOCU MIND"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    redis_url: str

    db_url: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # JWT / auth (for SecurityService)
    secret_key: SecretStr
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    refresh_token_expire_days: int = 21
    langchain_api_key: str

    aws_access_key_id: str = ""
    aws_secret_access_key: SecretStr = SecretStr("")
    aws_region: str = "us-east-1"
    aws_signature_version: str = "s3v4"
    aws_bucket_name: str 
    # OpenAI (for embeddings / vectorstore)
    openai_api_key: str | None = None

    @property
    def aws_config(self) -> AWSConfig:
        return AWSConfig(
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key.get_secret_value(),
            region=self.aws_region,
            signature_version=self.aws_signature_version,
            default_bucket=self.aws_bucket_name,
        )
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        cache_strings="none",
        frozen=False,
    )



settings = BaseConfig()  # type: ignore
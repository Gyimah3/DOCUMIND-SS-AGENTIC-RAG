"""Base wrapper for document loading"""

import asyncio
import io
import mimetypes
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import IO, Any, AsyncGenerator, Dict, List, Union
from urllib.parse import urlparse

import aiofiles
import aiohttp
from langchain_core.documents import Document
from loguru import logger


class ChunkType(str, Enum):
    text = "text"
    table = "table"
    image = "image"


class NamedBytesIO(io.BytesIO):
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Union[str, int]] = {},
        initial_bytes: bytes = b"",
    ):
        super().__init__(initial_bytes)
        self.name = name
        self.metadata = metadata
        self.mime_type = mimetypes.guess_type(name)[0]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class BaseDocumentLoader(ABC):
    @abstractmethod
    async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
        """Load a single document asynchronously"""
        pass

    async def load_batch_async(
        self, sources: List[Union[str, IO[bytes]]], return_exceptions=False
    ) -> List[Any]:
        """Load multiple documents asynchronously"""
        tasks = [self.load_async(source) for source in sources]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def lazy_load_async(
        self, sources: List[Union[str, IO[bytes]]]
    ) -> AsyncGenerator[List[Document], None]:
        """Load multiple documents lazily"""
        logger.info(f"lazy_load_async called with {len(sources)} sources")
        for i, source in enumerate(sources):
            logger.info(
                f"Source {i}: {getattr(source, 'filename', 'unknown')} - {type(source)}"
            )

        tasks = [self.load_async(source) for source in sources]
        logger.info(f"Created {len(tasks)} tasks")

        for task in asyncio.as_completed(tasks):
            logger.info("Processing completed task...")
            result = await task
            logger.info(f"Task completed, got {len(result)} documents")
            yield result

    @staticmethod
    def is_url(file: str) -> bool:
        try:
            result = urlparse(file)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def _get_filename(source: Union[str, IO[bytes]]) -> str:
        if isinstance(source, str):
            return Path(source).name
        elif isinstance(source, io.IOBase) and hasattr(source, "name"):
            return Path(source.name).name
        else:
            return "unknown_file"

    @staticmethod
    async def read_file_async(file_path: Union[str, Path]) -> bytes:
        """Read a file asynchronously"""
        async with aiofiles.open(file_path, mode="rb") as f:
            return await f.read()

    @staticmethod
    def read_files_sync(file_path: Union[str, Path]) -> bytes:
        raise NotImplementedError()

    @staticmethod
    async def read_url_async(url: str) -> bytes:
        """Read content from a URL asynchronously"""
        # Create SSL context that doesn't verify certificates for development
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    @classmethod
    async def read_source_async(cls, source: Union[str, IO[bytes]]) -> bytes:
        """Read content from a source (file path, URL, or file-like object) asynchronously"""
        if isinstance(source, str):
            if cls.is_url(source):
                return await cls.read_url_async(source)
            else:
                return await cls.read_file_async(source)
        elif isinstance(source, (bytes, bytearray, io.BytesIO)):
            if isinstance(source, io.BytesIO):
                return source.getvalue()
            else:
                return source
        elif hasattr(source, "read"):
            return source.read()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

from typing import Optional
from io import BytesIO, StringIO
import os
from bs4 import BeautifulSoup
from pydantic import BaseModel
import aiohttp
from langchain_community.document_loaders.web_base import WebBaseLoader

from .base import BaseDocumentLoader

WebBaseLoader()


class HTMLMetadata(BaseModel):
    source_type: str
    source: Optional[str] = None
    size: int
    title: Optional[str]


class HTMLLoader(BaseDocumentLoader):
    def __init__(self):
        self.soup = None
        self.metadata = None

    async def load(self, source):
        if isinstance(source, str):
            if os.path.exists(source):
                await self._load_from_disk(source)
            elif source.startswith("http://") or source.startswith("https://"):
                await self._load_from_url(source)
            else:
                raise ValueError(
                    "Invalid source string. Provide a valid file path or URL."
                )
        elif isinstance(source, BytesIO) or isinstance(source, StringIO):
            await self._load_from_memory(source)
        elif isinstance(source, bytes):
            await self._load_from_bytes(source)
        else:
            raise TypeError(
                "Unsupported source type. Provide a file path, URL, in-memory file, or bytes object."
            )

    async def _load_from_disk(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        self.soup = BeautifulSoup(html_content, "html.parser")
        self._set_metadata(
            source_type="file", source=file_path, html_content=html_content
        )

    async def _load_from_memory(self, in_memory_file):
        if isinstance(in_memory_file, BytesIO):
            in_memory_file.seek(0)
            html_content = in_memory_file.read().decode("utf-8")
        elif isinstance(in_memory_file, StringIO):
            in_memory_file.seek(0)
            html_content = in_memory_file.read()
        else:
            raise TypeError("Expected in-memory file to be of type BytesIO or StringIO")
        self.soup = BeautifulSoup(html_content, "html.parser")
        self._set_metadata(source_type="memory", html_content=html_content)

    async def _load_from_bytes(self, byte_content):
        if not isinstance(byte_content, bytes):
            raise TypeError("Expected byte_content to be of type bytes")
        html_content = byte_content.decode("utf-8")
        self.soup = BeautifulSoup(html_content, "html.parser")
        self._set_metadata(source_type="bytes", html_content=html_content)

    async def _load_from_url(self, url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ConnectionError(f"Failed to retrieve content from URL: {url}")
                html_content = await response.text()
        self.soup = BeautifulSoup(html_content, "html.parser")
        self._set_metadata(source_type="url", source=url, html_content=html_content)

    def _set_metadata(self, source_type, html_content, source=None):
        title = self.soup.title.string if self.soup.title else None
        size = len(html_content.encode("utf-8"))
        self.metadata = HTMLMetadata(
            source_type=source_type, source=source, size=size, title=title
        )

    def extract_tables(self):
        if self.soup is None:
            raise ValueError(
                "HTML content not loaded. Please load HTML content before extracting tables."
            )
        tables = self.soup.find_all("table")
        return [str(table) for table in tables]

"""Document loader utilities: categorize_elements and text helpers."""

import os
import time
from pathlib import Path
import base64
from io import BytesIO  
from typing import IO, List, Union  
from PIL import Image
from functools import wraps
from langchain_core.messages import HumanMessage
from loguru import logger
import requests
from langchain_core.documents import Document
from unstructured.documents.elements import Element
from .base import BaseDocumentLoader, ChunkType
from app.utils import load_llm


class TextFileLoader(BaseDocumentLoader):
    async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
        """Load a single text file asynchronously."""
        start_time = time.time()
        content = await self.read_source_async(source)
        text = content.decode("utf-8")
        filename = self._get_filename(source)
        metadata = self._generate_metadata(source, content, start_time)
        return [Document(page_content=text, metadata=metadata)]

    @staticmethod
    def read_files_sync(file_path: Union[str, Path]) -> bytes:
        """Read a file synchronously."""
        with open(file_path, mode="rb") as f:
            return f.read()

    @staticmethod
    def read_url_sync(url: str) -> bytes:
        """Read content from a URL synchronously."""
        response = requests.get(url)
        response.raise_for_status()
        return response.content

    def _generate_metadata(
        self, source: Union[str, IO[bytes]], content: bytes, start_time: float
    ) -> dict:
        """Generate comprehensive metadata for the document."""
        metadata = {
            "filename": self._get_filename(source),
            "file_size": len(content),
            "type": ChunkType.text,
            "encoding": "utf-8",
            "read_time": round(time.time() - start_time, 4),
        }

        if isinstance(source, str) and not self.is_url(source):
            path = Path(source)
            metadata["modified"] = self._get_last_modified_time(path)
            metadata["created_at"] = self._get_creation_date(path)

        return metadata

    @staticmethod
    def _get_source_type(source: Union[str, IO[bytes]]) -> str:
        """Determine the source type: 'file', 'url', or 'memory'."""
        if isinstance(source, str):
            if BaseDocumentLoader.is_url(source):
                return "url"
            else:
                return "file"
        elif hasattr(source, "read"):
            return "memory"
        else:
            return "unknown"

    @staticmethod
    def _get_last_modified_time(path: Path) -> str:
        """Get the last modified time of the file."""
        return time.ctime(os.path.getmtime(path))

    @staticmethod
    def _get_creation_date(path: Path) -> str:
        """Get the creation date of the file (where supported)."""
        try:
            return time.ctime(os.path.getctime(path))
        except Exception:
            return ""

def desired_metadata(metadata: dict):
    return {
        "filename": metadata.get("filename"),
        "page_number": metadata.get("page_number"),
        "filetype": metadata.get("filetype"),
        "languages": metadata.get("languages"),
    }


def convert_image_to_png_and_encode(image_data: bytes) -> str:
    image = Image.open(BytesIO(image_data))
    png_image_io = BytesIO()
    image.save(png_image_io, "PNG")
    b64_image = base64.b64encode(png_image_io.getvalue()).decode()
    return b64_image


def categorize_elements(raw_elements: List[Element]) -> List[Document]:
    """
    Categorize extracted elements into tables, texts, and images.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    documents: List[Document] = []

    for element in raw_elements:
        if element.category == "Image":
            el_dict: dict = element.metadata.to_dict()
            base64_image_data = el_dict.pop("image_base64", "")
            if not base64_image_data:
                continue
            _ = el_dict.pop("table_as_cells", "")
            base64_image_data = base64.b64decode(base64_image_data)
            base64_png_image = convert_image_to_png_and_encode(base64_image_data)
            llm = load_llm()
            message = HumanMessage(content=[
                {"type": "text", "text": "What is the image about, what is in the image, and what is the image trying to convey?, explicitly mention the image in the summary and explan in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_png_image}"}}
            ])
            response = llm.invoke([message])
            summary = response.content
            logger.info(f"Summary: {summary}")
            documents.append(
                Document(
                    page_content=summary,
                    metadata={"type": ChunkType.image, **desired_metadata(el_dict)},
                )
            )
        elif element.category == "Table":
            el_dict = element.metadata.to_dict()
            _ = el_dict.pop("table_as_cells", "")
            text_as_html = el_dict.pop("text_as_html")
            _ = el_dict.pop("image_base64", None)
            documents.append(
                Document(
                    page_content=text_as_html,
                    metadata={"type": ChunkType.table, **desired_metadata(el_dict)},
                )
            )

            documents.append(
                Document(
                    page_content=element.text,
                    metadata={"type": ChunkType.text, **desired_metadata(el_dict)},
                )
            )
        else:
            el_dict = element.metadata.to_dict()
            _ = el_dict.pop("table_as_cells", "")

            documents.append(
                Document(
                    page_content=element.text,
                    metadata={"type": ChunkType.text, **desired_metadata(el_dict)},
                )
            )
    return documents


def time_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            logger.info(f"Execution time: {time.time() - start_time}")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    return wrapper


def async_time_logger(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Execution time: {time.time() - start_time}")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    return wrapper

import mimetypes
from typing import IO, List, Union, Type
from loguru import logger
import requests
from starlette import datastructures
from fastapi import File, UploadFile
from langchain_core.documents import Document
from typing_extensions import Any, Dict
import nltk


from .base import BaseDocumentLoader, NamedBytesIO

from ._html import HTMLLoader
from .md import MarkdownLoader
from .ms_docx import DocxLoader
from .ms_excel import MSExcelLoader
from .ms_pptx import PowerPointLoader
from .pdf import PDFLoader
from .csv_loader import CSVDocumentLoader
from .text import TextFileLoader

__all__ = [
    "PDFLoader",
    "TextFileLoader",
    "MarkdownLoader",
    "DocxLoader",
    "PowerPointLoader",
    "MSExcelLoader",
]


mime_types_to_loaders: Dict[str, Type[BaseDocumentLoader]] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxLoader,
    "text/html": HTMLLoader,
    "text/markdown": MarkdownLoader,
    "text/plain": TextFileLoader,
    "text/csv": CSVDocumentLoader,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": MSExcelLoader,
    "application/vnd.ms-excel": MSExcelLoader,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PowerPointLoader,
    "application/pdf": PDFLoader,
}


class DocumentLoader(BaseDocumentLoader):
    def __init__(
        self,
        sources: List[UploadFile] = File(...),
        extract_images: bool = False,
        extract_tables: bool = True,
        ):
        self.sources = sources
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        nltk.download("averaged_perceptron_tagger_eng")

    async def detect_document_type(
        self, source: Union[str, IO[bytes], datastructures.UploadFile]
    ) -> str:
        """
        Detect the MIME type of the document based on the file extension or magic number.
        """
        if isinstance(source, str):
            if self.is_url(source):
                response = requests.get(source)
                response.raise_for_status()
                return self._detect_by_magic_number(response.content[:4])
            else:
                # Check the file extension
                mime_type, _ = mimetypes.guess_type(source)
                if mime_type:
                    return mime_type
                else:
                    with open(source, "rb") as file:
                        return self._detect_by_magic_number(file.read(4))
        elif isinstance(source, NamedBytesIO):
            return source.mime_type  # type: ignore
        elif (
            isinstance(source, datastructures.UploadFile)
            and source.content_type is not None
        ):
            return source.content_type
        else:
            raise ValueError("Unsupported source type")

    def _detect_by_magic_number(self, signature: bytes) -> str:
        # Add more file signatures as needed
        signatures = {
            b"\x25\x50\x44\x46": "application/pdf",  # PDF
            b"\x50\x4b\x03\x04": "application/zip",  # ZIP (DOCX, XLSX, PPTX)
        }

        for sig, mime_type in signatures.items():
            if signature.startswith(sig):
                return mime_type
        return "application/octet-stream"

    def metadata(self, file: datastructures.UploadFile) -> Dict[str, Any]:
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
        }

    async def load_async(self, source: datastructures.UploadFile) -> List[Document]:
        """
        Load the document asynchronously in async iterator.
        """
        mime_type = await self.detect_document_type(source)
        logger.debug(f"Detected MIME type: {mime_type}")
        if mime_type not in mime_types_to_loaders:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        else:
            loader = mime_types_to_loaders[mime_type]
            source_bytesio = NamedBytesIO(
                initial_bytes=await source.read(),
                name=source.filename,  # type: ignore
                metadata=self.metadata(source),
            )
            # Initialize specialized loader with flags if applicable
            loader_kwargs = {}
            if issubclass(loader, PDFLoader):
                # PDFLoader uses include_images, not extract_images/extract_tables
                loader_kwargs["include_images"] = self.extract_images

            return await loader(**loader_kwargs).load_async(source_bytesio)

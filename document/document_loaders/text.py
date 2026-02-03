"""Document loader for text files with intelligent chunking."""

import os
import time
from pathlib import Path
from typing import IO, List, Union

import requests
from langchain_core.documents import Document

from .base import BaseDocumentLoader, ChunkType


class TextFileLoader(BaseDocumentLoader):
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """
        Initialize TextFileLoader with chunking parameters.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 4000, matching DocxLoader)
            chunk_overlap: Characters to overlap between chunks for context (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
        """Load a text file asynchronously and split into intelligent chunks."""
        start_time = time.time()
        content = await self.read_source_async(source)
        text = content.decode("utf-8")
        filename = self._get_filename(source)
        
        # Split text into chunks
        chunks = self._smart_split_text(text)
        
        # Create documents for each chunk
        documents = []
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = self._generate_metadata(source, content, start_time)
            metadata["chunk_index"] = chunk_idx
            metadata["total_chunks"] = len(chunks)
            
            documents.append(
                Document(page_content=chunk_text, metadata=metadata)
            )
        
        return documents

    def _smart_split_text(self, text: str) -> List[str]:
        """
        Split text intelligently by paragraphs first, then by character limit.
        Preserves document structure better than naive splitting.
        """
        # First, try to split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size, finalize current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add any remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle case where a single paragraph exceeds chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._split_long_paragraph(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [text]

    def _split_long_paragraph(self, text: str) -> List[str]:
        """Split a long paragraph that exceeds chunk_size by sentences."""
        sentences = text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_chunk and len(current_chunk) + len(sentence) > self.chunk_size:
                chunks.append(current_chunk.strip())
                # Add overlap
                current_chunk = current_chunk[-self.chunk_overlap:] + " " + sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]

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


# """Document loader for text files."""

# import os
# import time
# from pathlib import Path
# from typing import IO, List, Union

# import requests
# from langchain_core.documents import Document

# from .base import BaseDocumentLoader, ChunkType


# class TextFileLoader(BaseDocumentLoader):
#     async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
#         """Load a single text file asynchronously."""
#         start_time = time.time()
#         content = await self.read_source_async(source)
#         text = content.decode("utf-8")
#         filename = self._get_filename(source)
#         metadata = self._generate_metadata(source, content, start_time)
#         return [Document(page_content=text, metadata=metadata)]

#     @staticmethod
#     def read_files_sync(file_path: Union[str, Path]) -> bytes:
#         """Read a file synchronously."""
#         with open(file_path, mode="rb") as f:
#             return f.read()

#     @staticmethod
#     def read_url_sync(url: str) -> bytes:
#         """Read content from a URL synchronously."""
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.content

#     def _generate_metadata(
#         self, source: Union[str, IO[bytes]], content: bytes, start_time: float
#     ) -> dict:
#         """Generate comprehensive metadata for the document."""
#         metadata = {
#             "filename": self._get_filename(source),
#             "file_size": len(content),
#             "type": ChunkType.text,
#             "encoding": "utf-8",
#             "read_time": round(time.time() - start_time, 4),
#         }

#         if isinstance(source, str) and not self.is_url(source):
#             path = Path(source)
#             metadata["modified"] = self._get_last_modified_time(path)
#             metadata["created_at"] = self._get_creation_date(path)

#         return metadata

#     @staticmethod
#     def _get_source_type(source: Union[str, IO[bytes]]) -> str:
#         """Determine the source type: 'file', 'url', or 'memory'."""
#         if isinstance(source, str):
#             if BaseDocumentLoader.is_url(source):
#                 return "url"
#             else:
#                 return "file"
#         elif hasattr(source, "read"):
#             return "memory"
#         else:
#             return "unknown"

#     @staticmethod
#     def _get_last_modified_time(path: Path) -> str:
#         """Get the last modified time of the file."""
#         return time.ctime(os.path.getmtime(path))

#     @staticmethod
#     def _get_creation_date(path: Path) -> str:
#         """Get the creation date of the file (where supported)."""
#         try:
#             return time.ctime(os.path.getctime(path))
#         except Exception:
#             return ""

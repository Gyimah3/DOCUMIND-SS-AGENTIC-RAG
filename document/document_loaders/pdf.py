import asyncio
import base64
import io
from pathlib import Path
from typing import IO, Dict, List, Union

import fitz
from langchain_core.messages import HumanMessage
from loguru import logger
import numpy as np
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from PIL import Image
from pydantic import SecretStr
from rapidocr_onnxruntime import RapidOCR

from app.utils import load_llm

from .base import BaseDocumentLoader, ChunkType, NamedBytesIO


class PDFLoader(BaseDocumentLoader):
    def __init__(self, include_images: bool = True, password: str = ""):
        self.include_images = include_images
        self.password = SecretStr(password)
        self.doc: fitz.Document | None = None
        self._ocr: RapidOCR | None = None
        self._upload_filename: str = ""  # uploaded file name (e.g. "Report.pdf")

    @property
    def ocr(self) -> RapidOCR:
        """Lazy-loaded OCR instance to avoid import errors when OCR is not needed."""
        if self._ocr is None:
            try:
                self._ocr = RapidOCR()
            except ImportError as e:
                raise ImportError(
                    f"Failed to initialize OCR. This may be due to missing system dependencies. "
                    f"Original error: {e}. "
                    f"Make sure OpenGL libraries are installed (libgl1-mesa-glx, libglib2.0-0, etc.)"
                ) from e
        return self._ocr

    async def load_async(self, source: str | IO[bytes]) -> List[Document]:
        self._upload_filename = ""
        if isinstance(source, str):
            if self.is_url(source):
                content = await self.read_url_async(source)
                self.doc = fitz.open(stream=content, filetype="pdf")
                if self.doc.needs_pass:
                    self.doc.authenticate(self.password.get_secret_value())
                self.plumber_pdf = pdfplumber.open(
                    source, password=self.password.get_secret_value()
                )
            else:
                self.doc = fitz.open(source)
                if self.doc.needs_pass:
                    self.doc.authenticate(self.password.get_secret_value())
                self.plumber_pdf = pdfplumber.open(
                    source, password=self.password.get_secret_value()
                )

        elif isinstance(source, (io.BytesIO, bytes)):
            if isinstance(source, NamedBytesIO):
                source = io.BytesIO(source.getvalue())
            self.doc = fitz.open(stream=source, filetype="pdf")
            if self.doc.needs_pass:
                self.doc.authenticate(self.password.get_secret_value())
            if isinstance(source, bytes):
                source = io.BytesIO(source)

            self.plumber_pdf = pdfplumber.open(
                source, password=self.password.get_secret_value()
            )

        else:
            raise ValueError(
                "Unsupported source type. Use file path, URL, file-like object, or bytes."
            )

        if self.include_images:
            content, images = await asyncio.gather(
                self.extract_text_and_tables(), self.extract_images()
            )
            return content + images
        return await self.extract_text_and_tables()

    async def extract_text_and_tables(
        self,
    ) -> List[Document]:
        results = []
        if not self.doc:
            raise ValueError("PDF document not loaded.")
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            plumber_page = self.plumber_pdf.pages[page_num]
            text = page.get_text()  # type: ignore
            if not text.strip():
                # If no selectable text, use OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                np_img = np.array(img)
                text, _ = await asyncio.get_event_loop().run_in_executor(
                    None, self.ocr, np_img
                )
                if not text:
                    continue
                text = " ".join([line[1] for line in text])

            tables = await self._extract_tables(plumber_page)
            results.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_number": page_num + 1,
                        "type": ChunkType.text,
                        **self.metadata,
                    },
                )
            )
            results.extend(tables)
        return results

    async def _extract_tables(self, page) -> List[Document]:
        tables = []
        if table := page.extract_table():
            df = pd.DataFrame(table).to_markdown()
            if df:
                tables = [
                    Document(
                        page_content=df,
                        metadata={
                            "type": ChunkType.table,
                            "page_number": page.page_number,
                            **self.metadata,
                        },
                    )
                ]

        return tables

    @property
    def metadata(self) -> Dict[str, Union[str, int]]:
        if not self.doc:
            raise ValueError("PDF document not loaded.")

        metadata: Dict = self.doc.metadata  # type: ignore

        return {
            "filename": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "created_at": metadata.get("creationDate", ""),
            "modified": metadata.get("modDate", ""),
        }
    async def extract_images(self) -> List[Document]:
        """Extract images from all pages concurrently"""
        if not self.doc:
            raise ValueError("PDF document not loaded.")
        
        # Process all pages concurrently
        page_tasks = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_tasks.append(self._extract_images_from_page(page, page_num))
        
        # Gather all results
        all_page_results = await asyncio.gather(*page_tasks, return_exceptions=True)
        
        # Flatten results
        images = []
        for result in all_page_results:
            if isinstance(result, list):
                images.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error extracting images from page: {result}")
        
        return images

    async def _extract_images_from_page(self, page, page_num: int) -> List[Document]:
        """Extract all images from a single page concurrently"""
        image_list = page.get_images(full=True)
        
        if not image_list:
            return []
        
        # Process all images on this page concurrently
        image_tasks = []
        for img_index, img in enumerate(image_list):
            image_tasks.append(
                self._process_single_image(img, page_num, img_index)
            )
        
        # Gather results
        image_results = await asyncio.gather(*image_tasks, return_exceptions=True)
        
        # Filter valid documents
        documents = []
        for result in image_results:
            if isinstance(result, Document):
                documents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error processing image on page {page_num + 1}: {result}")
        
        return documents

    async def _process_single_image(
        self, img, page_num: int, img_index: int
    ) -> Document | None:
        """Process a single image: extract, encode, and get LLM summary"""
        try:
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Skip small images (likely icons/decorations)
            if len(image_bytes) < 30000:
                logger.info(f"Skipping small image on page {page_num + 1}, index {img_index}")
                return None
            
            # Convert to PNG and encode
            image = Image.open(io.BytesIO(image_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Get LLM summary (run in executor to avoid blocking)
            llm = load_llm()
            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": "What is the image about, what is in the image, and what is the image trying to convey? Explicitly mention the image in the summary and explain in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_str}"}
                }
            ])
            
            # Run LLM call in executor to not block event loop
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, [message]
            )
            summary = response.content
            logger.info(f"Generated summary for page {page_num + 1}, image {img_index}")
            
            return Document(
                page_content=summary,
                metadata={
                    "page_number": page_num + 1,
                    "type": ChunkType.image,
                    "image_index": img_index,
                    "image_base64": img_str,  # Optional: store original
                    **self.metadata,
                },
            )
        
        except Exception as e:
            logger.error(f"Error processing image on page {page_num + 1}, index {img_index}: {e}")
            return None
    # async def extract_images(self) -> List[Document]:
    #     images = []
    #     if not self.doc:
    #         raise ValueError("PDF document not loaded.")
    #     for page_num in range(len(self.doc)):
    #         page = self.doc[page_num]
    #         image_list = page.get_images(full=True)
    #         for _, img in enumerate(image_list):
    #             xref = img[0]
    #             base_image = self.doc.extract_image(xref)
    #             image_bytes = base_image["image"]
    #             image = Image.open(io.BytesIO(image_bytes))
    #             buffered = io.BytesIO()
    #             image.save(buffered, format="PNG")
    #             img_str = base64.b64encode(buffered.getvalue()).decode()
    #             llm = load_llm()
    #             message = HumanMessage(content=[
    #                 {"type": "text", "text": "What is the image about, what is in the image, and what is the image trying to convey?, explicitly mention the image in the summary and explan in detail."},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
    #                 }
    #             ])

    #             response = llm.invoke([message])
    #             summary = response.content
    #             logger.info(f"Summary: {summary}")
    #             images.append(
    #                 Document(
    #                     page_content=summary,
    #                     metadata={
    #                         "page_number": page_num + 1,
    #                         "type": ChunkType.image,
    #                         **self.metadata,
    #                     },
    #                 )
    #             )
    #     return images

    def __del__(self):
        if hasattr(self, "doc") and self.doc:
            self.doc.close()

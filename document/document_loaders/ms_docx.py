import asyncio
import base64
from functools import partial
from io import BytesIO
from typing import IO, List, Union
from zipfile import ZipFile

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from loguru import logger
from unstructured.partition.docx import partition_docx

from app.utils import load_llm

from .base import BaseDocumentLoader, ChunkType
from .utils import categorize_elements


class DocxLoader(BaseDocumentLoader):
    def __init__(
        self,
        include_images: bool = True,
        include_tables: bool = True,
        **kwargs,
    ):
        self.include_images = include_images
        self.include_tables = include_tables
        self.loader = partial(
            partition_docx,
            infer_table_structure=self.include_tables,
            extract_images_in_docx=self.include_images,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            **kwargs,
        )

    async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
        try:
            logger.info(f"Loading DOCX {source}")
            content = await self.read_source_async(source)
            docs = await asyncio.to_thread(self.loader, file=BytesIO(content))

            images = []
            if self.include_images:
                images = await self.extract_images_async(content)

            filename = self._get_filename(source)
            for doc in docs:
                doc.metadata.filename = filename

            documents = categorize_elements(docs)
            return documents + images
        except Exception as e:
            logger.error(f"Error loading DOCX asynchronously: {repr(e)}")
            raise e

    @staticmethod
    async def extract_images_async(content: bytes) -> List[Document]:
        images = []
        try:
            logger.info("Extracting images from DOCX")
            with ZipFile(BytesIO(content)) as archive:
                for file in archive.filelist:
                    if (
                        file.filename.startswith("word/media/")
                        and file.file_size > 30000
                    ):
                        image_content = archive.read(file.filename)
                        # base64_image_data = base64.b64decode(image_content).decode("utf-8")
                        base64_image_data = base64.b64encode(image_content).decode("utf-8")
                        llm = load_llm()
                        message = HumanMessage(content=[
                            {"type": "text", "text": "What is the image about, what is in the image, and what is the image trying to convey?, explicitly mention the image in the summary and explan in detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image_data}"}}
                        ])
                        response = llm.invoke([message])
                        summary = response.content
                        logger.info(f"Summary: {summary}")
                        images.append(
                            Document(
                                    page_content=summary,
                                metadata={
                                    "filename": file.filename,
                                    "type": ChunkType.image,
                                    "created_at": file.date_time,
                                    "size": file.file_size,
                                },
                            )
                        )
        except Exception as e:
            logger.error(f"Error extracting images asynchronously: {repr(e)}")
            raise e
        return images

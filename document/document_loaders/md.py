"""This module is responsible for loading markdown files and extracting images from them."""

import asyncio
import base64
import re
from functools import partial
from io import BytesIO, StringIO
from typing import List

import aiohttp
import requests
from langchain_core.documents import Document
from loguru import logger
from unstructured.partition.md import partition_md

from .base import BaseDocumentLoader
from .utils import categorize_elements


class MarkdownLoader(BaseDocumentLoader):
    def __init__(self, include_images: bool = True, **kwargs):
        self.include_images = include_images
        self.kwargs = kwargs

    async def load_async(self, source) -> List[Document]:
        loader = partial(
            partition_md,
            extract_images_in_md=self.include_images,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            **self.kwargs,
        )
        filename = self._get_filename(source)
        try:
            logger.info(f"Loading MD: {source}")
            if isinstance(source, str):
                if self.is_url(source):
                    content = await self.read_url_async(source)
                    md_file = BytesIO(content)
                    docs = await asyncio.to_thread(
                        loader,
                        file=md_file,
                        metadata_filename=source.split("/")[-1],
                    )
                else:
                    docs = await asyncio.to_thread(loader, file=source)
            elif isinstance(source, BytesIO):
                docs = await asyncio.to_thread(loader, file=source)

            else:
                raise Exception(
                    "Invalid source type. Must be file path, URL, or StringIO."
                )
            for doc in docs:
                doc.metadata.filename = filename

            images = []
            if self.include_images:
                images = await self.extract_images(source)

            return categorize_elements(docs) + images

        except Exception as e:
            logger.error(f"Error loading MD: {source}")
            raise e

    @staticmethod
    async def extract_images(source, wait=2, retries=3) -> List[Document]:
        logger.info(f"Extracting images from MD: {source}")
        filename = MarkdownLoader._get_filename(source)
        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                response = requests.get(source)
                md_file = StringIO(response.content.decode("utf-8"))
            else:
                with open(source, "r") as f:
                    md_file = StringIO(f.read())
        elif isinstance(source, StringIO):
            md_file = source
        elif isinstance(source, BytesIO):
            md_file = StringIO(source.getvalue().decode("utf-8"))
        else:
            raise Exception("Invalid source type. Must be file path, URL, or StringIO.")

        async def download_image(
            img, image_name, session: aiohttp.ClientSession, wait=2, retries=3
        ) -> Document | None:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
                "Accept-Encoding": "none",
                "Accept-Language": "en-US,en;q=0.8",
                "Connection": "keep-alive",
            }
            try:
                async with session.get(img, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    content = base64.b64encode(content).decode("utf-8")
                return Document(
                    page_content=content,
                    metadata={
                        "url": str(img),
                        "type": "image",
                        "filename": filename,
                        "image_name": image_name,
                    },
                )

            except (aiohttp.ClientConnectionError, aiohttp.ConnectionTimeoutError) as e:
                logger.error("%s: Retrying..." % str(e))
                if retries > 0:
                    await asyncio.sleep(wait)
                    return await download_image(
                        img, filename, session, wait, retries - 1
                    )
                else:
                    return None
            except Exception:
                pass

        images = []
        pattern = r'!\[([^\]]*)\]\s*(\[[^\]]*\]|\((?:(?:https?://)?(?:(?:[a-zA-Z0-9\-_]+\.)+[a-zA-Z]{2,})?(?:/[^)\s]+)?)(?:\s+"[^"]*")?\))'
        matches = re.finditer(pattern, md_file.read())

        # Create SSL context that doesn't verify certificates for development
        import ssl

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for match in matches:
                url_or_ref = match.group(2)

                if url_or_ref.startswith("["):
                    url_or_ref = url_or_ref[1:-1]  # Remove brackets
                else:
                    url_or_ref = url_or_ref[1:-1]
                    parts = url_or_ref.split('" ', 1)
                    url = parts[0]
                    url_or_ref = url
                image_name = str(source)
                tasks.append(
                    download_image(url_or_ref, image_name, session, wait, retries)
                )
            images = await asyncio.gather(*tasks, return_exceptions=False)
            images = [img for img in images if img is not None]
        return images

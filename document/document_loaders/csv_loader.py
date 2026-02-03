"""CSV document loader — one chunk per row with row_number metadata."""

import io
from typing import IO, List, Union

import pandas as pd
from langchain_core.documents import Document
from pydantic_core import to_json

from .base import BaseDocumentLoader, ChunkType


class CSVDocumentLoader(BaseDocumentLoader):
    """Load CSV files into per-row Document chunks with row_number metadata."""

    def __init__(self) -> None:
        self.filename: str | None = None

    async def load_async(self, source: Union[str, IO[bytes]]) -> List[Document]:
        self.filename = self._get_filename(source)
        data = await self.read_source_async(source)

        df = pd.read_csv(io.BytesIO(data))
        df.dropna(axis=0, how="all", inplace=True)
        df.fillna("missing", inplace=True)

        if df.empty:
            return []

        rows = df.apply(lambda x: str(x.to_dict()), axis=1).tolist()
        return [
            Document(
                page_content=to_json(row),
                metadata={
                    "type": ChunkType.text,
                    "filename": self.filename,
                    "row_number": row_idx + 1,
                },
            )
            for row_idx, row in enumerate(rows)
        ]

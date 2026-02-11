from typing import Any, Dict, List, Literal, Optional, Type, Union
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.vectorstores.redis import Redis, RedisTag, RedisFilter
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from app.config import settings
from app.utils import load_vector_store


class FileNameFilter(BaseModel):
    filename: str = Field(
        ...,
        description="The name of the file to filter the search results. The name should be an exact match.",
    )


class CreatedAtFilter(BaseModel):
    """
    A filter for the creation date of the document.
    """

    year: int | None = Field(None, description="The year of the document.")
    month: int | None = Field(None, description="The month of the document.")
    day: int | None = Field(None, description="The day the document.")
    operator: Literal["lt", "lte", "gt", "gte", "eq"] = Field(
        ..., description="The operator to use for comparison."
    )


class ModifiedFilter(CreatedAtFilter):
    """
    A filter for the modification date of the document.
    """

    pass


class PageNumberFilter(BaseModel):
    """
    A filter for the page number of the document.
    """

    page_number: int = Field(..., description="The page number of the document.")
    operator: Literal["lt", "lte", "gt", "gte", "eq"] = Field(
        ..., description="The operator to use for comparison."
    )


class TypeFilter(BaseModel):
    """
    A filter for the type of the document.
    """

    type: Literal["image", "text", "table"] = Field(
        ..., description="The type of the document."
    )


FILTERS = List[
    Union[FileNameFilter, CreatedAtFilter, ModifiedFilter, PageNumberFilter, TypeFilter]
]


class ToolInput(BaseModel):
    query: str = Field(
        ...,
        description="The user query to search for relevant information. Should be more comprehensive and search-friendly while maintaining the original intent.",
    )
    filters: Optional[FILTERS] = Field(
        None,
        description="Filters to apply to the search results. The filters should be used to narrow down the search results based on specific criteria.",
    )


def prepare_filters(filters: FILTERS) -> Any:
    metadata_filters = []
    for filter in filters:
        if isinstance(filter, FileNameFilter):
            metadata_filters.append(RedisFilter.tag("filename") == filter.filename)


class InformationLookupTool(BaseTool):
    name = "information_lookup_tool"
    description = "A tool for searching the user's uploaded documents to find relevant information. Use this for ANY question about document content, file data, names, numbers, topics, summaries, or anything that could be answered from the documents."
    args_schema: Type[BaseModel] | None = ToolInput
    embedding_model: str
    index_name: str | None = None  # Optional: if None, searches all indexes
    key_prefix: str | None = None
    top_k: int = 5
    user_index_names: List[str] | None = None  # Scoped indexes for multi-index search
    rerank: bool = True
    retriever: BaseRetriever

    @root_validator(pre=True)
    def add_retriever(cls, values: Dict) -> Dict:
        if (embedding_model := values.get("embedding_model")) is None:
            raise ValueError("The 'embedding_model' argument is required.")

        # index_name is now optional - if None, searches all indexes
        index_name = values.get("index_name")
        key_prefix = values.get("key_prefix", index_name if index_name else "default")
        user_index_names = values.get("user_index_names")

        # Conditionally load the reranker module
        reranker_module = None
        if values.get("rerank", True):
            from services import reranker as reranker_module

        retriever = load_vector_store(
            redis_url=settings.redis_url,
            index_name=index_name or "default",  # Placeholder, not used when index_name is None
            key_prefix=key_prefix,
            embedding_model=embedding_model,
            existing=False,
        ).as_retriever(
            index_name=index_name,  # Pass None to search all indexes
            key_prefix=key_prefix,
            search_kwargs={"k": values.get("top_k", 5)},
            user_index_names=user_index_names,
            reranker=reranker_module,
        )
        return {**values, "retriever": retriever}

    @staticmethod
    def _format_results(results: List) -> List[str]:
        """Format results with source metadata so the LLM can cite correctly."""
        formatted = []
        for r in results:
            meta = getattr(r, "metadata", {})
            filename = meta.get("filename", "Unknown")
            page = meta.get("page_number", 0)
            source = f"[Source: {filename}, Page {page}]" if page else f"[Source: {filename}]"
            formatted.append(f"{source}\n{r.page_content}")
        return formatted

    def _run(
        self,
        query: str,
        filters: Optional[FILTERS] = None,
    ) -> List:
        results = self.retriever.invoke(query)
        return self._format_results(results)

    async def _arun(
        self,
        query: str,
        filters: Optional[FILTERS] = None,
    ) -> List:
        logger.info("Embedding: {}", self.embedding_model)
        results = await self.retriever.ainvoke(query)
        logger.info("Results: {}", results)
        return self._format_results(results)
# def vectorstore(self, index_name: str, embedding: "Embeddings") -> "VectorStore":
#     vs = Redis.from_existing_index(
#         redis_url=settings.redis_url,
#         index_name=index_name,
#         embedding=embedding,
#         schema=INDEX_SECHEMA_PATH,
#     )
#     return vs



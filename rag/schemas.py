from pydantic import BaseModel, Field


class QueryRewrite(BaseModel):
    """
    A model for rewriting user queries to improve retrieval effectiveness.
    """

    rewritten_query: str = Field(
        ...,
        description="The rewritten query optimized for retrieval. Should be more comprehensive and search-friendly while maintaining the original intent.",
    )


class RagInputSchema(BaseModel):
    message: str

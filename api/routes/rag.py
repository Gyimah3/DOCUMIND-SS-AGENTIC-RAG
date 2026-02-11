"""RAG router: invoke chat with thread_id (BaseRouter + @action)."""

import json
from typing import Any, AsyncGenerator

from fastapi import Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

from api.decorator import action
from api.routes.base_di import BaseRouter
from app.config import settings
from database.models.user import User
from middleware.dep import get_current_user
from rag.engine import RagEngine
from rag.schemas import RagInputSchema
from rag.tools import InformationLookupTool

from database.session import db_session_manager
from sqlalchemy import select
from database.models.vectorstore import VectorStore
from database.models.data_source import DataSource
from loguru import logger
from services.pii_masker import mask_pii

def _get_llm():
    """Return ChatOpenAI LLM from app config (OpenAI API key)."""
    from langchain_openai import ChatOpenAI

    api_key = getattr(settings, "openai_api_key", None) or ""
    if not api_key:
        raise ValueError(
            "OpenAI API key is not set. Add OPENAI_API_KEY to your .env or set openai_api_key in app config."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )


_checkpointer = MemorySaver()


class RagRouter(BaseRouter):
    """Router for RAG chat: invoke with thread_id, optional index_name, embedding_model, top_k."""

    def __init__(self) -> None:
        router = APIRouter(prefix="/chat", tags=["rag"])
        super().__init__(router)

    @action(
        method="POST",
        url_path="invoke/{thread_id}",
        response_model=None,
        status_code=200,
        dependencies=[Depends(get_current_user)],
        summary="Invoke RAG",
        description="Run RAG over the given index; stream response and optional used_docs. Requires login.",
    )
    async def invoke_rag(
        self,
        thread_id: str,
        input_body: RagInputSchema,
        current_user: User = Depends(get_current_user),
        index_name: str | None = Query(None, description="Vectorstore index name (optional; if None, searches all indexes)"),
        embedding_model: str = Query("text-embedding-3-small", description="Embedding model"),
        top_k: int = Query(5, description="Number of chunks to retrieve"),
    ) -> Any:
        """Stream RAG response; body must include { \"message\": \"...\" }."""

        # If index_name is provided, find associated file titles
        file_titles_str = None
        if index_name:

            async for db in db_session_manager.get_db():
                # Fetch VectorStore with given index_name
                vectorstore = await db.scalar(
                    select(VectorStore).where(VectorStore.index_name == index_name)
                )
                if vectorstore is not None:
                    # Get DataSource entries for this VectorStore
                    data_sources = (
                        await db.execute(
                            select(DataSource).where(DataSource.vector_store_id == vectorstore.id)
                        )
                    ).scalars().all()
                    # Get all titles (filenames)
                    titles = [ds.title for ds in data_sources if hasattr(ds, "title")]
                    file_titles_str = ", ".join(titles) if titles else None
                    logger.info(f"File titles: {file_titles_str}")
                    break
        else:
            file_titles_str = "all documents"

        # When searching all documents, fetch only the current user's index names
        user_index_names: list[str] | None = None
        if not index_name:
            async for db in db_session_manager.get_db():
                user_vectorstores = (
                    await db.execute(
                        select(VectorStore.index_name).where(VectorStore.user_id == current_user.id)
                    )
                ).scalars().all()
                user_index_names = list(user_vectorstores)
                logger.info(f"User {current_user.id} has {len(user_index_names)} indexes: {user_index_names}")
                break

        tool = InformationLookupTool(
            embedding_model=embedding_model,
            index_name=index_name,
            top_k=top_k,
            user_index_names=user_index_names,
            rerank=True,
        )
        engine: CompiledGraph = RagEngine(
            llm=_get_llm(),
            lookup_tool=tool,
            multimodal=False,
            active_file_name=file_titles_str,
        ).compile_graph(checkpointer=_checkpointer)

        async def stream_model_response() -> AsyncGenerator[str, None]:
            async for stream in engine.astream_events(
                {"messages": [("human", input_body.message)]},
                version="v2",
                config={"configurable": {"thread_id": thread_id}},
            ):
                event = stream.get("event")
                if event == "on_chat_model_stream":
                    chunk = stream.get("data", {}).get("chunk")
                    if chunk and (content := getattr(chunk, "content", None)):
                        content_str = str(content)
                        if (content_str.strip().startswith("{") and 
                            any(keyword in content_str for keyword in ["intent", "rephrased", "confidence"])):
                            continue
                        
                        if isinstance(content, list):
                            if content and isinstance(content[0], dict) and "text" in content[0]:
                                yield json.dumps({"stream": mask_pii(str(content[0]["text"]))}) + "\n"
                            elif not (content and isinstance(content[0], dict) and content[0].get("type") == "tool_use"):
                                yield json.dumps({"stream": mask_pii(str(content))}) + "\n"
                        else:
                            yield json.dumps({"stream": mask_pii(content) if isinstance(content, str) else content}) + "\n"
                elif event == "on_retriever_end":
                    results = stream.get("data", {}).get("output", [])
                    used = []
                    for r in results:
                        if hasattr(r, "page_content"):
                            used.append({
                                "page_content": mask_pii(r.page_content),
                                "metadata": getattr(r, "metadata", {}),
                            })
                        elif isinstance(r, dict):
                            used.append(r)
                        else:
                            used.append({"content": mask_pii(str(r))})
                    yield json.dumps({"used_docs": used}) + "\n"

        return StreamingResponse(stream_model_response(), media_type="application/x-ndjson")

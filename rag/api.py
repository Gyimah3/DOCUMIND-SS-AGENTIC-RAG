# import json
# from typing import Any, AsyncGenerator
# from fastapi import Depends, Query
# from fastapi.responses import StreamingResponse
# from fastapi.routing import APIRouter
# from sse_starlette.sse import EventSourceResponse
# from langgraph.graph.graph import CompiledGraph
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
# from backend.auth.services.dependencies import AccessBearer
# from rag.engine import RagEngine
# from rag.tools import InformationLookupTool
# from utilities.embeddings import EMBEDDING_MODELS
# from .schemas import RagInputSchema
# from backend.library.utils import llm, load_embedding_model
# # from backend


# router = APIRouter(prefix="/chat", tags=["rag"])
# security = AccessBearer()
# checkpointer = MemorySaver()


# @router.post("/invoke/{thread_id}", dependencies=[Depends(security)])
# async def invoke_rag(
#     thread_id: str,
#     input: RagInputSchema,
#     index_name: str = Query(...),
#     embedding_model: EMBEDDING_MODELS = Query("text-embedding-3-small"),
#     top_k: int = Query(4),
# ) -> Any:
#     tool = InformationLookupTool(  # type: ignore
#         embedding_model=embedding_model,
#         index_name=index_name,
#         top_k=top_k,
#     )
#     engine: CompiledGraph = RagEngine(
#         llm=llm, lookup_tool=tool, multimodal=False
#     ).compile_graph(checkpointer=checkpointer)

#     async def stream_model_response() -> AsyncGenerator[str, None]:
#         async for stream in engine.astream_events(
#             {"messages": [("human", input.message)]},
#             version="v2",
#             config={"configurable": {"thread_id": thread_id}},
#         ):
#             if (event := stream["event"]) == "on_chat_model_stream":
#                 content = stream["data"]["chunk"].content  # type: ignore
#                 if content:
#                     if isinstance(content, list):
#                         if "text" in content[0]:
#                             yield json.dumps({"stream": str(content[0]["text"])}) + "\n"
#                         elif content[0].get("type") != "tool_use":
#                             yield json.dumps({"stream": str(content)}) + "\n"
#                     else:
#                         yield json.dumps({"stream": content}) + "\n"
#             elif event == "on_retriever_end":
#                 results = stream["data"]["output"]  # type: ignore
#                 yield (
#                     json.dumps({"used_docs": [result.dict() for result in results]})
#                     + "\n"
#                 )

#     return StreamingResponse(stream_model_response())

import operator
import os
from typing import Annotated, Dict, Literal, Sequence, TypedDict, List
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field
from .prompts import GENERATE_RESPONSE_PROMPT, INTENT_CHECK_PROMPT, QUERY_REWRITE_PROMPT
from .tools import InformationLookupTool
from app.config import settings


os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "documind"


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    is_document_query: str  # "DOCUMENT_QUERY" or "GENERAL_CONVERSATION"
    confidence: float
    keywords: list[str] 


INDEX_SECHEMA_PATH = "backend/document/services/schema.yml"

class UserIntent(BaseModel):
    """Classify user intent"""
    intent: Literal["DOCUMENT_QUERY", "GENERAL_CONVERSATION"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the intent classification")


class RephrasedQuery(BaseModel):
    """Rephrased query for better search"""
    rephrased: str = Field(description="Improved query", max_length=1000)
    keywords: list[str] = Field(description="Key search terms extracted", max_items=5)


class RagEngine:
    def __init__(
        self,
        llm: BaseChatModel,
        lookup_tool: InformationLookupTool,
        multimodal: bool = False,
        active_file_name: str | None = None,
    ):
        self.llm = llm
        self.lookup_tool = lookup_tool
        self.multimodal = multimodal
        self.tool_output_parser = JsonOutputToolsParser()
        self.active_file_name = active_file_name

        self.intent_classifier = llm.with_structured_output(
            UserIntent,
            method="json_schema"
        )
        self.query_rephraser = llm.with_structured_output(
            RephrasedQuery,
            method="json_schema"
        )

    async def detect_intent(self, state: GraphState) -> Dict:
        """
        NEW: Classify if user is asking about documents or making general conversation.
        This prevents "I don't know" responses to "hello", "thank you", etc.
        """
        last_message = state["messages"][-1]
        
        # Handle both tuple and object formats
        if isinstance(last_message, tuple):
            user_input = last_message[1] if len(last_message) > 1 else str(last_message)
        else:
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        result = await self.intent_classifier.ainvoke([
            SystemMessage(content=INTENT_CHECK_PROMPT.format(input=user_input))
        ])
        
        return {
            "messages": [AIMessage(content=f"[Intent: {result.intent}]")],
            "is_document_query": result.intent,
            "confidence": result.confidence,
        }
    async def rephrase_query_node(self, state: GraphState) -> Dict:
        """
        Rephrase document queries for better retrieval.
        Only called if is_document_query == "DOCUMENT_QUERY"
        """
        last_message = state["messages"][-1]
        
        # Handle both tuple and object formats
        if isinstance(last_message, tuple):
            user_input = last_message[1] if len(last_message) > 1 else str(last_message)
        else:
            user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        result = await self.query_rephraser.ainvoke([
            SystemMessage(content=QUERY_REWRITE_PROMPT.format(input=user_input))
        ])
        
        return {
            "messages": [AIMessage(content=f"[Rephrased: {result.rephrased}]")],
            "keywords": result.keywords,
        }
    # async def assistant(self, state: GraphState) -> Dict:
    #     system_content = GENERATE_RESPONSE_PROMPT.format(
    #         company="seedstars",
    #         active_file_name=self.active_file_name,
    #     )
    async def assistant(self, state: GraphState) -> Dict:
        """
        Generate response using information_lookup_tool.
        Now aware of intent from detect_intent node.
        """
        system_content = GENERATE_RESPONSE_PROMPT.format(
            company="seedstars",
            active_file_name=self.active_file_name,
        )
        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = await self.llm.bind_tools(tools=[self.lookup_tool]).ainvoke(messages)
        return {"messages": [response]}

    async def lookup(self, state: GraphState) -> Dict:
        """Execute the information lookup tool"""
        # ToolNode handles this, I don't need to implement
        pass

    def should_rephrase(self, state: GraphState) -> str:
        """
        NEW: Route based on intent.
        - DOCUMENT_QUERY → rephrase the query
        - GENERAL_CONVERSATION → skip to assistant
        """
        intent = state.get("is_document_query", "DOCUMENT_QUERY")
        
        if intent == "DOCUMENT_QUERY":
            return "rephrase"
        else:
            return "assistant"

    # def should_lookup(self, state: GraphState) -> str:
    #     """Existing logic: check if tool was called"""
    #     last_msg = state["messages"][-1]
    #     # Check if last message has tool calls
    #     if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
    #         return "lookup"
    #     return "end"

    def should_lookup(self, state: GraphState) -> str:
        last_msg = state["messages"][-1]
        if len(self.tool_output_parser.invoke(last_msg)) > 0:
            return "lookup"
        return "end"
    
    # async def rephrase_query(self, state: GraphState) -> Dict:
    #     system_content = QUERY_REWRITE_PROMPT.format(

    #     )
    #     messages = [SystemMessage(content=system_content)] + state["messages"]
    #     response = await self.llm.bind_tools(tools=[self.lookup_tool]).ainvoke(messages)
    #     return {"messages": [response]}

    def compile_graph(
        self,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_before: Sequence[str] | None = None,
    ) -> CompiledGraph:
        graph = StateGraph(GraphState)
        # graph.add_node("assistant", self.assistant)
        # graph.add_node("lookup", ToolNode(tools=[self.lookup_tool]))
        # graph.set_entry_point("assistant")
        # graph.add_conditional_edges(
        #     "assistant", self.should_lookup, {"lookup": "lookup", "end": END}
        # )
        # graph.add_edge("lookup", "assistant")
        # Add nodes
        graph.add_node("detect_intent", self.detect_intent)
        graph.add_node("rephrase", self.rephrase_query_node)
        graph.add_node("assistant", self.assistant)
        graph.add_node("lookup", ToolNode(tools=[self.lookup_tool]))
        
        # Set entry point
        graph.set_entry_point("detect_intent")
        
        # After intent detection, decide whether to rephrase
        graph.add_conditional_edges(
            "detect_intent",
            self.should_rephrase,
            {
                "rephrase": "rephrase",
                "assistant": "assistant",
            }
        )
        
        # After rephrase, go to assistant
        graph.add_edge("rephrase", "assistant")
        
        # After assistant, check if lookup needed
        graph.add_conditional_edges(
            "assistant",
            self.should_lookup,
            {
                "lookup": "lookup",
                "end": END,
            }
        )
        
        # After lookup, go back to assistant for response
        graph.add_edge("lookup", "assistant")

        return graph.compile(
            checkpointer=checkpointer, interrupt_before=interrupt_before
        )

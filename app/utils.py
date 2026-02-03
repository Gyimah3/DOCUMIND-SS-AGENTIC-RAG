import asyncio
import os
import time
from typing import Callable, Any, Literal, Tuple, Dict
import redis
from loguru import logger
from services.redis_vectorstore import RedisVectorStore as DocumindRedisVectorStore
from collections import OrderedDict
from functools import wraps
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from app.config import settings
from utilities.embeddings import EMBEDDING_MODELS

api_key = getattr(settings, "openai_api_key", None) or ""
if not api_key:
    raise ValueError("OpenAI API key is not set. Add OPENAI_API_KEY to your .env or set openai_api_key in app config.")
os.environ["OPENAI_API_KEY"] = api_key

class RetryRun:
    def __init__(self, max_retries: int = 3, delay: int = 1):
        self.max_retries = max_retries
        self.delay = delay

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args: Tuple, **kwargs: Dict) -> Any:
            for _ in range(self.max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(e)
                    await asyncio.sleep(self.delay)
            return None

        @wraps(func)
        def sync_wrapper(*args: Tuple, **kwargs: Dict) -> Any:
            for _ in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(e)
                    time.sleep(self.delay)
            return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class CacheRun:
    cache: Dict[str, Tuple[Any, float]] = OrderedDict()

    def __init__(self, max_size: int = 100, ttl: int = 60):
        self.max_size = max_size
        self.ttl = ttl

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def check_ttl(key: str, current_time: float) -> Any:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if current_time - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]

        @wraps(func)
        async def async_wrapper(*args: Tuple, **kwargs: Dict) -> Any:
            key = f"{func.__name__}-{args}{kwargs}"
            current_time = time.time()
            result = check_ttl(key, current_time)
            if result:
                logger.info("Cache hit")
                return result
            logger.info("Cache miss")
            result = await func(*args, **kwargs)
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # type: ignore[call-arg]
            self.cache[key] = (result, current_time)
            return result

        @wraps(func)
        def sync_wrapper(*args: Tuple, **kwargs: Dict) -> Any:
            key = f"{args}{kwargs}"
            current_time = time.time()
            result = check_ttl(key, current_time)
            if result:
                logger.info("Cache hit")
                return result
            logger.info("Cache miss")
            result = func(*args, **kwargs)
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # type: ignore[call-arg]
            self.cache[key] = (result, current_time)
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    async def wrapper_async(*args: Tuple, **kwargs: Dict) -> Any:
        start = time.time()
        result = await func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start}")
        return result

    @wraps(func)
    def wrapper_sync(*args: Tuple, **kwargs: Dict) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start}")
        return result

    if asyncio.iscoroutinefunction(func):
        return wrapper_async
    else:
        return wrapper_sync


EMBEDDING_MODELS_DICT: Dict[str, str] = {
    "text-embedding-3-small": "openai",
    "text-embedding-3-large": "openai",
    "text-embedding-ada-002": "openai",
    # "embed-english-v3.0": "cohere",
    # "all‑MiniLM‑L6‑v2.gguf2.f16.gguf": "gpt4all",
    "sentence-transformers/all-mpnet-base-v2": "huggingface",
}

EMBEDDING_MODELS_PROVIDER = Literal["openai", "cohere", "huggingface"]


def load_embedding_model(model: EMBEDDING_MODELS) -> Embeddings:
    if EMBEDDING_MODELS_DICT.get(model) == "huggingface":
        return HuggingFaceHubEmbeddings(model=model)  # type: ignore[call-arg]
    elif EMBEDDING_MODELS_DICT.get(model) == "openai":
        return OpenAIEmbeddings(model=model)
    else:
        raise ValueError("Only HuggingFace and OpenAI models are supported")


def load_llm() -> BaseChatModel:
    model = getattr(settings, "LLM", "gpt-4o")
    provider = getattr(settings, "LLM_PROVIDER", "openai")
    temperature = getattr(settings, "LLM_TEMPERATURE", 0)
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            stream_usage=True,
            streaming=True,
            temperature=temperature,
        )
    else:
        raise ValueError("Invalid provider")


def load_vector_store(
    redis_url,
    index_name: str,
    key_prefix: str,
    embedding_model: EMBEDDING_MODELS,
    existing: bool = False,
):
    """Load vector store for RAG. Uses custom RedisVectorStore (HASH schema, robust result parsing)."""
    embedding = load_embedding_model(model=embedding_model)
    client = redis.Redis.from_url(redis_url)
    return DocumindRedisVectorStore(client=client, embedding_func=embedding)


# Lazy: only call load_llm() when something needs it (avoids requiring LLM in config at import time)
llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    global llm
    if llm is None:
        llm = load_llm()
    return llm

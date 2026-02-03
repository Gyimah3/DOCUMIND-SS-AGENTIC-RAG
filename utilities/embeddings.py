from typing import Literal


EMBEDDING_MODELS = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    # "embed-english-v3.0",
    # "all‑MiniLM‑L6‑v2.gguf2.f16.gguf",
    "sentence-transformers/all-mpnet-base-v2",
]

EMBEDDING_MODELS_PROVIDER = Literal["openai", "cohere", "huggingface"]

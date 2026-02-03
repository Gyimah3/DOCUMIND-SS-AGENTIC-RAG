from database.registry import model_registry
from .user import User
from .base_model import Base
from .vectorstore import VectorStore
from .data_source import DataSource
from .vs_index import VectorIndex


model_registry.register_model("Base")
model_registry.register_model("User")
model_registry.register_model("VectorStore")
model_registry.register_model("DataSource")
model_registry.register_model("VectorIndex")

__all__ = [
    "User",
    "Base",
    "VectorStore",
    "DataSource",
    "VectorIndex",
]
"""Router registry: wires all routers to the FastAPI app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from api.routes.auth import AuthRouter
from api.routes.docs import DocsRouter
from api.routes.rag import RagRouter
from api.routes.vectorstore import VectorstoreRouter
from api.routes.documents import DocumentRouter

if TYPE_CHECKING:
    from app.container import ServiceContainer
    from app.config import BaseConfig


class RouterRegistry:
    """Registers all API routers with the FastAPI application."""

    def __init__(self, config: "BaseConfig", services: "ServiceContainer"):
        self.config = config
        self.services = services

    def register_all(self, app: FastAPI) -> None:
        """Attach all routers to the app via setup_routes."""
        # Auth: login, register, refresh, logout, verify-token
        auth_router = AuthRouter(
            user_service=self.services.users,
            auth_service=self.services.auth,
        )
        auth_router.setup_routes(app)

        # Vectorstore: add-document, list-documents, delete-document, list-indexes, delete-index
        vectorstore_router = VectorstoreRouter()
        vectorstore_router.setup_routes(app)

        # RAG: POST /chat/invoke/{thread_id}
        rag_router = RagRouter()
        rag_router.setup_routes(app)

        # Documents: S3 upload, list, bulk delete
        document_router = DocumentRouter()
        document_router.setup_routes(app)

        # Scalar API docs at /docs (default Swagger disabled in container)
        docs_router = DocsRouter()
        docs_router.setup_routes(app)

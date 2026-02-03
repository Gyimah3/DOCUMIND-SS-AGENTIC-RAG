"""Application factory and dependency injection container."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

if TYPE_CHECKING:
    from app.config import BaseConfig


@dataclass(frozen=True)
class AppConfig:
    version: str
    title: str = "DocuMindSS API"
    description: str = "DocuMindSS – document indexing and vector search API for Seedstars."
    contact: dict[str, str] | None = field(default_factory=lambda: {"name": "DocuMindSS"})
    openapi_url: str = "/openapi.json"
    docs_url: str | None = "/docs"
    redoc_url: str | None = None


class ServiceContainer:
    """Lazy-initialized service container."""

    def __init__(self, config: "BaseConfig"):
        self._config = config
        self._services: dict[str, object] = {}

    def _lazy_init(self, service_name: str, factory: object):
        if service_name not in self._services:
            self._services[service_name] = (
                factory() if callable(factory) else factory
            )
        return self._services[service_name]

    @property
    def security(self):
        from services.security import SecurityService

        return self._lazy_init(
            "security",
            lambda: SecurityService(
                secret_key=self._config.secret_key,
                algorithm=self._config.algorithm,
                access_token_expire_minutes=self._config.access_token_expire_minutes,
                refresh_token_expire_days=self._config.refresh_token_expire_days,
            ),
        )

    @property
    def users(self):
        from services.user_service import UserService

        return self._lazy_init(
            "users",
            lambda: UserService(security_service=self.security),
        )

    @property
    def auth(self):
        from services.auth import AuthService

        return self._lazy_init(
            "auth",
            lambda: AuthService(security_service=self.security),
        )


class ApplicationContainer:
    """Holds config, services, and the FastAPI app instance."""

    def __init__(self, config: "BaseConfig"):
        self.config = config
        self.services = ServiceContainer(config)
        self.metadata = AppConfig(version=config.APP_VERSION, title=config.APP_NAME)
        self.started_at = datetime.now(UTC)
        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title=self.metadata.title,
            description=self.metadata.description,
            version=self.metadata.version,
            contact=self.metadata.contact,
            lifespan=self._lifespan,
            default_response_class=ORJSONResponse,
            openapi_url=self.metadata.openapi_url,
            docs_url=None,  # We serve Scalar at /docs via DocsRouter
            redoc_url=None,
        )
        self._configure(app)
        return app

    def _configure(self, app: FastAPI) -> None:
        self._setup_cors(app)
        self._setup_routes(app)
        self._setup_static_and_pages(app)
        self._setup_error_handlers(app)

    def _setup_cors(self, app: FastAPI) -> None:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self, app: FastAPI) -> None:
        from app.routes import RouterRegistry

        RouterRegistry(self.config, self.services).register_all(app)

    def _setup_static_and_pages(self, app: FastAPI) -> None:
        """Mount frontend static files and serve /, /rag (same pattern as old backend)."""
        root = Path(__file__).resolve().parent.parent
        frontend_dir = root / "frontend"
        documentation_dir = root / "documentation"

        # GET / -> redirect to /rag (or serve documentation if present)
        @app.get("/")
        async def root(_request: Request):
            if documentation_dir.is_dir():
                templates = Jinja2Templates(directory=str(documentation_dir))
                return templates.TemplateResponse("index.html", {"request": _request})
            return RedirectResponse(url="/rag", status_code=302)

        # GET /rag -> Jinja2 chat.html (frontend); static assets at /rag/* (register route first)
        if frontend_dir.is_dir():
            templates = Jinja2Templates(directory=str(frontend_dir))

            @app.get("/rag")
            async def rag_page(request: Request):
                return templates.TemplateResponse(
                    "chat.html",
                    {"request": request, "base_path": "/rag"},
                )

            @app.get("/documents-page")
            async def documents_page(request: Request):
                return templates.TemplateResponse(
                    "documents.html",
                    {"request": request, "base_path": "/rag"},
                )

            # Static files for frontend at /rag (mount after route so GET /rag hits route above)
            app.mount("/rag", StaticFiles(directory=str(frontend_dir)), name="rag")

        # Optional: documentation static files at /static
        if documentation_dir.is_dir():
            app.mount("/static", StaticFiles(directory=str(documentation_dir)), name="static")

    def _setup_error_handlers(self, app: FastAPI) -> None:
        from errors.errors_handler import register_error_handlers

        register_error_handlers(app)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        await self._startup()
        try:
            yield
        finally:
            await self._shutdown()

    async def _startup(self) -> None:
        logger.info("Starting application...")
        logger.success("Application started successfully")

    async def _shutdown(self) -> None:
        logger.info("Shutting down application...")

    @property
    def instance(self) -> FastAPI:
        return self._app

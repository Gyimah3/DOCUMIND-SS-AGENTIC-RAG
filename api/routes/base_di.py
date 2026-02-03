from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, TypeVar
from uuid import UUID

from fastapi import APIRouter

from errors import ForbiddenError, NotFoundError
from api.utils.types import ActionType
from api.decorator import ActionMetadata

if TYPE_CHECKING:
    from fastapi import FastAPI

T = TypeVar("T", bound=Any)


class BaseRouter(ABC):
    """Base router class with dependency injection support.

    This router provides convenient access to common dependencies through
    the dependency injection system, eliminating the need to pass services
    through constructors.
    """

    def __init__(
        self,
        router: APIRouter,
    ):
        self.router = router
        self._register_decorated_routes()

    def _register_decorated_routes(self):
        routes_to_register = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_action_metadata"):
                metadata: ActionMetadata = attr._action_metadata  # type: ignore

                if metadata.url_path is not None:
                    path = f"/{metadata.url_path}"
                    if metadata.detail and "{id}" not in metadata.url_path:
                        path = f"/{{id}}{path}"
                else:
                    if metadata.detail:
                        path = "/{id}"
                    else:
                        path = "/"

                routes_to_register.append((path, attr, metadata))

        def route_sort_key(route_info):
            path, _, _ = route_info
            param_count = path.count("{")
            return (param_count, len(path))

        routes_to_register.sort(key=route_sort_key)

        for path, attr, metadata in routes_to_register:
            self.router.add_api_route(
                path,
                attr,
                methods=[metadata.method],
                response_model=metadata.response_model,
                status_code=metadata.status_code,
                dependencies=metadata.dependencies,  # type: ignore
                summary=metadata.kwargs.pop(
                    "summary", f"{' '.join(attr.__name__.split('_')).title()}"
                ),
                description=metadata.description,
                tags=metadata.tags,
                deprecated=metadata.deprecated,
                **metadata.kwargs,
            )

    def setup_routes(self, app: "FastAPI"):
        app.include_router(self.router)

    def get_router(self) -> APIRouter:
        return self.router

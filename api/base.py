from abc import ABC
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import APIRouter, Depends

from api.decorator import ActionMetadata
from errors import NotFoundError

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
        prefix: str = "",
        tags: list[str | Enum] | None = None,
        dependencies: Sequence[Callable[..., Any]] | None = None,
        **router_kwargs: Any,
    ):
        self.router = APIRouter(
            prefix=prefix,
            tags=tags,
            dependencies=[Depends(dep) for dep in dependencies] if dependencies else None,
            **router_kwargs,
        )
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
                elif metadata.detail:
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

    def check_not_found_error(self, resource: T | None, detail: str = "Resource not found") -> T:
        if resource is None:
            raise NotFoundError(
                detail=detail,
            )
        return resource

    def setup_routes(self, app: "FastAPI"):
        app.include_router(self.router)

    def get_router(self) -> APIRouter:
        return self.router

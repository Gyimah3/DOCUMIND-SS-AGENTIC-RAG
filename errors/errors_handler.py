from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from pydantic import ValidationError

if TYPE_CHECKING:
    from fastapi import FastAPI


def register_error_handlers(app: "FastAPI") -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        return ORJSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return ORJSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(_: Request, exc: ValidationError):
        return ORJSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": exc.errors(
                    include_input=False,
                    include_url=False,
                    include_context=False,
                )
            },
        )

    @app.exception_handler(Exception)
    async def exception_handler(_: Request, exc: Exception):
        return ORJSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred"},
        )

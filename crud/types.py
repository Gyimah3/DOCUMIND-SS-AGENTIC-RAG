from typing import Any, Literal, TypedDict, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase

ModelType = TypeVar("ModelType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


OPERATORS = Literal[
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "like",
    "ilike",
    "not_ilike",
    "contains",
    "startswith",
    "endswith",
    "in",
    "not_in",
    "any",
    "not_any",
    "is_null",
    "is_not_null",
    "is",
    "is_not",
    "between",
    "and",
    "or",
    "not",
]


class FilterSchema(TypedDict):
    field: str | None
    op: OPERATORS
    value: Any

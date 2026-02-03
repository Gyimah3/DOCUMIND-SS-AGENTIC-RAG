from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Type, Union

from sqlalchemy import Column, ColumnElement, and_, not_, or_
from sqlalchemy.sql.elements import BinaryExpression

from errors import FilterError

from .types import OPERATORS, FilterSchema, ModelType

FilterParam = FilterSchema

if TYPE_CHECKING:
    from sqlalchemy.sql import Select


class QueryFilter(Generic[ModelType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

        self._operators: Dict[str, Callable[[Column, Any], ColumnElement[bool]]] = {
            "eq": lambda c, v: c == v,
            "ne": lambda c, v: c != v,
            "gt": lambda c, v: c > v,
            "ge": lambda c, v: c >= v,
            "lt": lambda c, v: c < v,
            "le": lambda c, v: c <= v,
            "==": lambda c, v: c == v,
            "!=": lambda c, v: c != v,
            ">": lambda c, v: c > v,
            ">=": lambda c, v: c >= v,
            "<": lambda c, v: c < v,
            "<=": lambda c, v: c <= v,
            "like": lambda c, v: c.like(v),
            "ilike": lambda c, v: c.ilike(v),
            "not_ilike": lambda c, v: c.not_ilike(v),
            "contains": lambda c, v: c.contains(v),
            "startswith": lambda c, v: c.startswith(v),
            "endswith": lambda c, v: c.endswith(v),
            "in": lambda c, v: c.in_(v),
            "not_in": lambda c, v: ~c.in_(v),
            "any": lambda c, v: c.any(v),
            "not_any": lambda c, v: ~c.any(v),
            "is_null": lambda c, v: c.is_(None),
            "is_not_null": lambda c, v: c.is_not(None),
            "is": lambda c, v: c.is_(v),
            "is_not": lambda c, v: c.is_not(v),
            "between": lambda c, v: c.between(v[0], v[1]),
        }

    def _build_filter_condition(
        self, filter_param: FilterParam
    ) -> Union[ColumnElement[bool], BinaryExpression, None]:
        if filter_param["op"] in ["and", "or", "not"]:
            return self._build_logical_condition(filter_param)

        if filter_param["field"] is None:
            raise FilterError(
                f"Field must be specified for operator '{filter_param['op']}'"
            )

        if "." in filter_param["field"]:
            return self._build_nested_filter_condition(filter_param)

        return self._build_simple_filter_condition(filter_param)

    def _build_simple_filter_condition(
        self, filter_param: FilterParam
    ) -> Union[ColumnElement[bool], BinaryExpression, None]:
        column = getattr(self.model, filter_param["field"])
        return self._apply_operator_to_column(
            column, filter_param["op"], filter_param["value"]
        )

    def _build_nested_filter_condition(
        self, filter_param: FilterParam
    ) -> Union[ColumnElement[bool], BinaryExpression, None]:
        field_path = filter_param["field"].split(".")
        current_model = self.model
        relationship_chain = []

        for segment in field_path[:-1]:
            rel_attr = getattr(current_model, segment)
            relationship_chain.append(rel_attr)
            current_model = rel_attr.property.mapper.class_

        column_name = field_path[-1]
        column_attr = getattr(current_model, column_name)
        condition = self._apply_operator_to_column(
            column_attr, filter_param["op"], filter_param["value"]
        )
        for rel_attr in reversed(relationship_chain):
            rel_property = rel_attr.property
            if rel_property.uselist:
                condition = rel_attr.any(condition)
            else:
                condition = rel_attr.has(condition)

        return condition

    def _apply_operator_to_column(
        self, column: Column, operator: OPERATORS, value: Any
    ) -> Union[ColumnElement[bool], BinaryExpression]:
        operator_func = self._operators.get(operator)
        if operator_func is None:
            raise FilterError(f"Unsupported operator: {operator}")
        return operator_func(column, value)

    def apply_filters(self, query: "Select", filters: List[FilterParam]) -> "Select":
        for filter_param in filters:
            condition = self._build_filter_condition(filter_param)
            if condition is not None:
                query = query.where(condition)
        return query

    def _build_logical_condition(
        self, filter_param: FilterParam
    ) -> Union[ColumnElement[bool], BinaryExpression, None]:
        conditions = [
            self._build_filter_condition(nested_filter)
            for nested_filter in filter_param["value"]
        ]
        conditions = [c for c in conditions if c is not None]

        if not conditions:
            return None

        if filter_param["op"] == "and":
            return and_(*conditions)
        elif filter_param["op"] == "or":
            return or_(*conditions)
        elif filter_param["op"] == "not":
            return not_(conditions[0])

        return None

from typing import Generic, Iterator, List, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: T) -> bool:
        return item in self.items


class SuccessResponse(BaseModel):
    success: bool
    message: str

    model_config = ConfigDict(from_attributes=True)

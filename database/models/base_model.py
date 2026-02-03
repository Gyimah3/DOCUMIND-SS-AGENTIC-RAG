from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    declarative_mixin,
)
from sqlalchemy import DateTime, MetaData
from datetime import datetime
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    metadata = MetaData()

    # @declared_attr.directive
    # def __tablename__(cls) -> str:
    #     return cls.__name__.lower() + "s"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.__dict__})>"


@declarative_mixin
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.__dict__})>"

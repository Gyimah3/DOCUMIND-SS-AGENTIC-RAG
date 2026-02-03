from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey
from sqlalchemy.dialects import postgresql as pg
from typing import TYPE_CHECKING
from uuid import UUID
from .base_model import Base, TimestampMixin


if TYPE_CHECKING:
    from .vectorstore import VectorStore
    from .data_source import DataSource


class VectorIndex(Base):
    __tablename__ = "vector_indexes"

    id: Mapped[UUID] = mapped_column(pg.UUID, primary_key=True, index=True)
    vector_store_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("vectorstores.id"), index=True
    )
    document_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("data_sources.id"), index=True
    )
    datasource: Mapped["DataSource"] = relationship(
        "DataSource", back_populates="vector_indexes"
    )

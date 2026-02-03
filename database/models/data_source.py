from uuid import UUID, uuid4
from typing import TYPE_CHECKING, List
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import relationship
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy.orm import Mapped, mapped_column
from .base_model import Base, TimestampMixin

if TYPE_CHECKING:
    from .user import User
    from .vectorstore import VectorStore
    from .vs_index import VectorIndex


class DataSource(TimestampMixin, Base):
    __tablename__ = "data_sources"

    id: Mapped[UUID] = mapped_column(
        pg.UUID, primary_key=True, index=True, default=uuid4
    )
    user_id: Mapped[UUID] = mapped_column(pg.UUID, ForeignKey("users.id"), index=True)
    vector_store_id: Mapped[UUID] = mapped_column(
        pg.UUID, ForeignKey("vectorstores.id"), index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=True)
    mimetype: Mapped[str] = mapped_column(String, nullable=True)
    s3_key: Mapped[str] = mapped_column(String, nullable=True)
    s3_url: Mapped[str] = mapped_column(String, nullable=True)
    user: Mapped["User"] = relationship("User", back_populates="data_sources")
    vectorstore: Mapped["VectorStore"] = relationship(
        "VectorStore", back_populates="data_sources"
    )
    vector_indexes: Mapped[List["VectorIndex"]] = relationship(
        "VectorIndex", back_populates="datasource", cascade="all, delete-orphan"
    )

    __mapper_args__ = {"eager_defaults": False}

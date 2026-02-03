from uuid import UUID, uuid4
from typing import List, TYPE_CHECKING
from .base_model import Base, TimestampMixin
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects import postgresql as pg

if TYPE_CHECKING:
    from .user import User
    from .data_source import DataSource


class VectorStore(TimestampMixin, Base):
    __tablename__ = "vectorstores"
    id: Mapped[UUID] = mapped_column(
        pg.UUID, primary_key=True, index=True, default=uuid4
    )
    index_name: Mapped[str] = mapped_column(
        String, nullable=False, index=True, unique=True
    )
    user_id: Mapped[UUID] = mapped_column(pg.UUID, ForeignKey("users.id"), index=True)
    embedding: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String, nullable=True)
    user: Mapped["User"] = relationship("User", back_populates="vectorstores")
    data_sources: Mapped[List["DataSource"]] = relationship(
        "DataSource", back_populates="vectorstore", cascade="all, delete-orphan"
    )

    __mapper_args__ = {"eager_defaults": False}

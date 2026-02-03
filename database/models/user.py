import uuid
from typing import Any, List, Literal, TYPE_CHECKING

import bcrypt
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.schema import UniqueConstraint

from .base_model import Base, TimestampMixin
from .data_source import DataSource
from .vectorstore import VectorStore

if TYPE_CHECKING:
    # from .knowledge_base import KnowledgeBase
    pass

# bcrypt has a 72-byte limit
BCRYPT_MAX_PASSWORD_BYTES = 72


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        pg.UUID, primary_key=True, index=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String, nullable=False)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[Literal["user", "admin"]] = mapped_column(
        String, server_default="user", nullable=False
    )
    data_sources: Mapped[List["DataSource"]] = relationship(
        "DataSource", back_populates="user", cascade="all, delete-orphan"
    )
    vectorstores: Mapped[List["VectorStore"]] = relationship(
        "VectorStore", back_populates="user", cascade="all, delete-orphan"
    )
    UniqueConstraint(email, name="unique_email")

    def __init__(self, **kw: Any):
        super().__init__(**kw)
        # Skip hashing if already bcrypt (e.g. when created via UserService with SecurityService hash)
        if getattr(self, "password", None) and not (
            isinstance(self.password, str) and self.password.startswith("$2b$")
        ):
            self.password = self.get_password_hash(self.password)

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def get_password_hash(self, password: str) -> str:
        secret = password.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]
        return bcrypt.hashpw(secret, bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES],
            hashed_password.encode("utf-8"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
        }

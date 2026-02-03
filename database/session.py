import ssl
from typing import AsyncGenerator, Dict, Optional
from uuid import uuid4

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


class DBSessionManager:
    def __init__(
        self,
        db_url: str,
    ):
        self.db_url = db_url


        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        self.engine: AsyncEngine = create_async_engine(
            self.db_url,
            pool_size=20,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=60 * 30,
            pool_pre_ping=True,
            connect_args={
                "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0,
            },
        )

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )

    # async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
    #     async with self.async_session() as session:
    #         tenant_id_value: Optional[str] = None
    #         role_set: bool = False
    #         try:
    #             if self.current_tenant_context is not None:
    #                 tenant_uuid = self.current_tenant_context.get()
    #                 if tenant_uuid is not None:
    #                     tenant_id_value = str(tenant_uuid)
    #                     await session.execute(
    #                         text("SET app.current_tenant = :tenant"),
    #                         {"tenant": tenant_id_value},
    #                     )
    #                     try:
    #                         await session.execute(text("SET ROLE application_role"))
    #                         role_set = True
    #                     except Exception:
    #                         # Role might not exist in some environments; ignore
    #                         pass

    #             yield session
    #         except Exception as e:
    #             await session.rollback()
    #             raise e
    #         finally:
    #             # Reset settings to avoid leaking to pooled connections
    #             try:
    #                 if tenant_id_value is not None:
    #                     await session.execute(text("RESET app.current_tenant"))
    #                 if role_set:
    #                     await session.execute(text("RESET ROLE"))
    #             except Exception:
    #                 pass
    #             await session.close()

    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session without tenant isolation.
        Use this for operations that don't require tenant context (e.g., knowledge base operations).
        """
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()


    async def get_db(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session without tenant isolation.
        Use this for operations that don't require tenant context (e.g., knowledge base operations).
        """
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()

    async def check_health(self) -> Dict[str, str]: 
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()
                return {
                    "status": "healthy" if row and row[0] == 1 else "unhealthy",
                    "message": "Database connection successful"
                    if row and row[0] == 1
                    else "Database connection failed",
                }

        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}",
            }

    async def close(self):
        """Close database connections and dispose engine"""
        await self.engine.dispose()

    async def close_engine(self):
        """Legacy method - use close() instead"""
        await self.close()


def get_session() -> AsyncGenerator[AsyncSession, None]:
    return db_session_manager.get_db()


def get_session_no_tenant() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session without tenant isolation for knowledge base operations."""
    return db_session_manager.get_db_session()


db_session_manager = DBSessionManager(
    db_url=settings.db_url
)

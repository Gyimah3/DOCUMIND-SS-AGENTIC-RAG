"""Initialize database tables if they don't exist."""

import asyncio
import sys
from loguru import logger
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from database.models.base_model import Base
from database.models import User, VectorStore, DataSource, VectorIndex  # noqa: F401 - import to register models


async def init_database():
    """Create all tables if they don't exist."""
    logger.info("Checking database connection...")

    engine = create_async_engine(
        settings.db_url,
        pool_pre_ping=True,
    )

    try:
        # Test connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")

        # Check if tables exist
        async with engine.begin() as conn:
            def check_tables(sync_conn):
                inspector = inspect(sync_conn)
                existing_tables = inspector.get_table_names()
                return existing_tables

            existing_tables = await conn.run_sync(check_tables)
            logger.info(f"Existing tables: {existing_tables}")

            # Get required tables from models
            required_tables = Base.metadata.tables.keys()
            logger.info(f"Required tables: {list(required_tables)}")

            # Check which tables are missing
            missing_tables = [t for t in required_tables if t not in existing_tables]

            if missing_tables:
                logger.info(f"Missing tables: {missing_tables}")
                logger.info("Creating missing tables...")

                # Create all tables (SQLAlchemy will only create missing ones)
                await conn.run_sync(Base.metadata.create_all)

                logger.info("Tables created successfully!")
            else:
                logger.info("All tables already exist. No changes needed.")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        await engine.dispose()

    logger.info("Database initialization complete.")


if __name__ == "__main__":
    asyncio.run(init_database())

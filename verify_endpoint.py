import os
import asyncio
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from database.models import DataSource, VectorStore
from app.config import settings

async def test_endpoint_logic():
    db_url = settings.db_url
    if not db_url:
        print("db_url not found in settings")
        return
        
    # Standardize to asyncpg if needed
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://')
        
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        # Find a vector store
        q = select(VectorStore).limit(1)
        result = await db.execute(q)
        vs = result.scalar_one_or_none()
        
        if not vs:
            print("No VectorStore found")
            return
            
        index_name = vs.index_name
        user_id = vs.user_id
        
        # Test the query logic
        q = select(DataSource).join(VectorStore).where(
            VectorStore.index_name == index_name,
            VectorStore.user_id == user_id
        ).order_by(DataSource.created_at.desc()).limit(1)
        
        result = await db.execute(q)
        doc = result.scalar_one_or_none()
        
        if doc:
            print(f"SUCCESS: Found document URL for index '{index_name}': {doc.s3_url or doc.url}")
        else:
            print(f"FAILURE: No document found for index '{index_name}'")

if __name__ == "__main__":
    asyncio.run(test_endpoint_logic())

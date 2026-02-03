#!/bin/bash
set -e

echo "Starting DocuMindSS..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
while ! python -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import os

async def check():
    engine = create_async_engine(os.environ.get('DB_URL'))
    try:
        async with engine.begin() as conn:
            await conn.execute(text('SELECT 1'))
        return True
    except:
        return False
    finally:
        await engine.dispose()

exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "PostgreSQL is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis..."
while ! python -c "
import redis
import os
r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
r.ping()
" 2>/dev/null; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done
echo "Redis is ready!"

# Initialize database tables
echo "Initializing database tables..."
python -m scripts.init_db

# Start the application
echo "Starting uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8001

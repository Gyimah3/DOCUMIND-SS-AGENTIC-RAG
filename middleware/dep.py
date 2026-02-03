from uuid import UUID

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from errors import (
    BadRequestError,
    UnauthorizedError,
)
from database.models.user import User
from database.session import db_session_manager
# from services.permission import PermissionService
from services.security import security_service
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(db_session_manager.get_db),
) -> User:
    try:
        payload = await security_service.verify_token_async(credentials.credentials)
        user_id = UUID(payload.get("sub"))
        if user_id is None:
            raise UnauthorizedError(detail="Invalid token payload")
    except Exception:
        raise UnauthorizedError(detail="Invalid token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise UnauthorizedError(detail="User not found")


    return user




async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    return current_user

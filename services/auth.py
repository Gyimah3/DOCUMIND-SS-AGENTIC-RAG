from typing import Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.user import User
from schemas.user import LoginRequest, TokenResponse, UserResponse
from services.security import SecurityService


class AuthService:
    def __init__(self, security_service: SecurityService):
        self.security_service = security_service

    async def authenticate_user(
        self, db: AsyncSession, login_data: LoginRequest
    ) -> Optional[User]:
        result = await db.execute(
            select(User).where(User.email == login_data.email)
        )
        user = result.scalar_one_or_none()

        if not user or not self.security_service.verify_password(
            login_data.password, user.password
        ):
            return None

        return user

    async def create_tokens(self, user: User) -> TokenResponse:
        access_token = self.security_service.create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
            }
        )
        refresh_token = self.security_service.create_refresh_token(str(user.id))

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=UserResponse.model_validate(user),
        )

    async def refresh_tokens(
        self, db: AsyncSession, refresh_token: str
    ) -> TokenResponse:
        try:
            payload = await self.security_service.verify_token_async(refresh_token)

            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")

            user_id = payload.get("sub")
            if not user_id:
                raise ValueError("Invalid token payload")

            result = await db.execute(select(User).where(User.id == UUID(user_id)))
            user = result.scalar_one_or_none()

            if not user:
                raise ValueError("User not found")

            return await self.create_tokens(user)

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise ValueError("Invalid or expired refresh token")



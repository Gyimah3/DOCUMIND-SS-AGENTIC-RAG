from typing import TYPE_CHECKING, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.user import User
from database.session import db_session_manager
from middleware.dep import get_current_active_user
from api.routes.base_di import BaseRouter
from api.decorator import action
from schemas.generic import SuccessResponse
from schemas.user import (
    LoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)

if TYPE_CHECKING:
    from services.auth import AuthService
    from services.user_service import UserService


class AuthRouter(BaseRouter):
    def __init__(self, user_service: "UserService", auth_service: "AuthService"):
        router = APIRouter(prefix="/auth", tags=["authentication"])
        super().__init__(router)
        self.user_service = user_service
        self.auth_service = auth_service

    @action(method="POST", url_path="login", response_model=TokenResponse)
    async def login(
        self,
        login_data: LoginRequest,
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> TokenResponse:
        """Authenticate user and return access tokens

        Args:
            login_data: User login credentials
            db: Database session manager
            auth_service: Authentication service

        Returns:
            TokenResponse containing access and refresh tokens

        Raises:
            HTTPException: If credentials are invalid

        """
        user = await self.auth_service.authenticate_user(db, login_data)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await self.auth_service.create_tokens(user)

    @action(
        method="POST",
        url_path="register",
        response_model=UserResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def register(
        self,
        user_data: UserCreate,
        db: AsyncSession = Depends(db_session_manager.get_db),
    ) -> UserResponse:
        """Create a new user (registration).

        Args:
            user_data: User creation payload (email, password, first_name, last_name)
            db: Database session

        Returns:
            Created user (without password)

        Raises:
            HTTPException 400: If email already exists
        """
        user = await self.user_service.create_user(db, user_data)
        return UserResponse.model_validate(user)

    @action(method="POST", url_path="refresh", response_model=TokenResponse)
    async def refresh_token(
        self,
        refresh_request: RefreshTokenRequest,
        db: AsyncSession = Depends(db_session_manager.get_db),
    ):
        """Refresh access token using refresh token

        Args:
            refresh_request: Request containing refresh token
            db: Database session

        Returns:
            TokenResponse with new tokens

        Raises:
            HTTPException: If refresh token is invalid

        """
        try:
            return await self.auth_service.refresh_tokens(
                db, refresh_request.refresh_token
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @action(
        method="POST",
        url_path="logout",
        response_model=SuccessResponse,
        dependencies=[Depends(get_current_active_user)],
    )
    async def logout(
        self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> SuccessResponse:
        """Logout current user and blacklist the token

        This endpoint blacklists the current access token, preventing it from
        being used for further requests. The token will remain blacklisted
        until it expires naturally.

        Returns:
            Success message

        """
        payload = self.auth_service.security_service._decode_jwt_token(
            credentials.credentials, verify=False
        )
        exp = payload.get("exp")

        if exp:
            from datetime import UTC, datetime

            exp_datetime = datetime.fromtimestamp(exp, tz=UTC)
            now = datetime.now(UTC)
            ttl_seconds = max(0, int((exp_datetime - now).total_seconds()))

            await self.auth_service.security_service.blacklist_token(
                credentials.credentials, ttl_seconds
            )

        return SuccessResponse(success=True, message="Logged out successfully")

    @action(method="GET", url_path="verify-token", response_model=Dict)
    async def verify_token(
        self,
        current_user: User = Depends(get_current_active_user),
    ) -> dict:
        """Verify if current token is valid and return user info

        This endpoint can be used by clients to verify if their
        current access token is still valid.

        Returns:
            Token validity status and user info

        """
        return {
            "valid": True,
            "user_id": str(current_user.id),
            "email": current_user.email,
        }

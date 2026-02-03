import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional
from uuid import UUID

import bcrypt
import jwt
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from errors import InternalServerError, UnauthorizedError
from services.redis_vectorstore import redis_service
# from services.redis_service import redis_service

# bcrypt has a 72-byte limit; truncate to avoid ValueError
BCRYPT_MAX_PASSWORD_BYTES = 72


def _password_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]


class SecurityService:
    def __init__(
        self,
        secret_key: SecretStr,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            _password_bytes(plain_password),
            hashed_password.encode("utf-8"),
        )

    def get_password_hash(self, password: str) -> str:
        return bcrypt.hashpw(
            _password_bytes(password),
            bcrypt.gensalt(),
        ).decode("utf-8")

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        now = datetime.now(UTC)
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire, "iat": now})
        return jwt.encode(
            to_encode, self.secret_key.get_secret_value(), algorithm=self.algorithm
        )

    def create_refresh_token(self, user_id: str) -> str:
        now = datetime.now(UTC)
        expire = now + timedelta(days=self.refresh_token_expire_days)
        to_encode = {"sub": str(user_id), "exp": expire, "type": "refresh", "iat": now}
        return jwt.encode(
            to_encode, self.secret_key.get_secret_value(), algorithm=self.algorithm
        )

    def _get_token_hash(self, token: str) -> str:
        """Generate SHA256 hash of token for Redis key"""
        return hashlib.sha256(token.encode()).hexdigest()

    def _decode_jwt_token(self, token: str, verify: bool = True) -> Dict[str, Any]:
        """Decode JWT token and convert errors to UnauthorizedError.

        Args:
            token: JWT token to decode
            verify: Whether to verify the token signature

        Returns:
            Decoded token payload

        Raises:
            UnauthorizedError: If token is invalid

        """
        try:
            if verify:
                payload = jwt.decode(
                    token,
                    self.secret_key.get_secret_value(),
                    algorithms=[self.algorithm],
                )
            else:
                payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except (jwt.InvalidTokenError, jwt.ExpiredSignatureError, jwt.DecodeError):
            raise UnauthorizedError("Invalid or expired token")

    async def blacklist_token(self, token: str, expires_in_seconds: int) -> bool:
        """Add a token to the blacklist with TTL matching token expiration.

        Args:
            token: JWT token to blacklist
            expires_in_seconds: TTL in seconds (should match token expiration)

        Returns:
            True if token was blacklisted successfully

        Raises:
            InternalServerError: If Redis operation fails

        """
        token_hash = self._get_token_hash(token)
        key = f"blacklist:token:{token_hash}"
        value = datetime.now(UTC).isoformat()
        success = await redis_service.set(key, value, ex=expires_in_seconds)
        if not success:
            raise InternalServerError("Failed to blacklist token")
        return success

    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted.

        Args:
            token: JWT token to check

        Returns:
            True if token is blacklisted, False otherwise

        Raises:
            InternalServerError: If Redis operation fails

        """
        token_hash = self._get_token_hash(token)
        key = f"blacklist:token:{token_hash}"
        exists = await redis_service.exists(key)
        return exists > 0

    async def blacklist_user_tokens(self, user_id: UUID) -> bool:
        """Blacklist all tokens for a user by storing a timestamp marker.
        This is used when password changes to invalidate all existing tokens.

        Args:
            user_id: User ID whose tokens should be blacklisted

        Returns:
            True if user tokens were blacklisted successfully

        Raises:
            InternalServerError: If Redis operation fails

        """
        key = f"blacklist:user:{user_id}"
        timestamp = datetime.now(UTC).isoformat()
        max_ttl = self.refresh_token_expire_days * 24 * 60 * 60
        success = await redis_service.set(key, timestamp, ex=max_ttl)
        if not success:
            raise InternalServerError("Failed to blacklist user tokens")
        return success

    async def _is_user_token_invalidated(
        self, user_id: UUID, token_issued_at: Optional[float] = None
    ) -> bool:
        """Check if user's tokens have been invalidated (e.g., after password change).

        Args:
            user_id: User ID to check
            token_issued_at: Optional timestamp when token was issued (from 'iat' claim)

        Returns:
            True if user tokens were invalidated, False otherwise

        Raises:
            InternalServerError: If Redis operation fails

        """
        key = f"blacklist:user:{user_id}"
        invalidation_timestamp = await redis_service.get(key)
        if invalidation_timestamp is None:
            return False

        if token_issued_at:
            invalidation_str = invalidation_timestamp.decode("utf-8")
            invalidation_dt = datetime.fromisoformat(
                invalidation_str.replace("Z", "+00:00")
            )
            token_issued_dt = datetime.fromtimestamp(token_issued_at, tz=UTC)
            return token_issued_dt < invalidation_dt

        return True

    async def verify_token_async(self, token: str) -> Dict[str, Any]:
        """Verify JWT token asynchronously with blacklist checking.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload

        Raises:
            UnauthorizedError: If token is invalid or blacklisted
            InternalServerError: If Redis operation fails

        """
        is_blacklisted = await self.is_token_blacklisted(token)
        if is_blacklisted:
            raise UnauthorizedError("Token has been revoked")

        payload = self._decode_jwt_token(token, verify=True)

        token_user_id = payload.get("sub")
        if token_user_id:
            user_uuid = UUID(token_user_id)
            token_issued_at = payload.get("iat")
            is_invalidated = await self._is_user_token_invalidated(
                user_uuid, token_issued_at
            )
            if is_invalidated:
                raise UnauthorizedError("Token has been revoked")

        return payload

    async def get_current_user_from_token(self, db: AsyncSession, token: str):
        from database.models.user import User

        payload = await self.verify_token_async(token)
        user_id: str | None = payload.get("sub")

        if user_id is None:
            raise UnauthorizedError("Invalid token payload")

        user_uuid = UUID(user_id)

        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalar_one_or_none()

        if user is None:
            raise UnauthorizedError("User not found")

        return user


security_service = SecurityService(
    secret_key=settings.secret_key,
    algorithm=settings.algorithm,
    access_token_expire_minutes=settings.access_token_expire_minutes,
    refresh_token_expire_days=settings.refresh_token_expire_days,
)

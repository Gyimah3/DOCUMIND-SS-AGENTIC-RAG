
from typing import List, Optional
from uuid import UUID
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from api.utils.types import ActionType
from crud.base import BaseCRUD
from crud.types import FilterSchema
from errors import BadRequestError, NotFoundError
from schemas.generic import PaginatedResponse
from schemas.user import UserCreate, UserUpdate
from services.security import SecurityService
from database.models.user import User



class UserCRUD(BaseCRUD[User, UserCreate, UserUpdate]):
    """CRUD operations for User model"""

    def __init__(self):
        super().__init__(User)

    async def get_by_email(
        self, session: AsyncSession, email: str, prefetch: Optional[List[str]] = None
    ) -> Optional[User]:
        """Get user by email address"""
        query = select(User).where(User.email == email)
        if prefetch:
            query = self._apply_prefetch(query, prefetch)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def get_active_users(
        self,
        session: AsyncSession,
        prefetch: Optional[List[str]] = None,
        page: Optional[int] = None,
        size: Optional[int] = None,
    ) -> List[User] | PaginatedResponse[User]:
        """Get active users"""
        filters = [FilterSchema(field="is_active", op="==", value=True)]

        return await self.list_objects(
            session=session,
            filters=filters,
            prefetch=prefetch,
            page=page,
            size=size,
        )

    async def list_users(
        self,
        session: AsyncSession,
        search: Optional[str] = None,
        search_fields: Optional[List[str]] = None,
        sorting: Optional[str] = None,
        prefetch: Optional[List[str]] = None,
        page: Optional[int] = None,
        size: Optional[int] = None,
    ) -> PaginatedResponse[User]:
        if not search_fields:
            search_fields = [
                "first_name",
                "last_name",
                "middle_name",
                "email",
                "address",
            ]

        filters = []
        filters.append(FilterSchema(field="is_active", op="==", value=True))

        return await self.list_objects(  # type: ignore
            session=session,
            search=search,
            search_fields=search_fields,
            filters=filters,
            sorting=sorting,
            prefetch=prefetch,
            page=page,
            size=size,
        )

    async def update_password(
        self,
        session: AsyncSession,
        user_id: UUID,
        new_password_hash: str,
        commit: bool = True,
    ) -> bool:
        """Update user password"""
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(password=new_password_hash, updated_at=func.now())
        )
        result = await session.execute(stmt)
        if commit:
            await session.commit()
        return result.rowcount > 0




class UserService:
    def __init__(
        self, security_service: SecurityService
    ):
        self.security_service = security_service
        self.user_crud = UserCRUD()

    async def create_user(
        self,
        session: AsyncSession,
        user_data: UserCreate,
    ) -> User:
        """Create a new user with validation and audit logging.

        Args:
            session: Database session
            user_data: User creation data
            created_by: User ID creating this user
            audit_ip: IP address for audit log
            audit_user_agent: User agent for audit log

        Returns:
            Created User instance

        """
        existing_user = await self.user_crud.exists(
            session, FilterSchema(field="email", op="==", value=user_data.email)
        )
        if existing_user:
            raise BadRequestError(
                detail="User with this email already exists",
            )

        hashed_password = self.security_service.get_password_hash(user_data.password)

        user_dict = user_data.model_dump(exclude={"password"})
        user_dict["password"] = hashed_password  # User model column is "password"; __init__ skips re-hash if already bcrypt

        user = await self.user_crud.create(
            session=session, obj_in=user_dict, commit=False
        )

        await session.commit()
        await session.refresh(user)

        return user

    async def update_user(
        self,
        session: AsyncSession,
        user_id: UUID,
        update_data: UserUpdate,
    ) -> User:
        """Update user information with validation and audit logging"""
        user = await self.user_crud.get_by_id(session, user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        if update_data.email and update_data.email != user.email:
            existing = await self.user_crud.get_by_email(session, update_data.email)
            if existing:
                raise BadRequestError(
                    detail="Email already in use",
                )

        updated_user = await self.user_crud.update(
            session=session, obj=user, obj_in=update_data, commit=False
        )

        await session.commit()
        logger.info(f"User updated: {user_id}")
        return updated_user

    async def delete_user(
        self,
        session: AsyncSession,
        user_id: UUID,
    ) -> bool:
        """Soft delete user (deactivate) with audit logging"""
        user = await self.user_crud.get_by_id(session, user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        success = await self.user_crud.delete(session=session, obj=user, commit=False)

        if success:

            await session.commit()
            logger.info(f"User deleted: {user_id}")

        return success

    async def change_password(
        self,
        session: AsyncSession,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Change user password with validation"""
        user = await self.user_crud.get_by_id(session, user_id)
        if not user:
            raise NotFoundError(detail="User not found")

        if not self.security_service.verify_password(
            current_password, user.hashed_password
        ):
            raise BadRequestError(
                detail="Current password is incorrect",
            )

        new_hashed_password = self.security_service.get_password_hash(new_password)

        success = await self.user_crud.update_password(
            session=session,
            user_id=user_id,
            new_password_hash=new_hashed_password,
            commit=False,
        )

        if success:

            await session.commit()

            await self.security_service.blacklist_user_tokens(user_id)

        return success

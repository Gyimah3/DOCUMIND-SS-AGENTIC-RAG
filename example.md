import asyncio
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.errors import ForbiddenError, NotFoundError
from crud.types import FilterSchema
from database.models.campaign import Campaign, CampaignTemplate, CitizenSegment
from database.models.user import User
from database.session import db_session_manager
from database.utils.types import (
    ActionType,
    CampaignStatus,
    ChannelType,
    DeliveryStatus,
)
from middleware.dep import (
    get_current_active_user,
    requires_user_permission,
)
from routes.base_di import BaseRouter
from routes.decorator import action
from schemas.campaign import (
    BulkSegmentMemberResponse,
    CampaignAnalyticsResponse,
    CampaignCreate,
    CampaignDeliveryResponse,
    CampaignDetailResponse,
    CampaignExecuteResponse,
    CampaignFilterParams,
    CampaignLifecycleAction,
    CampaignResponse,
    CampaignScheduleCreate,
    CampaignScheduleResponse,
    CampaignScheduleUpdate,
    CampaignStatistics,
    CampaignTemplateCreate,
    CampaignTemplateResponse,
    CampaignTemplateUpdate,
    CampaignUpdate,
    CitizenSegmentResponse,
    DynamicSegmentPreviewRequest,
    OrganizationCampaignSummary,
    SegmentCampaignAssignmentResponse,
    SegmentCreate,
    SegmentFilterParams,
    SegmentMembershipRemove,
    SegmentMembershipResponse,
    SegmentMembershipUpdate,
    SegmentUpdate,
    TemplateRenderRequest,
    TemplateRenderResponse,
)
from schemas.citizen import CitizenResponse
from schemas.document import DocumentFilter, DocumentResponse, DocumentUploadRequest
from schemas.generic import PaginatedResponse

if TYPE_CHECKING:
    from services.campaign import CampaignService
    from services.document import DocumentService
    from services.permission import PermissionService


class CampaignRouter(BaseRouter):
    def __init__(
        self,
        service: "CampaignService",
        permission_service: "PermissionService",
        document_service: "DocumentService",
    ):
        router = APIRouter(prefix="/campaigns", tags=["campaigns"])
        super().__init__(router, permission_service)
        self.permission_service = permission_service
        self.service = service
        self.document_service = document_service

    @action(
        method="POST",
        detail=False,
        summary="Create campaign",
        response_model=CampaignResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("Campaign", ActionType.CREATE)],
    )
    async def create_campaign(
        self,
        name: Annotated[str, Form(..., min_length=1, max_length=255)],
        channel_type: Annotated[ChannelType, Form(...)],
        status: Annotated[CampaignStatus, Form(...)] = CampaignStatus.DRAFT,
        description: Annotated[Optional[str], Form()] = None,
        message_subject: Annotated[Optional[str], Form(max_length=255)] = None,
        template_id: Annotated[Optional[UUID], Form()] = None,
        message_body: Annotated[Optional[str], Form()] = None,
        branch_id: Annotated[Optional[UUID], Form()] = None,
        files: Annotated[Optional[List[UploadFile]], File(...)] = None,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        if template_id:
            await self._check_template_permission(
                db_session=db_session,
                user=current_user,
                template_id=template_id,
                action=ActionType.READ,
            )

        campaign_data = CampaignCreate(
            name=name,
            description=description,
            message_subject=message_subject,
            channel_type=channel_type,
            template_id=template_id,
            status=status,
            message_body=message_body,
        )
        campaign = await self.service.create_campaign(
            session=db_session,
            campaign_data=campaign_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=branch_id,
            commit=False,
        )
        if files:
            upload_request = DocumentUploadRequest(
                entity_id=campaign.id,
                entity_type="CAMPAIGN",
                is_public=False,
                attributes={"campaign_id": str(campaign.id)},
            )
            _ = await asyncio.gather(
                *[
                    self.document_service.upload_document(
                        db_session=db_session,
                        file=file,
                        upload_request=upload_request,
                        user=current_user,
                        branch_id=branch_id,
                        commit=False,
                    )
                    for file in files
                ]
            )

        await db_session.commit()
        await db_session.refresh(campaign)

        return campaign

    @action(
        method="GET",
        detail=False,
        summary="List campaigns",
        response_model=PaginatedResponse[CampaignResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_campaigns(
        self,
        filters: CampaignFilterParams = Depends(),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.list_campaigns(
            session=db_session,
            organization_id=current_user.organization_id,
            filters_params=filters,
        )

    @action(
        method="GET",
        detail=True,
        summary="Get campaign details",
        response_model=CampaignDetailResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_campaign(
        self,
        id: UUID = Path(..., description="Campaign ID"),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        campaign = await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
            prefetch=["template", "schedule"],
        )
        return campaign

    @action(
        method="PATCH",
        detail=True,
        summary="Update campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def update_campaign(
        self,
        id: UUID,
        updates: CampaignUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_campaign(
            session=db_session,
            campaign_id=id,
            campaign_data=updates,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="DELETE",
        detail=True,
        summary="Delete campaign",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("Campaign", ActionType.DELETE)],
    )
    async def delete_campaign(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_campaign(
            session=db_session,
            campaign_id=id,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="activate",
        summary="Activate campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def activate_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.activate_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="pause",
        summary="Pause campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def pause_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        campaign = await self.service.pause_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )
        return CampaignResponse.model_validate(campaign)

    @action(
        method="POST",
        detail=True,
        url_path="cancel",
        summary="Cancel campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def cancel_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        campaign = await self.service.cancel_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )
        return CampaignResponse.model_validate(campaign)

    @action(
        method="POST",
        detail=True,
        url_path="execute",
        summary="Execute campaign immediately",
        response_model=CampaignExecuteResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def execute_campaign_immediately(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        """Execute a campaign immediately.

        Validates:
        - Campaign must be approved (approval_status == APPROVED)
        - Campaign status must not be CANCELLED, ARCHIVED, or COMPLETED
        - Campaign must have targets assigned

        Then creates deliveries and sends them immediately.
        """
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.execute_campaign_immediately(
            session=db_session,
            campaign_id=id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="schedule",
        summary="Create or replace schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.UPDATE),
            requires_user_permission("CampaignSchedule", ActionType.CREATE),
        ],
    )
    async def create_schedule(
        self,
        id: UUID,
        schedule_data: CampaignScheduleCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.create_schedule(
            session=db_session,
            campaign_id=id,
            organization_id=current_user.organization_id,
            schedule_data=schedule_data,
        )

    @action(
        method="PATCH",
        detail=True,
        url_path="schedule",
        summary="Update schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.UPDATE),
            requires_user_permission("CampaignSchedule", ActionType.UPDATE),
        ],
    )
    async def update_schedule(
        self,
        id: UUID,
        schedule_data: CampaignScheduleUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        _, schedule = await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self.service.get_schedule(db_session, id),
        )
        return await self.service.update_schedule(
            session=db_session,
            schedule_id=schedule.id,
            schedule_data=schedule_data,
        )

    @action(
        method="GET",
        detail=True,
        url_path="schedule",
        summary="Get schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.READ),
            requires_user_permission("CampaignSchedule", ActionType.READ),
        ],
    )
    async def get_schedule(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_schedule(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="segments",
        summary="Create segment",
        response_model=CitizenSegmentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.CREATE)],
    )
    async def create_segment(
        self,
        segment_data: SegmentCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.create_segment(
            session=db_session,
            segment_data=segment_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=current_user.branch_id,
        )

    @action(
        method="GET",
        detail=False,
        url_path="segments",
        summary="List segments",
        response_model=PaginatedResponse[CitizenSegmentResponse],
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def list_segments(
        self,
        params: SegmentFilterParams = Depends(),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.list_segments(
            session=db_session,
            organization_id=current_user.organization_id,
            params=params,
        )

    @action(
        method="GET",
        url_path="segments/{id}",
        summary="Get segment",
        response_model=CitizenSegmentResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def get_segment(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.READ,
        )

    @action(
        method="PATCH",
        url_path="segments/{id}",
        summary="Update segment",
        response_model=CitizenSegmentResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def update_segment(
        self,
        id: UUID,
        segment_data: SegmentUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        segment = await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_segment(db_session, segment, segment_data)

    @action(
        method="DELETE",
        url_path="segments/{id}",
        summary="Delete segment",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.DELETE)],
    )
    async def delete_segment(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_segment(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="segments/preview",
        summary="Preview dynamic segment",
        response_model=PaginatedResponse[CitizenResponse],
        dependencies=[
            requires_user_permission("CitizenSegment", ActionType.READ),
            Depends(get_current_active_user),
        ],
    )
    async def preview_dynamic_segment(
        self,
        preview_data: DynamicSegmentPreviewRequest,
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.preview_dynamic_segment(
            session=db_session,
            filter_criteria=dict(preview_data.filter_criteria),
            page=preview_data.page,
            size=preview_data.size,
        )

    @action(
        method="GET",
        # detail=True,
        url_path="segments/{id}/members",
        summary="List segment members",
        response_model=PaginatedResponse[SegmentMembershipResponse],
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def list_segment_members(
        self,
        id: UUID,
        page: int = Query(1, ge=1),
        size: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_segment_members(
            session=db_session,
            segment_id=id,
            page=page,
            size=size,
        )

    @action(
        method="POST",
        detail=True,
        url_path="segments/{segment_id}/members",
        summary="Add members to segment",
        response_model=BulkSegmentMemberResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def add_segment_members(
        self,
        id: UUID,
        segment_id: UUID,
        citizen_ids: List[UUID] = Body(..., embed=True),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.UPDATE,
            ),
        )
        return await self.service.add_citizens_to_segment(
            session=db_session,
            segment_id=segment_id,
            citizen_ids=citizen_ids,
            user_id=current_user.id,
            campaign_id=id,
        )

    @action(
        method="DELETE",
        detail=True,
        url_path="segments/{segment_id}/members",
        summary="Remove members from segment",
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def remove_segment_members(
        self,
        id: UUID,
        segment_id: UUID,
        membership_data: SegmentMembershipRemove,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.UPDATE,
            ),
        )
        await self.service.remove_citizens_from_segment(
            session=db_session,
            segment_id=segment_id,
            citizen_ids=membership_data.citizen_ids,
            campaign_id=id,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @action(
        method="PATCH",
        url_path="segments/members/{membership_id}",
        summary="Update segment membership",
        response_model=SegmentMembershipResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def update_segment_membership(
        self,
        membership_id: UUID,
        membership_data: SegmentMembershipUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        membership = await self.service.membership_crud.get_by_id(
            session=db_session,
            id=membership_id,
        )

        if not membership:
            raise NotFoundError(detail=f"Membership {membership_id} not found")

        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=membership.segment_id,
            action=ActionType.UPDATE,
        )

        return await self.service.update_segment_membership(
            session=db_session,
            membership_id=membership_id,
            subscription_status=membership_data.subscription_status,
            is_blacklisted=membership_data.is_blacklisted,
        )

    @action(
        method="POST",
        detail=True,
        url_path="segments/{segment_id}/campaign",
        summary="Assign campaign to all segment members",
        response_model=SegmentCampaignAssignmentResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def assign_campaign_to_segment(
        self,
        id: UUID,
        segment_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.READ,
            ),
        )
        return await self.service.assign_campaign_to_segment(
            session=db_session,
            segment_id=segment_id,
            campaign_id=id,
        )

    @action(
        method="GET",
        detail=True,
        url_path="deliveries",
        summary="List campaign deliveries",
        response_model=PaginatedResponse[CampaignDeliveryResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_deliveries(
        self,
        id: UUID,
        status_filter: Optional[DeliveryStatus] = Query(None),
        page: int = Query(1, ge=1),
        size: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_campaign_deliveries(
            session=db_session,
            campaign_id=id,
            status=status_filter,
            page=page,
            size=size,
        )

    @action(
        method="GET",
        detail=True,
        url_path="stats",
        summary="Get campaign statistics",
        response_model=CampaignStatistics,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_statistics(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignStatistics:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        stats = await self.service.get_campaign_statistics(db_session, id)
        return CampaignStatistics(
            campaign_id=id,
            total_targets=stats["total_targets"],
            total_deliveries=stats["total_deliveries"],
            delivery_stats=stats["delivery_stats"],
            success_rate=stats["success_rate"],
            failure_rate=stats["failure_rate"],
        )

    @action(
        method="GET",
        detail=True,
        url_path="analytics",
        summary="Get campaign analytics",
        response_model=CampaignAnalyticsResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_analytics(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignAnalyticsResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        analytics = await self.service.get_campaign_analytics(db_session, id)
        return CampaignAnalyticsResponse(**analytics)

    @action(
        method="GET",
        detail=False,
        url_path="summary",
        summary="Get organization campaign summary",
        response_model=OrganizationCampaignSummary,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_org_summary(
        self,
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> OrganizationCampaignSummary:
        summary = await self.service.get_organization_campaign_summary(
            session=db_session,
            organization_id=current_user.organization_id,
        )
        return OrganizationCampaignSummary.model_validate(
            {
                "organization_id": current_user.organization_id,
                **summary,
            }
        )

    @action(
        method="POST",
        detail=False,
        url_path="templates",
        summary="Create template",
        response_model=CampaignTemplateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.CREATE)],
    )
    async def create_template(
        self,
        template_data: CampaignTemplateCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.create_template(
            session=db_session,
            template_data=template_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=current_user.branch_id,
        )

    @action(
        method="GET",
        detail=False,
        url_path="templates",
        summary="List templates",
        response_model=PaginatedResponse[CampaignTemplateResponse],
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def list_templates(
        self,
        channel_type: Optional[str] = Query(None),
        is_active: bool = Query(True),
        page: Optional[int] = Query(1, ge=1),
        size: Optional[int] = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        from database.utils.types import ChannelType

        channel_enum = ChannelType(channel_type) if channel_type else None

        return await self.service.list_templates(
            session=db_session,
            organization_id=current_user.organization_id,
            channel_type=channel_enum,
            is_active=is_active,
            page=page,
            size=size,
        )

    @action(
        method="GET",
        url_path="templates/{id}",
        summary="Get template",
        response_model=CampaignTemplateResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def get_template(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        template = await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.READ,
        )
        return template

    @action(
        method="PATCH",
        detail=False,
        url_path="templates/{id}",
        summary="Update template",
        response_model=CampaignTemplateResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.UPDATE)],
    )
    async def update_template(
        self,
        id: UUID,
        template_data: CampaignTemplateUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_template(
            session=db_session,
            template_id=id,
            template_data=template_data,
        )

    @action(
        method="DELETE",
        url_path="templates/{id}",
        summary="Delete template",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.DELETE)],
    )
    async def delete_template(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_template(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="templates/{id}/render",
        summary="Render template",
        response_model=TemplateRenderResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def render_template(
        self,
        id: UUID,
        render_request: TemplateRenderRequest,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        template = await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.READ,
        )
        return await self.service.render_template(
            template=template,
            placeholder_values=render_request.placeholder_values,
        )

    @action(
        method="GET",
        detail=True,
        url_path="attachments",
        summary="List campaign attachments",
        response_model=List[DocumentResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_attachments(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[DocumentResponse]:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )

        filters = DocumentFilter(
            entity_id=id,
            entity_type="CAMPAIGN",
            min_file_size=None,
            max_file_size=None,
        )
        documents = await self.document_service.list_documents(
            db_session=db_session, filters=filters, page=None, size=None
        )
        return await self.document_service.to_list_response_models(documents)  # type: ignore

    @action(
        method="DELETE",
        detail=True,
        url_path="attachments/{attachment_id}",
        summary="Delete attachment",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def delete_attachment(
        self,
        id: UUID,
        attachment_id: UUID = Path(..., description="Attachment document ID"),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        document = await self.document_service.crud.get_one(
            session=db_session,
            filters=[
                FilterSchema(field="id", op="==", value=attachment_id),
                FilterSchema(field="entity_id", op="==", value=id),
                FilterSchema(field="entity_type", op="==", value="CAMPAIGN"),
            ],
            prefetch=["campaign"],
        )
        if not document:
            raise NotFoundError(detail="Document not found")
        if document.campaign:
            if document.campaign.id != id:
                raise ForbiddenError(detail="Document not attached to this campaign")
            await self._check_resource_permission(
                db_session=db_session,
                user=current_user,
                resource=document.campaign,
                resource_id=document.campaign.id,
                resource_type="Campaign",
                action=ActionType.UPDATE,
            )
        else:
            raise ForbiddenError(detail="Document not attached to any campaign")
        await self.document_service.delete_document(
            db_session=db_session,
            document=document,
            user=current_user,
        )

    async def _check_campaign_permission(
        self,
        db_session: AsyncSession,
        user: User,
        campaign_id: UUID,
        action: ActionType,
        prefetch: Optional[List[str]] = None,
    ) -> Campaign:
        """Check permission and return campaign if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        Reuses the fetched campaign to avoid duplicate DB calls.
        Includes ABAC check for organization isolation.
        """
        campaign = await self.service.get_campaign(
            session=db_session,
            campaign_id=campaign_id,
            prefetch=prefetch,
        )

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=campaign,
            resource_id=campaign_id,
            resource_type="Campaign",
            action=action,
        )

    async def _check_segment_permission(
        self,
        db_session: AsyncSession,
        user: User,
        segment_id: UUID,
        action: ActionType,
    ) -> CitizenSegment:
        """Check permission and return segment if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        """
        segment = await self.service.get_segment(db_session, segment_id)

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=segment,
            resource_id=segment_id,
            resource_type="CitizenSegment",
            action=action,
        )

    async def _check_template_permission(
        self,
        db_session: AsyncSession,
        user: User,
        template_id: UUID,
        action: ActionType,
    ) -> CampaignTemplate:
        """Check permission and return template if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        """
        template = await self.service.get_template(db_session, template_id)
        if not template:
            raise NotFoundError(detail="Template not found")

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=template,
            resource_id=template_id,
            resource_type="CampaignTemplate",
            action=action,
        )



import asyncio
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Path,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from app.errors import ForbiddenError, NotFoundError
from crud.types import FilterSchema
from database.models.campaign import Campaign, CampaignTemplate, CitizenSegment
from database.models.user import User
from database.session import db_session_manager
from database.utils.types import (
    ActionType,
    CampaignStatus,
    ChannelType,
    DeliveryStatus,
)
from middleware.dep import (
    get_current_active_user,
    requires_user_permission,
)
from routes.base_di import BaseRouter
from routes.decorator import action
from schemas.campaign import (
    BulkSegmentMemberResponse,
    CampaignAnalyticsResponse,
    CampaignCreate,
    CampaignDeliveryResponse,
    CampaignDetailResponse,
    CampaignExecuteResponse,
    CampaignFilterParams,
    CampaignLifecycleAction,
    CampaignResponse,
    CampaignScheduleCreate,
    CampaignScheduleResponse,
    CampaignScheduleUpdate,
    CampaignStatistics,
    CampaignTemplateCreate,
    CampaignTemplateResponse,
    CampaignTemplateUpdate,
    CampaignUpdate,
    CitizenSegmentResponse,
    DynamicSegmentPreviewRequest,
    OrganizationCampaignSummary,
    SegmentCampaignAssignmentResponse,
    SegmentCreate,
    SegmentFilterParams,
    SegmentMembershipRemove,
    SegmentMembershipResponse,
    SegmentMembershipUpdate,
    SegmentUpdate,
    TemplateRenderRequest,
    TemplateRenderResponse,
)
from schemas.citizen import CitizenResponse
from schemas.document import DocumentFilter, DocumentResponse, DocumentUploadRequest
from schemas.generic import PaginatedResponse

if TYPE_CHECKING:
    from services.campaign import CampaignService
    from services.document import DocumentService
    from services.permission import PermissionService


class CampaignRouter(BaseRouter):
    def __init__(
        self,
        service: "CampaignService",
        permission_service: "PermissionService",
        document_service: "DocumentService",
    ):
        router = APIRouter(prefix="/campaigns", tags=["campaigns"])
        super().__init__(router, permission_service)
        self.permission_service = permission_service
        self.service = service
        self.document_service = document_service

    @action(
        method="POST",
        detail=False,
        summary="Create campaign",
        response_model=CampaignResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("Campaign", ActionType.CREATE)],
    )
    async def create_campaign(
        self,
        name: Annotated[str, Form(..., min_length=1, max_length=255)],
        channel_type: Annotated[ChannelType, Form(...)],
        status: Annotated[CampaignStatus, Form(...)] = CampaignStatus.DRAFT,
        description: Annotated[Optional[str], Form()] = None,
        message_subject: Annotated[Optional[str], Form(max_length=255)] = None,
        template_id: Annotated[Optional[UUID], Form()] = None,
        message_body: Annotated[Optional[str], Form()] = None,
        branch_id: Annotated[Optional[UUID], Form()] = None,
        files: Annotated[Optional[List[UploadFile]], File(...)] = None,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        if template_id:
            await self._check_template_permission(
                db_session=db_session,
                user=current_user,
                template_id=template_id,
                action=ActionType.READ,
            )

        campaign_data = CampaignCreate(
            name=name,
            description=description,
            message_subject=message_subject,
            channel_type=channel_type,
            template_id=template_id,
            status=status,
            message_body=message_body,
        )
        campaign = await self.service.create_campaign(
            session=db_session,
            campaign_data=campaign_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=branch_id,
            commit=False,
        )
        if files:
            upload_request = DocumentUploadRequest(
                entity_id=campaign.id,
                entity_type="CAMPAIGN",
                is_public=False,
                attributes={"campaign_id": str(campaign.id)},
            )
            _ = await asyncio.gather(
                *[
                    self.document_service.upload_document(
                        db_session=db_session,
                        file=file,
                        upload_request=upload_request,
                        user=current_user,
                        branch_id=branch_id,
                        commit=False,
                    )
                    for file in files
                ]
            )

        await db_session.commit()
        await db_session.refresh(campaign)

        return campaign

    @action(
        method="GET",
        detail=False,
        summary="List campaigns",
        response_model=PaginatedResponse[CampaignResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_campaigns(
        self,
        filters: CampaignFilterParams = Depends(),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.list_campaigns(
            session=db_session,
            organization_id=current_user.organization_id,
            filters_params=filters,
        )

    @action(
        method="GET",
        detail=True,
        summary="Get campaign details",
        response_model=CampaignDetailResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_campaign(
        self,
        id: UUID = Path(..., description="Campaign ID"),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        campaign = await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
            prefetch=["template", "schedule"],
        )
        return campaign

    @action(
        method="PATCH",
        detail=True,
        summary="Update campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def update_campaign(
        self,
        id: UUID,
        updates: CampaignUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_campaign(
            session=db_session,
            campaign_id=id,
            campaign_data=updates,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="DELETE",
        detail=True,
        summary="Delete campaign",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("Campaign", ActionType.DELETE)],
    )
    async def delete_campaign(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_campaign(
            session=db_session,
            campaign_id=id,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="activate",
        summary="Activate campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def activate_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.activate_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="pause",
        summary="Pause campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def pause_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        campaign = await self.service.pause_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )
        return CampaignResponse.model_validate(campaign)

    @action(
        method="POST",
        detail=True,
        url_path="cancel",
        summary="Cancel campaign",
        response_model=CampaignResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def cancel_campaign(
        self,
        id: UUID,
        payload: CampaignLifecycleAction = Body(
            default=CampaignLifecycleAction(reason=None)
        ),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        campaign = await self.service.cancel_campaign(
            session=db_session,
            campaign_id=id,
            lifecycle_data=payload,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
        )
        return CampaignResponse.model_validate(campaign)

    @action(
        method="POST",
        detail=True,
        url_path="execute",
        summary="Execute campaign immediately",
        response_model=CampaignExecuteResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def execute_campaign_immediately(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        """Execute a campaign immediately.

        Validates:
        - Campaign must be approved (approval_status == APPROVED)
        - Campaign status must not be CANCELLED, ARCHIVED, or COMPLETED
        - Campaign must have targets assigned

        Then creates deliveries and sends them immediately.
        """
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.execute_campaign_immediately(
            session=db_session,
            campaign_id=id,
        )

    @action(
        method="POST",
        detail=True,
        url_path="schedule",
        summary="Create or replace schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.UPDATE),
            requires_user_permission("CampaignSchedule", ActionType.CREATE),
        ],
    )
    async def create_schedule(
        self,
        id: UUID,
        schedule_data: CampaignScheduleCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.create_schedule(
            session=db_session,
            campaign_id=id,
            organization_id=current_user.organization_id,
            schedule_data=schedule_data,
        )

    @action(
        method="PATCH",
        detail=True,
        url_path="schedule",
        summary="Update schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.UPDATE),
            requires_user_permission("CampaignSchedule", ActionType.UPDATE),
        ],
    )
    async def update_schedule(
        self,
        id: UUID,
        schedule_data: CampaignScheduleUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        _, schedule = await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self.service.get_schedule(db_session, id),
        )
        return await self.service.update_schedule(
            session=db_session,
            schedule_id=schedule.id,
            schedule_data=schedule_data,
        )

    @action(
        method="GET",
        detail=True,
        url_path="schedule",
        summary="Get schedule",
        response_model=CampaignScheduleResponse,
        dependencies=[
            requires_user_permission("Campaign", ActionType.READ),
            requires_user_permission("CampaignSchedule", ActionType.READ),
        ],
    )
    async def get_schedule(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_schedule(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="segments",
        summary="Create segment",
        response_model=CitizenSegmentResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.CREATE)],
    )
    async def create_segment(
        self,
        segment_data: SegmentCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.create_segment(
            session=db_session,
            segment_data=segment_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=current_user.branch_id,
        )

    @action(
        method="GET",
        detail=False,
        url_path="segments",
        summary="List segments",
        response_model=PaginatedResponse[CitizenSegmentResponse],
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def list_segments(
        self,
        params: SegmentFilterParams = Depends(),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.list_segments(
            session=db_session,
            organization_id=current_user.organization_id,
            params=params,
        )

    @action(
        method="GET",
        url_path="segments/{id}",
        summary="Get segment",
        response_model=CitizenSegmentResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def get_segment(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.READ,
        )

    @action(
        method="PATCH",
        url_path="segments/{id}",
        summary="Update segment",
        response_model=CitizenSegmentResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def update_segment(
        self,
        id: UUID,
        segment_data: SegmentUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        segment = await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_segment(db_session, segment, segment_data)

    @action(
        method="DELETE",
        url_path="segments/{id}",
        summary="Delete segment",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.DELETE)],
    )
    async def delete_segment(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_segment(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="segments/preview",
        summary="Preview dynamic segment",
        response_model=PaginatedResponse[CitizenResponse],
        dependencies=[
            requires_user_permission("CitizenSegment", ActionType.READ),
            Depends(get_current_active_user),
        ],
    )
    async def preview_dynamic_segment(
        self,
        preview_data: DynamicSegmentPreviewRequest,
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.preview_dynamic_segment(
            session=db_session,
            filter_criteria=dict(preview_data.filter_criteria),
            page=preview_data.page,
            size=preview_data.size,
        )

    @action(
        method="GET",
        # detail=True,
        url_path="segments/{id}/members",
        summary="List segment members",
        response_model=PaginatedResponse[SegmentMembershipResponse],
        dependencies=[requires_user_permission("CitizenSegment", ActionType.READ)],
    )
    async def list_segment_members(
        self,
        id: UUID,
        page: int = Query(1, ge=1),
        size: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_segment_members(
            session=db_session,
            segment_id=id,
            page=page,
            size=size,
        )

    @action(
        method="POST",
        detail=True,
        url_path="segments/{segment_id}/members",
        summary="Add members to segment",
        response_model=BulkSegmentMemberResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def add_segment_members(
        self,
        id: UUID,
        segment_id: UUID,
        citizen_ids: List[UUID] = Body(..., embed=True),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.UPDATE,
            ),
        )
        return await self.service.add_citizens_to_segment(
            session=db_session,
            segment_id=segment_id,
            citizen_ids=citizen_ids,
            user_id=current_user.id,
            campaign_id=id,
        )

    @action(
        method="DELETE",
        detail=True,
        url_path="segments/{segment_id}/members",
        summary="Remove members from segment",
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def remove_segment_members(
        self,
        id: UUID,
        segment_id: UUID,
        membership_data: SegmentMembershipRemove,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.UPDATE,
            ),
        )
        await self.service.remove_citizens_from_segment(
            session=db_session,
            segment_id=segment_id,
            citizen_ids=membership_data.citizen_ids,
            campaign_id=id,
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @action(
        method="PATCH",
        url_path="segments/members/{membership_id}",
        summary="Update segment membership",
        response_model=SegmentMembershipResponse,
        dependencies=[requires_user_permission("CitizenSegment", ActionType.UPDATE)],
    )
    async def update_segment_membership(
        self,
        membership_id: UUID,
        membership_data: SegmentMembershipUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        membership = await self.service.membership_crud.get_by_id(
            session=db_session,
            id=membership_id,
        )

        if not membership:
            raise NotFoundError(detail=f"Membership {membership_id} not found")

        await self._check_segment_permission(
            db_session=db_session,
            user=current_user,
            segment_id=membership.segment_id,
            action=ActionType.UPDATE,
        )

        return await self.service.update_segment_membership(
            session=db_session,
            membership_id=membership_id,
            subscription_status=membership_data.subscription_status,
            is_blacklisted=membership_data.is_blacklisted,
        )

    @action(
        method="POST",
        detail=True,
        url_path="segments/{segment_id}/campaign",
        summary="Assign campaign to all segment members",
        response_model=SegmentCampaignAssignmentResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def assign_campaign_to_segment(
        self,
        id: UUID,
        segment_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await asyncio.gather(
            self._check_campaign_permission(
                db_session=db_session,
                user=current_user,
                campaign_id=id,
                action=ActionType.UPDATE,
            ),
            self._check_segment_permission(
                db_session=db_session,
                user=current_user,
                segment_id=segment_id,
                action=ActionType.READ,
            ),
        )
        return await self.service.assign_campaign_to_segment(
            session=db_session,
            segment_id=segment_id,
            campaign_id=id,
        )

    @action(
        method="GET",
        detail=True,
        url_path="deliveries",
        summary="List campaign deliveries",
        response_model=PaginatedResponse[CampaignDeliveryResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_deliveries(
        self,
        id: UUID,
        status_filter: Optional[DeliveryStatus] = Query(None),
        page: int = Query(1, ge=1),
        size: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        return await self.service.get_campaign_deliveries(
            session=db_session,
            campaign_id=id,
            status=status_filter,
            page=page,
            size=size,
        )

    @action(
        method="GET",
        detail=True,
        url_path="stats",
        summary="Get campaign statistics",
        response_model=CampaignStatistics,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_statistics(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignStatistics:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        stats = await self.service.get_campaign_statistics(db_session, id)
        return CampaignStatistics(
            campaign_id=id,
            total_targets=stats["total_targets"],
            total_deliveries=stats["total_deliveries"],
            delivery_stats=stats["delivery_stats"],
            success_rate=stats["success_rate"],
            failure_rate=stats["failure_rate"],
        )

    @action(
        method="GET",
        detail=True,
        url_path="analytics",
        summary="Get campaign analytics",
        response_model=CampaignAnalyticsResponse,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_analytics(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> CampaignAnalyticsResponse:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )
        analytics = await self.service.get_campaign_analytics(db_session, id)
        return CampaignAnalyticsResponse(**analytics)

    @action(
        method="GET",
        detail=False,
        url_path="summary",
        summary="Get organization campaign summary",
        response_model=OrganizationCampaignSummary,
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def get_org_summary(
        self,
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> OrganizationCampaignSummary:
        summary = await self.service.get_organization_campaign_summary(
            session=db_session,
            organization_id=current_user.organization_id,
        )
        return OrganizationCampaignSummary.model_validate(
            {
                "organization_id": current_user.organization_id,
                **summary,
            }
        )

    @action(
        method="POST",
        detail=False,
        url_path="templates",
        summary="Create template",
        response_model=CampaignTemplateResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.CREATE)],
    )
    async def create_template(
        self,
        template_data: CampaignTemplateCreate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        return await self.service.create_template(
            session=db_session,
            template_data=template_data,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            branch_id=current_user.branch_id,
        )

    @action(
        method="GET",
        detail=False,
        url_path="templates",
        summary="List templates",
        response_model=PaginatedResponse[CampaignTemplateResponse],
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def list_templates(
        self,
        channel_type: Optional[str] = Query(None),
        is_active: bool = Query(True),
        page: Optional[int] = Query(1, ge=1),
        size: Optional[int] = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        from database.utils.types import ChannelType

        channel_enum = ChannelType(channel_type) if channel_type else None

        return await self.service.list_templates(
            session=db_session,
            organization_id=current_user.organization_id,
            channel_type=channel_enum,
            is_active=is_active,
            page=page,
            size=size,
        )

    @action(
        method="GET",
        url_path="templates/{id}",
        summary="Get template",
        response_model=CampaignTemplateResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def get_template(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        template = await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.READ,
        )
        return template

    @action(
        method="PATCH",
        detail=False,
        url_path="templates/{id}",
        summary="Update template",
        response_model=CampaignTemplateResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.UPDATE)],
    )
    async def update_template(
        self,
        id: UUID,
        template_data: CampaignTemplateUpdate,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.UPDATE,
        )
        return await self.service.update_template(
            session=db_session,
            template_id=id,
            template_data=template_data,
        )

    @action(
        method="DELETE",
        url_path="templates/{id}",
        summary="Delete template",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.DELETE)],
    )
    async def delete_template(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.DELETE,
        )
        await self.service.delete_template(db_session, id)

    @action(
        method="POST",
        detail=False,
        url_path="templates/{id}/render",
        summary="Render template",
        response_model=TemplateRenderResponse,
        dependencies=[requires_user_permission("CampaignTemplate", ActionType.READ)],
    )
    async def render_template(
        self,
        id: UUID,
        render_request: TemplateRenderRequest,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ):
        template = await self._check_template_permission(
            db_session=db_session,
            user=current_user,
            template_id=id,
            action=ActionType.READ,
        )
        return await self.service.render_template(
            template=template,
            placeholder_values=render_request.placeholder_values,
        )

    @action(
        method="GET",
        detail=True,
        url_path="attachments",
        summary="List campaign attachments",
        response_model=List[DocumentResponse],
        dependencies=[requires_user_permission("Campaign", ActionType.READ)],
    )
    async def list_attachments(
        self,
        id: UUID,
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> List[DocumentResponse]:
        await self._check_campaign_permission(
            db_session=db_session,
            user=current_user,
            campaign_id=id,
            action=ActionType.READ,
        )

        filters = DocumentFilter(
            entity_id=id,
            entity_type="CAMPAIGN",
            min_file_size=None,
            max_file_size=None,
        )
        documents = await self.document_service.list_documents(
            db_session=db_session, filters=filters, page=None, size=None
        )
        return await self.document_service.to_list_response_models(documents)  # type: ignore

    @action(
        method="DELETE",
        detail=True,
        url_path="attachments/{attachment_id}",
        summary="Delete attachment",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[requires_user_permission("Campaign", ActionType.UPDATE)],
    )
    async def delete_attachment(
        self,
        id: UUID,
        attachment_id: UUID = Path(..., description="Attachment document ID"),
        current_user: User = Depends(get_current_active_user),
        db_session: AsyncSession = Depends(db_session_manager.get_db),
    ) -> None:
        document = await self.document_service.crud.get_one(
            session=db_session,
            filters=[
                FilterSchema(field="id", op="==", value=attachment_id),
                FilterSchema(field="entity_id", op="==", value=id),
                FilterSchema(field="entity_type", op="==", value="CAMPAIGN"),
            ],
            prefetch=["campaign"],
        )
        if not document:
            raise NotFoundError(detail="Document not found")
        if document.campaign:
            if document.campaign.id != id:
                raise ForbiddenError(detail="Document not attached to this campaign")
            await self._check_resource_permission(
                db_session=db_session,
                user=current_user,
                resource=document.campaign,
                resource_id=document.campaign.id,
                resource_type="Campaign",
                action=ActionType.UPDATE,
            )
        else:
            raise ForbiddenError(detail="Document not attached to any campaign")
        await self.document_service.delete_document(
            db_session=db_session,
            document=document,
            user=current_user,
        )

    async def _check_campaign_permission(
        self,
        db_session: AsyncSession,
        user: User,
        campaign_id: UUID,
        action: ActionType,
        prefetch: Optional[List[str]] = None,
    ) -> Campaign:
        """Check permission and return campaign if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        Reuses the fetched campaign to avoid duplicate DB calls.
        Includes ABAC check for organization isolation.
        """
        campaign = await self.service.get_campaign(
            session=db_session,
            campaign_id=campaign_id,
            prefetch=prefetch,
        )

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=campaign,
            resource_id=campaign_id,
            resource_type="Campaign",
            action=action,
        )

    async def _check_segment_permission(
        self,
        db_session: AsyncSession,
        user: User,
        segment_id: UUID,
        action: ActionType,
    ) -> CitizenSegment:
        """Check permission and return segment if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        """
        segment = await self.service.get_segment(db_session, segment_id)

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=segment,
            resource_id=segment_id,
            resource_type="CitizenSegment",
            action=action,
        )

    async def _check_template_permission(
        self,
        db_session: AsyncSession,
        user: User,
        template_id: UUID,
        action: ActionType,
    ) -> CampaignTemplate:
        """Check permission and return template if granted.
        Raises ForbiddenError or NotFoundError if not authorized or not found.
        """
        template = await self.service.get_template(db_session, template_id)
        if not template:
            raise NotFoundError(detail="Template not found")

        return await self._check_resource_permission(
            db_session=db_session,
            user=user,
            resource=template,
            resource_id=template_id,
            resource_type="CampaignTemplate",
            action=action,
        )


"""Application factory and dependency injection container."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

if TYPE_CHECKING:
    from .config import BaseConfig


@dataclass(frozen=True)
class AppConfig:
    version: str
    title: str = "Pizza Man Chicken Man API"
    description: str = """
        Pizza Man Chicken Man Backend API
        
        A comprehensive backend API for Pizza Man Chicken Man restaurant orders and management
    """
    contact: dict[str, str] | None = field(
        default_factory=lambda: {
            "name": "Pizza Man Chicken Man Team",
            "email": "info@pizzamanchickenman.com",
        }
    )
    openapi_url: str = "/openapi.json"
    docs_url: str | None = None
    redoc_url: str | None = None


class ServiceContainer:
    def __init__(self, config: "BaseConfig"):
        self._config = config
        self._services = {}

    def _lazy_init(self, service_name: str, factory):
        if service_name not in self._services:
            self._services[service_name] = factory()
        return self._services[service_name]

    @property
    def storage(self):
        from services.storage import AWSStorageService

        return self._lazy_init(
            "storage", lambda: AWSStorageService(self._config.aws_config)
        )

    @property
    def database(self):
        from database.session import DBSessionManager

        return self._lazy_init(
            "database", lambda: DBSessionManager(self._config.db_url)
        )

    @property
    def cache(self):
        from services.redis_service import RedisService

        return self._lazy_init("cache", lambda: RedisService(self._config.redis_url))

    @property
    def security(self):
        from services.security import SecurityService

        return self._lazy_init(
            "security",
            lambda: SecurityService(
                secret_key=self._config.secret_key,
                algorithm=self._config.algorithm,
                access_token_expire_minutes=self._config.access_token_expire_minutes,
                refresh_token_expire_days=self._config.refresh_token_expire_days,
            ),
        )

    @property
    def permissions(self):
        from services.permission import PermissionService

        return self._lazy_init("permissions", lambda: PermissionService(self.cache))

    @property
    def documents(self):
        from services.document import DocumentService

        return self._lazy_init("documents", lambda: DocumentService(self.storage))

    @property
    def users(self):
        from services.user_service import UserService

        return self._lazy_init(
            "users",
            lambda: UserService(
                security_service=self.security,
                permission_service=self.permissions,
            ),
        )

    @property
    def organizations(self):
        from services.org_service import OrganizationService

        return self._lazy_init("organizations", OrganizationService)

    @property
    def tasks(self):
        from services.task_service import TaskService

        return self._lazy_init(
            "tasks", lambda: TaskService(document_service=self.documents)
        )

    @property
    def campaigns(self):
        from services.campaign import CampaignService

        return self._lazy_init("campaigns", CampaignService)

    @property
    def approvals(self):
        from services.approval import ApprovalService

        return self._lazy_init("approvals", ApprovalService)

    @property
    def sms(self):
        from services.messaging import SMSService

        return self._lazy_init("sms", lambda: SMSService(self._config.sms_config))

    @property
    def whatsapp(self):
        from services.messaging import WhatsAppService

        return self._lazy_init(
            "whatsapp", lambda: WhatsAppService(self._config.whatsapp_config)
        )

    @property
    def email(self):
        from services.messaging import EmailService

        return self._lazy_init("email", lambda: EmailService(self._config.email_config))

    @property
    def otp(self):
        from services.otp_service import OTPService

        return self._lazy_init("otp", lambda: OTPService(self.email))

    @property
    def auth(self):
        from services.auth import AuthService

        return self._lazy_init("auth", lambda: AuthService(self.security, self.otp))


class ApplicationContainer:
    def __init__(self, config: "BaseConfig"):
        self.config = config
        self.services = ServiceContainer(config)
        self.metadata = AppConfig(version=config.api_version)
        self.started_at = datetime.now(UTC)

        self._app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title=self.metadata.title,
            description=self.metadata.description,
            version=self.metadata.version,
            contact=self.metadata.contact,
            lifespan=self._lifespan,
            default_response_class=ORJSONResponse,
            docs_url=None,
            redoc_url=None,
        )

        self._configure(app)
        return app

    def _configure(self, app: FastAPI) -> None:
        self._setup_templates(app)
        self._setup_routes(app)
        self._setup_middleware(app)
        self._setup_error_handlers(app)

    def _setup_templates(self, app: FastAPI) -> None:
        templates_dir = Path(__file__).parent.parent / "templates"
        app.state.templates = Jinja2Templates(directory=str(templates_dir))

    def _setup_routes(self, app: FastAPI) -> None:
        from .routes import RouterRegistry

        RouterRegistry(self.config, self.services).register_all(app)

    def _setup_middleware(self, app: FastAPI) -> None:
        if self.config.environment != "development":
            return

        from fastapi.middleware.cors import CORSMiddleware

        from middleware.tenant_isolation import tenant_isolation_middleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        app.middleware("http")(tenant_isolation_middleware)

    def _setup_error_handlers(self, app: FastAPI) -> None:
        from .errors_handler import register_error_handlers

        register_error_handlers(app)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        await self._startup()
        # WebSocket connection cleanup is now handled by ARQ task scheduler
        try:
            yield
        finally:
            await self._shutdown()

    async def _startup(self) -> None:
        logger.info("Starting application...")

        tasks = [
            self._init_task_service,
            self._init_database_signals,
            self._init_seed_data,
            self._init_weather_service,
        ]

        for task in tasks:
            try:
                await task()
            except Exception as e:
                logger.error(f"Startup task {task.__name__} failed: {e}")

        logger.success("Application started successfully")

    async def _init_task_service(self) -> None:
        from tasks.service import task_service

        await task_service.initialize()

    async def _init_database_signals(self) -> None:
        from database.signals import register_all_signals

        register_all_signals()

    async def _init_seed_data(self) -> None:
        from .initializer import create_seed_data

        await create_seed_data(user_service=self.services.users)

    async def _init_weather_service(self) -> None:
        from services.weather_service import weather_service
        from tasks.weather_tasks import trigger_immediate_refresh

        await weather_service.initialize()

        locations = await weather_service.get_available_locations()
        if locations:
            logger.info(f"Weather service ready with {len(locations)} locations")
        else:
            logger.info("Triggering initial weather data fetch...")
            await trigger_immediate_refresh()

    async def _shutdown(self) -> None:
        logger.info("Shutting down application...")

        # Close services in reverse dependency order
        # TODO: Implement proper cleanup for:
        # - Background tasks
        # - Redis connections
        # - Database connections

        logger.info("Application stopped")

    @property
    def instance(self) -> FastAPI:
        return self._app

"""Email authentication, verification, password reset, and organizations."""

import asyncio
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.security import create_access_token, create_guest_access_token, create_one_time_token, create_refresh_token, decode_token, hash_one_time_token, hash_password, verify_password
from app.db.database import get_db
from app.db.models import EmailToken, Organization, OrganizationInvite, OrganizationMember, Profile
from app.dependencies import GuestUser, require_user
from app.models.schemas import CreateOrganizationRequest, EmailRequest, InviteMemberRequest, LoginRequest, MemberItem, OrganizationSummary, RefreshRequest, RegisterRequest, ResetPasswordRequest, SupabaseExchangeRequest, TokenRequest, TokenResponse, UserProfile
from app.services.email_service import EmailDeliveryError, send_email
from app.services.supabase_auth import SupabaseAuthError, get_user as get_supabase_user, sign_in as supabase_sign_in

router = APIRouter(prefix="/api/auth", tags=["auth"])
org_router = APIRouter(prefix="/api/organizations", tags=["organizations"])


def _cross_site_cookie_options() -> tuple[bool, str]:
    """Use cross-site cookies when the API and frontend are separately hosted."""
    cross_site = settings.app_env.lower() == "production" or settings.frontend_url.lower().startswith("https://")
    return cross_site, "none" if cross_site else "lax"


@router.post("/guest", response_model=TokenResponse)
async def guest_login(response: Response):
    """Create a short-lived signed visitor session without a database user."""
    guest_id = f"guest:{secrets.token_urlsafe(18)}"
    access = create_guest_access_token(guest_id)
    secure, samesite = _cross_site_cookie_options()
    response.set_cookie(
        "review_access", access, httponly=True, secure=secure,
        samesite=samesite, max_age=settings.guest_session_minutes * 60, path="/",
    )
    response.set_cookie(
        "review_csrf", secrets.token_urlsafe(32), httponly=False, secure=secure,
        samesite=samesite, max_age=settings.guest_session_minutes * 60, path="/",
    )
    return TokenResponse(
        access_token=access,
        user=UserProfile(
            id=guest_id, email="guest@local.invalid", display_name="游客体验",
            email_verified=True, organizations=[], created_at=datetime.now(timezone.utc).isoformat(), is_guest=True,
        ),
    )


@router.get("/csrf")
async def csrf_token(request: Request, response: Response):
    """Expose the double-submit token to the separate frontend origin.

    The cookie is intentionally readable by the browser, but a cross-origin
    frontend cannot access cookies belonging to the API origin. Returning the
    existing value lets the frontend send the matching header without
    weakening CSRF validation on mutating endpoints.
    """
    token = request.cookies.get("review_csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        secure, samesite = _cross_site_cookie_options()
        response.set_cookie(
            "review_csrf",
            token,
            httponly=False,
            secure=secure,
            samesite=samesite,
            max_age=settings.refresh_token_days * 86400,
            path="/",
        )
    return {"csrf_token": token}


def _expired(value: datetime) -> bool:
    normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return normalized < datetime.now(timezone.utc)


def _profile_response(user: Profile, memberships: list[OrganizationMember]) -> TokenResponse:
    orgs = [OrganizationSummary(id=m.organization.id, name=m.organization.name, role=m.role) for m in memberships]
    return TokenResponse(user=UserProfile(id=user.id, email=user.email, display_name=user.display_name, email_verified=bool(user.email_verified_at), organizations=orgs, created_at=str(user.created_at)))


def _set_auth_cookies(response: Response, access: str, refresh: str, *, remember_me: bool = False) -> None:
    secure, samesite = _cross_site_cookie_options()
    access_max_age = (settings.refresh_token_days * 24 * 60 if remember_me else settings.access_token_minutes) * 60
    response.set_cookie("review_access", access, httponly=True, secure=secure, samesite=samesite, max_age=access_max_age, path="/")
    response.set_cookie("review_refresh", refresh, httponly=True, secure=secure, samesite=samesite, max_age=settings.refresh_token_days * 86400, path="/api/auth")
    response.set_cookie("review_csrf", secrets.token_urlsafe(32), httponly=False, secure=secure, samesite=samesite, max_age=settings.refresh_token_days * 86400, path="/")


async def _load_memberships(db: AsyncSession, user_id: str) -> list[OrganizationMember]:
    result = await db.execute(select(OrganizationMember).options(selectinload(OrganizationMember.organization)).where(OrganizationMember.user_id == user_id))
    return list(result.scalars().all())


async def _send_token(db: AsyncSession, user: Profile, purpose: str, subject: str, path: str, expires: timedelta) -> None:
    raw, token_hash = create_one_time_token()
    db.add(EmailToken(user_id=user.id, purpose=purpose, token_hash=token_hash, expires_at=datetime.now(timezone.utc) + expires))
    await db.commit()
    await asyncio.to_thread(send_email, user.email, subject, f'<p>请打开链接完成操作：</p><p><a href="{settings.frontend_url}{path}?token={raw}">继续操作</a></p>')


def _email_error(error: EmailDeliveryError) -> HTTPException:
    if error.status_code == 401:
        detail = "Resend API Key 无效或已撤销，请在同一 Resend 账户重新创建并更新 Render 配置"
    elif error.status_code == 403:
        detail = "Resend 拒绝了发件请求（403），请确认 API Key 属于当前账户且发件地址/测试收件人符合 Resend 限制"
    elif error.status_code:
        detail = f"邮件服务返回 HTTP {error.status_code}，请检查 Resend 配置"
    else:
        detail = "邮件服务暂时不可用，请检查 Render 环境变量或稍后重试"
    return HTTPException(502, detail)


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, response: Response, db: AsyncSession = Depends(get_db)):
    email = str(req.email).lower()
    if await db.scalar(select(Profile).where(Profile.email == email)):
        raise HTTPException(409, "该邮箱已注册")
    user = Profile(email=email, password_hash=hash_password(req.password), display_name=req.display_name or email.split("@")[0])
    db.add(user)
    await db.flush()
    invited = None
    if req.invite_token:
        invited = await db.scalar(select(OrganizationInvite).where(OrganizationInvite.token_hash == hash_one_time_token(req.invite_token), OrganizationInvite.accepted_at.is_(None)))
        if not invited or _expired(invited.expires_at) or invited.email != email:
            raise HTTPException(400, "工作区邀请无效或已过期")
    if invited:
        now = datetime.now(timezone.utc)
        consumed = await db.execute(
            update(OrganizationInvite)
            .where(OrganizationInvite.id == invited.id, OrganizationInvite.accepted_at.is_(None), OrganizationInvite.expires_at > now)
            .values(accepted_at=now)
        )
        if consumed.rowcount != 1:
            raise HTTPException(400, "工作区邀请无效或已使用")
        db.add(OrganizationMember(organization_id=invited.organization_id, user_id=user.id, role="member"))
    else:
        organization = Organization(name=req.organization_name or f"{user.display_name} 的工作区", created_by=user.id)
        db.add(organization)
        await db.flush()
        db.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="owner"))
    await db.commit()
    await db.refresh(user)
    try:
        await _send_token(db, user, "verify_email", "验证您的评审平台邮箱", "/verify-email", timedelta(hours=24))
    except EmailDeliveryError as exc:
        raise _email_error(exc) from exc
    memberships = await _load_memberships(db, user.id)
    access, refresh = create_access_token(user.id, token_version=user.token_version), create_refresh_token(user.id, token_version=user.token_version)
    _set_auth_cookies(response, access, refresh)
    return _profile_response(user, memberships)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(Profile).where(Profile.email == str(req.email).lower()))
    # Supabase-managed profiles intentionally have no local password hash.
    # Treat them as invalid credentials instead of passing None to passlib,
    # which would surface an internal 500 error.
    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "邮箱或密码错误")
    if not user.email_verified_at:
        raise HTTPException(403, "邮箱尚未验证，请先验证邮箱")
    memberships = await _load_memberships(db, user.id)
    access, refresh = create_access_token(user.id, req.remember_me, user.token_version), create_refresh_token(user.id, user.token_version)
    _set_auth_cookies(response, access, refresh, remember_me=req.remember_me)
    return _profile_response(user, memberships)


@router.post("/supabase/exchange", response_model=TokenResponse)
async def exchange_supabase_session(req: SupabaseExchangeRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Exchange a verified Supabase session for the platform's existing cookies."""
    try:
        remote_user = await asyncio.to_thread(get_supabase_user, req.access_token)
    except SupabaseAuthError as exc:
        raise HTTPException(401, "Supabase 登录会话无效或已过期") from exc

    supabase_id = str(remote_user["id"])
    email = str(remote_user["email"]).lower()
    confirmed_at = remote_user.get("email_confirmed_at") or remote_user.get("confirmed_at")
    if not confirmed_at:
        raise HTTPException(403, "邮箱尚未验证，请先完成 Supabase 邮箱验证")
    verified_at = datetime.now(timezone.utc)
    if isinstance(confirmed_at, str):
        try:
            verified_at = datetime.fromisoformat(confirmed_at.replace("Z", "+00:00"))
        except ValueError:
            pass
    metadata = remote_user.get("user_metadata") if isinstance(remote_user.get("user_metadata"), dict) else {}
    display_name = req.display_name or metadata.get("display_name") or metadata.get("full_name")
    user = await db.get(Profile, supabase_id)
    if not user:
        user = await db.scalar(select(Profile).where(Profile.email == email))
    if user:
        user.email = email
        user.email_verified_at = user.email_verified_at or verified_at
        if display_name and not user.display_name:
            user.display_name = str(display_name)[:100]
    else:
        user = Profile(
            id=supabase_id,
            email=email,
            password_hash=None,
            display_name=str(display_name)[:100] if display_name else email.split("@")[0],
            email_verified_at=verified_at,
        )
        db.add(user)
        await db.flush()

    memberships = await _load_memberships(db, user.id)
    if not memberships and req.invite_token:
        invited = await db.scalar(select(OrganizationInvite).where(OrganizationInvite.token_hash == hash_one_time_token(req.invite_token), OrganizationInvite.accepted_at.is_(None)))
        if not invited or _expired(invited.expires_at) or invited.email != email:
            raise HTTPException(400, "工作区邀请无效或已过期")
        consumed = await db.execute(update(OrganizationInvite).where(OrganizationInvite.id == invited.id, OrganizationInvite.accepted_at.is_(None), OrganizationInvite.expires_at > datetime.now(timezone.utc)).values(accepted_at=datetime.now(timezone.utc)))
        if consumed.rowcount != 1:
            raise HTTPException(400, "工作区邀请无效或已使用")
        db.add(OrganizationMember(organization_id=invited.organization_id, user_id=user.id, role="member"))
        await db.commit()
        memberships = await _load_memberships(db, user.id)
    elif not memberships:
        organization_name = req.organization_name or metadata.get("organization_name") or f"{user.display_name or email} 的工作区"
        organization = Organization(name=str(organization_name)[:120], created_by=user.id)
        db.add(organization)
        await db.flush()
        db.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="owner"))
        await db.commit()
        memberships = await _load_memberships(db, user.id)
    else:
        await db.commit()
    await db.refresh(user)
    access, refresh = create_access_token(user.id, token_version=user.token_version), create_refresh_token(user.id, token_version=user.token_version)
    _set_auth_cookies(response, access, refresh)
    return _profile_response(user, memberships)


@router.post("/supabase/login", response_model=TokenResponse)
async def supabase_login(req: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Proxy Supabase password login so browsers do not depend on Supabase CORS/TLS."""
    try:
        session = await asyncio.to_thread(supabase_sign_in, str(req.email).lower(), req.password)
    except SupabaseAuthError as exc:
        message = str(exc)
        normalized = message.strip().lower().replace("-", "_")
        if normalized in {"invalid login credentials", "invalid_credentials", "invalid_grant"} or "invalid login" in normalized:
            raise HTTPException(401, "邮箱或密码错误") from exc
        if "email not confirmed" in normalized or "email_not_confirmed" in normalized:
            raise HTTPException(403, "邮箱尚未验证，请先完成 Supabase 邮箱验证") from exc
        raise HTTPException(502, "Supabase 登录服务暂时不可用，请稍后重试") from exc
    access_token = session.get("access_token")
    if not isinstance(access_token, str):
        raise HTTPException(502, "Supabase 未返回有效登录会话")
    # Reuse the existing exchange path to keep profile/organization behavior identical.
    return await exchange_supabase_session(
        SupabaseExchangeRequest(access_token=access_token), response, db
    )


@router.post("/verify-email")
async def verify_email(req: TokenRequest, db: AsyncSession = Depends(get_db)):
    token_hash = hash_one_time_token(req.token)
    token = await db.scalar(select(EmailToken).where(EmailToken.token_hash == token_hash, EmailToken.purpose == "verify_email", EmailToken.used_at.is_(None)))
    if not token or _expired(token.expires_at):
        raise HTTPException(400, "验证链接无效或已过期")
    consumed = await db.execute(update(EmailToken).where(EmailToken.id == token.id, EmailToken.used_at.is_(None)).values(used_at=datetime.now(timezone.utc)))
    if consumed.rowcount != 1:
        raise HTTPException(400, "验证链接无效或已使用")
    user = await db.get(Profile, token.user_id)
    user.email_verified_at = datetime.now(timezone.utc)
    await db.commit()
    return {"status": "verified"}


@router.post("/resend-verification")
async def resend_verification(req: EmailRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(Profile).where(Profile.email == str(req.email).lower()))
    if user and not user.email_verified_at:
        try:
            await _send_token(db, user, "verify_email", "重新验证评审平台邮箱", "/verify-email", timedelta(hours=24))
        except EmailDeliveryError as exc:
            raise _email_error(exc) from exc
    return {"status": "sent"}


@router.post("/forgot-password")
async def forgot_password(req: EmailRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(Profile).where(Profile.email == str(req.email).lower()))
    if user:
        try:
            await _send_token(db, user, "reset_password", "重置评审平台密码", "/reset-password", timedelta(hours=1))
        except EmailDeliveryError as exc:
            raise _email_error(exc) from exc
    return {"status": "sent"}


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    token = await db.scalar(select(EmailToken).where(EmailToken.token_hash == hash_one_time_token(req.token), EmailToken.purpose == "reset_password", EmailToken.used_at.is_(None)))
    if not token or _expired(token.expires_at):
        raise HTTPException(400, "重置链接无效或已过期")
    consumed = await db.execute(update(EmailToken).where(EmailToken.id == token.id, EmailToken.used_at.is_(None)).values(used_at=datetime.now(timezone.utc)))
    if consumed.rowcount != 1:
        raise HTTPException(400, "重置链接无效或已使用")
    user = await db.get(Profile, token.user_id)
    user.password_hash = hash_password(req.password)
    user.token_version += 1
    await db.execute(update(EmailToken).where(EmailToken.user_id == user.id, EmailToken.purpose == "reset_password", EmailToken.used_at.is_(None)).values(used_at=datetime.now(timezone.utc)))
    await db.commit()
    return {"status": "reset"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh(response: Response, req: RefreshRequest | None = None, refresh_cookie: str | None = Cookie(default=None, alias="review_refresh"), db: AsyncSession = Depends(get_db)):
    raw_refresh = (req.refresh_token if req else None) or refresh_cookie
    payload = decode_token(raw_refresh, "refresh") if raw_refresh else None
    user = await db.get(Profile, payload["sub"]) if payload else None
    if not user or not user.is_active or not user.email_verified_at or int(payload.get("ver", 0)) != user.token_version:
        raise HTTPException(401, "刷新令牌无效")
    access, refresh_token = create_access_token(user.id, token_version=user.token_version), create_refresh_token(user.id, user.token_version)
    _set_auth_cookies(response, access, refresh_token)
    return _profile_response(user, await _load_memberships(db, user.id))


@router.post("/logout")
async def logout(response: Response, user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    if isinstance(user, GuestUser):
        from app.services.guest_review_service import guest_review_store
        guest_review_store.purge(user.id)
        response.delete_cookie("review_access", path="/")
        response.delete_cookie("review_csrf", path="/")
        return {"status": "logged_out"}
    user.token_version += 1
    await db.commit()
    response.delete_cookie("review_access", path="/")
    response.delete_cookie("review_refresh", path="/api/auth")
    response.delete_cookie("review_csrf", path="/")
    return {"status": "logged_out"}


@router.get("/me", response_model=UserProfile)
async def me(user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    if isinstance(user, GuestUser):
        return UserProfile(id=user.id, email=user.email, display_name=user.display_name, email_verified=True, organizations=[], created_at=user.email_verified_at.isoformat(), is_guest=True)
    memberships = await _load_memberships(db, user.id)
    return _profile_response(user, memberships).user


@org_router.post("", response_model=OrganizationSummary, status_code=201)
async def create_organization(req: CreateOrganizationRequest, user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    organization = Organization(name=req.name, created_by=user.id)
    db.add(organization)
    await db.flush()
    db.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="owner"))
    await db.commit()
    return OrganizationSummary(id=organization.id, name=organization.name, role="owner")


@org_router.get("/{organization_id}/members", response_model=list[MemberItem])
async def list_members(organization_id: str, user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    member = await db.scalar(select(OrganizationMember).where(OrganizationMember.organization_id == organization_id, OrganizationMember.user_id == user.id))
    if not member:
        raise HTTPException(404, "资源不存在")
    result = await db.execute(select(OrganizationMember).options(selectinload(OrganizationMember.user)).where(OrganizationMember.organization_id == organization_id))
    return [MemberItem(user_id=m.user_id, email=m.user.email, display_name=m.user.display_name, role=m.role, joined_at=str(m.created_at)) for m in result.scalars().all()]


@org_router.post("/{organization_id}/invites")
async def invite_member(organization_id: str, req: InviteMemberRequest, user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    member = await db.scalar(select(OrganizationMember).where(OrganizationMember.organization_id == organization_id, OrganizationMember.user_id == user.id, OrganizationMember.role == "owner"))
    if not member:
        raise HTTPException(404, "资源不存在")
    raw = secrets.token_urlsafe(32)
    invite = OrganizationInvite(organization_id=organization_id, email=str(req.email).lower(), token_hash=hash_one_time_token(raw), invited_by=user.id, expires_at=datetime.now(timezone.utc) + timedelta(hours=72))
    db.add(invite)
    await db.commit()
    try:
        await asyncio.to_thread(send_email, str(req.email), "加入评审工作区", f'<p><a href="{settings.frontend_url}/register?invite={raw}">接受工作区邀请</a></p>')
    except EmailDeliveryError as exc:
        raise _email_error(exc) from exc
    return {"status": "sent"}


@org_router.delete("/{organization_id}/members/{member_user_id}", status_code=204)
async def remove_member(organization_id: str, member_user_id: str, user: Profile = Depends(require_user), db: AsyncSession = Depends(get_db)):
    owner = await db.scalar(select(OrganizationMember).where(OrganizationMember.organization_id == organization_id, OrganizationMember.user_id == user.id, OrganizationMember.role == "owner"))
    if not owner or member_user_id == user.id:
        raise HTTPException(404, "资源不存在")
    member = await db.scalar(select(OrganizationMember).where(OrganizationMember.organization_id == organization_id, OrganizationMember.user_id == member_user_id))
    if not member or member.role == "owner":
        raise HTTPException(404, "资源不存在")
    await db.delete(member)
    await db.commit()

"""Small HTTP adapter for Supabase Auth's user endpoint.

The service role key is kept server-side. The anon key is preferred when it is
configured, because it is the key intended for Auth API calls from clients.
"""

import json
import urllib.error
import urllib.request

from app.core.config import settings


class SupabaseAuthError(RuntimeError):
    pass


def _api_keys() -> list[str]:
    """Return configured keys in least-privileged order, without duplicates."""
    values = [settings.supabase_anon_key, settings.supabase_service_role_key]
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def _is_api_key_error(message: str | None) -> bool:
    normalized = (message or "").lower()
    return "api key" in normalized or "apikey" in normalized or "invalid jwt" in normalized


def sign_in(email: str, password: str) -> dict:
    """Authenticate against Supabase without exposing the auth request to the browser."""
    if not settings.supabase_url:
        raise SupabaseAuthError("SUPABASE_URL is not configured")
    api_keys = _api_keys()
    if not api_keys:
        raise SupabaseAuthError("SUPABASE_ANON_KEY is not configured")
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    result = None
    for index, api_key in enumerate(api_keys):
        request = urllib.request.Request(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/token?grant_type=password",
            data=payload,
            headers={"apikey": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {}
            message = body.get("msg") or body.get("error_description") or body.get("message")
            # A stale anon key should not prevent login when the server-side
            # service-role key is valid. Never retry ordinary credential errors.
            if index + 1 < len(api_keys) and exc.code in (401, 403) and _is_api_key_error(message):
                continue
            if exc.code in (400, 401) and isinstance(message, str):
                raise SupabaseAuthError(message) from exc
            raise SupabaseAuthError("Supabase Auth is unavailable") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise SupabaseAuthError("Supabase Auth is unavailable") from exc
    if result is None:
        raise SupabaseAuthError("Supabase Auth is unavailable")
    access_token = result.get("access_token") if isinstance(result, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise SupabaseAuthError("Supabase did not return a login session")
    return result


def get_user(access_token: str) -> dict:
    if not settings.supabase_url:
        raise SupabaseAuthError("SUPABASE_URL is not configured")
    api_keys = _api_keys()
    if not api_keys:
        raise SupabaseAuthError("SUPABASE_ANON_KEY is not configured")
    payload = None
    for index, api_key in enumerate(api_keys):
        request = urllib.request.Request(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"apikey": api_key, "Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if index + 1 < len(api_keys) and exc.code in (401, 403):
                continue
            raise SupabaseAuthError("Supabase session was rejected") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise SupabaseAuthError("Supabase Auth is unavailable") from exc
    if payload is None:
        raise SupabaseAuthError("Supabase Auth is unavailable")
    if not isinstance(payload, dict) or not payload.get("id") or not payload.get("email"):
        raise SupabaseAuthError("Supabase returned an incomplete user")
    return payload

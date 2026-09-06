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


def sign_in(email: str, password: str) -> dict:
    """Authenticate against Supabase without exposing the auth request to the browser."""
    if not settings.supabase_url:
        raise SupabaseAuthError("SUPABASE_URL is not configured")
    api_key = settings.supabase_anon_key or settings.supabase_service_role_key
    if not api_key:
        raise SupabaseAuthError("SUPABASE_ANON_KEY is not configured")
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.supabase_url.rstrip('/')}/auth/v1/token?grant_type=password",
        data=payload,
        headers={"apikey": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        message = body.get("msg") or body.get("error_description") or body.get("message")
        if exc.code in (400, 401) and isinstance(message, str):
            raise SupabaseAuthError(message) from exc
        raise SupabaseAuthError("Supabase Auth is unavailable") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SupabaseAuthError("Supabase Auth is unavailable") from exc
    access_token = result.get("access_token") if isinstance(result, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise SupabaseAuthError("Supabase did not return a login session")
    return result


def get_user(access_token: str) -> dict:
    if not settings.supabase_url:
        raise SupabaseAuthError("SUPABASE_URL is not configured")
    api_key = settings.supabase_anon_key or settings.supabase_service_role_key
    if not api_key:
        raise SupabaseAuthError("SUPABASE_ANON_KEY is not configured")
    request = urllib.request.Request(
        f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
        headers={"apikey": api_key, "Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SupabaseAuthError("Supabase session was rejected") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise SupabaseAuthError("Supabase Auth is unavailable") from exc
    if not isinstance(payload, dict) or not payload.get("id") or not payload.get("email"):
        raise SupabaseAuthError("Supabase returned an incomplete user")
    return payload

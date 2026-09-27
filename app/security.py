"""
Access control for the HTTP surface.

Two separate problems, two separate guards:

* **Staff endpoints** (`/orders`, `/escalations`, `/kb/*`, `/query/analyze`)
  expose customer names, phone numbers, delivery addresses and the order book.
  They were open to anyone who could reach the port. They now need a shared
  secret in `X-API-Key`.

* **Twilio webhooks** (`/voice/*`) cannot use a header - Twilio decides what it
  sends - so they are authenticated the way Twilio intends: every request is
  signed with the account's auth token, and `X-Twilio-Signature` is verified
  against the URL and the exact POST body. Without this, anyone who learns the
  public URL can inject a fake call into the pharmacy's order book.

Both are on by default. `DISABLE_AUTH=true` and `VALIDATE_TWILIO_SIGNATURE=false`
exist for a purely local demo and are refused a quiet default - turning either
off logs a warning every time the app starts.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger("meditrack.security")

API_KEY_HEADER = "X-API-Key"
TWILIO_SIGNATURE_HEADER = "X-Twilio-Signature"

# A fastapi.security class, rather than a plain Header(...) parameter, is what
# makes FastAPI register a security scheme in the OpenAPI spec - which is what
# puts the "Authorize" padlock in /docs. auto_error=False so a missing header
# reaches require_api_key() as None and gets our own 401/503 wording instead
# of FastAPI's generic one.
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def startup_warnings() -> list[str]:
    """Anything unsafe about the current configuration, for logs and /health."""
    warnings: list[str] = []
    if settings.DISABLE_AUTH:
        warnings.append(
            "DISABLE_AUTH is set: the staff endpoints are unauthenticated. "
            "Never run this way anywhere reachable from a network."
        )
    elif not settings.ADMIN_API_KEY:
        warnings.append(
            "ADMIN_API_KEY is not set: the staff endpoints will refuse every "
            "request until it is configured in .env."
        )
    if not settings.VALIDATE_TWILIO_SIGNATURE:
        warnings.append(
            "VALIDATE_TWILIO_SIGNATURE is false: the voice webhooks accept "
            "unsigned requests from any source."
        )
    elif not settings.TWILIO_AUTH_TOKEN:
        warnings.append(
            "TWILIO_AUTH_TOKEN is not set: signed voice webhooks cannot be "
            "verified, so /voice/* will refuse every request."
        )
    return warnings


def require_api_key(x_api_key: str | None = Depends(_api_key_scheme)) -> None:
    """
    FastAPI dependency guarding the staff endpoints.

    Refusing to serve when no key is configured is deliberate: an unset secret
    must fail closed, not silently open the order book to the internet.
    """
    if settings.DISABLE_AUTH:
        return

    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "This endpoint is disabled because ADMIN_API_KEY is not configured. "
                "Set it in .env (see .env.example) and restart."
            ),
        )

    # Constant-time comparison; a plain == leaks the key one byte at a time to
    # anyone who can measure the response.
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"A valid {API_KEY_HEADER} header is required.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


def _public_url_for(request: Request) -> str:
    """
    The URL Twilio actually signed.

    Twilio signs the public URL it was configured with. Behind ngrok or any
    proxy, `request.url` is the internal one (http://localhost:8000/...), and
    verifying against that fails every time - so the configured public base URL
    wins when one is set.
    """
    base = (settings.PUBLIC_BASE_URL or "").rstrip("/")
    if not base:
        return str(request.url)
    path = request.url.path
    query = request.url.query
    return f"{base}{path}" + (f"?{query}" if query else "")


async def verify_twilio_signature(request: Request) -> None:
    """
    FastAPI dependency guarding the voice webhooks.

    Reads the form body to validate it; Starlette caches the parsed form on the
    request, so the endpoint's own `Form(...)` parameters still work.
    """
    if not settings.VALIDATE_TWILIO_SIGNATURE:
        logger.warning("Accepting an unverified Twilio webhook (signature checking is off).")
        return

    if not settings.TWILIO_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Voice webhooks are disabled because TWILIO_AUTH_TOKEN is not configured. "
                "Set it in .env, or set VALIDATE_TWILIO_SIGNATURE=false for a local demo."
            ),
        )

    signature = request.headers.get(TWILIO_SIGNATURE_HEADER)
    if not signature:
        logger.warning("Rejected a voice webhook with no %s header.", TWILIO_SIGNATURE_HEADER)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing {TWILIO_SIGNATURE_HEADER}.",
        )

    from twilio.request_validator import RequestValidator

    form = await request.form()
    params = {key: str(value) for key, value in form.items()}
    validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)

    if not validator.validate(_public_url_for(request), params, signature):
        logger.warning(
            "Rejected a voice webhook with an invalid signature (url=%s).",
            _public_url_for(request),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Twilio signature.",
        )


# Convenience aliases so route definitions read as documentation.
StaffAuth = Depends(require_api_key)
TwilioAuth = Depends(verify_twilio_signature)

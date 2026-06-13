"""
Refresh-token cookie helpers.

The refresh token is the only credential that lives long enough to be worth
stealing, so we put it in an httpOnly cookie (browsers won't expose it to
JavaScript -> immune to XSS token theft). The short-lived access token stays
in the response body and the frontend keeps it in memory.

Cookie flags are driven by env vars set in animal/settings.py so dev and
prod can use the same code path with different SameSite/Secure settings.
"""

from django.conf import settings
from rest_framework_simplejwt.tokens import RefreshToken


def _cookie_kwargs(max_age: int):
    return {
        "key": settings.REFRESH_COOKIE_NAME,
        "max_age": max_age,
        "httponly": True,
        "secure": settings.REFRESH_COOKIE_SECURE,
        "samesite": settings.REFRESH_COOKIE_SAMESITE,
        "domain": settings.REFRESH_COOKIE_DOMAIN,
        "path": settings.REFRESH_COOKIE_PATH,
    }


def set_refresh_cookie(response, refresh_token: str):
    """Attach the refresh token as an httpOnly cookie on `response`."""
    max_age = int(settings.SIMPLE_JWT["REFRESH_TOKEN_LIFETIME"].total_seconds())
    response.set_cookie(value=refresh_token, **_cookie_kwargs(max_age))
    return response


def clear_refresh_cookie(response):
    """Expire the refresh cookie. Match the same path/domain that set it."""
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=settings.REFRESH_COOKIE_PATH,
        domain=settings.REFRESH_COOKIE_DOMAIN,
        samesite=settings.REFRESH_COOKIE_SAMESITE,
    )
    return response


def issue_tokens_for(user):
    """Mint a (refresh, access) pair for the given user."""
    refresh = RefreshToken.for_user(user)
    return str(refresh), str(refresh.access_token)

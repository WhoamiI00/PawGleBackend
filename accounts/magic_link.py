"""
Magic-link sign-in: passwordless login over a signed, time-limited URL.

Used in two scenarios:
  1. A non-registered finder uses a public flow (e.g. QR-tag scan) and
     leaves their email; we create an inactive User behind the scenes and
     mail them a magic link that activates the account when clicked.
  2. A registered user who's forgotten their password (alternative to
     reset-password) can request a magic link to get back in.

Tokens are produced with django.core.signing.TimestampSigner so we don't
have to persist anything in the DB - the signer's secret + the token's
own timestamp are enough to verify it.
"""

import logging
from typing import Tuple

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import signing
from django.core.mail import send_mail
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .cookie_auth import issue_tokens_for, set_refresh_cookie

logger = logging.getLogger(__name__)

# 24 hours - balance "user might check email later" with "stolen token blast radius".
MAGIC_LINK_MAX_AGE_SECONDS = 24 * 60 * 60

# Namespace so a token forged for some other Django signer can't be reused here.
SIGNER_SALT = 'pawgle.magic_link'


def _signer() -> signing.TimestampSigner:
    return signing.TimestampSigner(salt=SIGNER_SALT)


def mint_token(user_id: int) -> str:
    """Sign a user_id into a short URL-safe token."""
    return _signer().sign(str(user_id))


def verify_token(token: str) -> int:
    """Validate a token and return the embedded user_id. Raises BadSignature/SignatureExpired."""
    raw = _signer().unsign(token, max_age=MAGIC_LINK_MAX_AGE_SECONDS)
    return int(raw)


def get_or_create_guest_user(email: str, display_name: str = '') -> Tuple['User', bool]:
    """Find a user by email or create an inactive 'guest' shell.

    Inactive accounts can still receive notifications and own conversations;
    the magic-link redemption is what flips is_active to True.
    Returns (user, created).
    """
    User = get_user_model()
    normalized = email.strip().lower()

    user = User.objects.filter(email__iexact=normalized).first()
    if user is not None:
        return user, False

    base_username = (display_name or normalized.split('@')[0])[:140] or 'guest'
    username = base_username
    suffix = 1
    while User.objects.filter(username=username).exists():
        suffix += 1
        username = f"{base_username}{suffix}"[:150]

    user = User.objects.create(
        username=username,
        email=normalized,
        is_active=False,
    )
    user.set_unusable_password()
    user.save()
    return user, True


def send_magic_link(user, *, intent: str = 'sign-in') -> None:
    """Mail a magic-link URL pointing at the frontend. Link-only - no PII."""
    token = mint_token(user.id)
    base = (getattr(settings, "SITE_URL", "") or "https://pawgle.neokit.app").rstrip("/")
    link = f"{base}/auth/magic?token={token}"

    try:
        send_mail(
            subject=f"PawGle: your sign-in link",
            message=(
                f"Hi,\n\n"
                f"Click the link below to {intent} on PawGle. "
                f"It expires in 24 hours and can be used once.\n\n"
                f"{link}\n\n"
                f"If you didn't request this, you can ignore this email."
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=True,
        )
    except Exception as e:
        logger.error(f"Failed to send magic link to {user.email}: {e}")


# ----- Endpoints -----

class RequestMagicLinkView(APIView):
    """POST /api/auth/magic/request/ {email}

    Always returns 200 (don't leak whether the address is registered) and
    fires off a magic-link email behind the scenes. Heavily rate-limited
    via DRF throttles to make it useless for enumeration.
    """
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_scope = 'password_reset'

    def post(self, request):
        email = (request.data.get('email') or '').strip()
        if not email or '@' not in email:
            return Response(
                {"detail": "A valid email address is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        User = get_user_model()
        user = User.objects.filter(email__iexact=email).first()
        if user is not None:
            send_magic_link(user, intent='sign in')

        # Generic response regardless of whether the user exists.
        return Response(
            {"detail": "If an account with that email exists, a sign-in link is on its way."},
            status=status.HTTP_200_OK,
        )


class ConsumeMagicLinkView(APIView):
    """POST /api/auth/magic/consume/ {token}

    Validates the signed token. On success: activates the user if needed,
    returns an access token in the body, and sets the refresh cookie.
    """
    permission_classes = [AllowAny]
    authentication_classes = []
    throttle_scope = 'login'

    def post(self, request):
        token = (request.data.get('token') or '').strip()
        if not token:
            return Response(
                {"detail": "Missing token."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user_id = verify_token(token)
        except signing.SignatureExpired:
            return Response(
                {"detail": "This link has expired. Request a new one."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except signing.BadSignature:
            return Response(
                {"detail": "Invalid or tampered link."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        User = get_user_model()
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response(
                {"detail": "Account no longer exists."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # First successful click activates a guest account.
        if not user.is_active:
            user.is_active = True
            user.save(update_fields=['is_active'])

        refresh, access = issue_tokens_for(user)
        response = Response(
            {
                "access": access,
                "user": {"id": user.id, "email": user.email, "username": user.username},
            },
            status=status.HTTP_200_OK,
        )
        return set_refresh_cookie(response, refresh)

"""
QR-code pet tags + public found-pet landing endpoint.

A registered owner can print/download a QR for their pet that links to
SITE_URL/found/<animal_id>. A stranger who finds the pet scans the QR,
the frontend hits the public landing endpoint (no auth needed) to show
pet basics + an action to contact the owner via chat / report a sighting.
"""

import io
import logging

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Pet

logger = logging.getLogger(__name__)


def _public_pet_url(pet: Pet) -> str:
    """The URL printed on the QR. Frontend handles whatever lands there."""
    base = (getattr(settings, "SITE_URL", "") or "https://pawgle.neokit.app").rstrip("/")
    return f"{base}/found/{pet.animal_id}"


class PetQRTagView(APIView):
    """GET /api/auth/pets/<pet_id>/qr/  -> PNG with the pet's tag QR.

    Owner-only so we don't leak QRs for someone else's pet.
    """
    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = 'user'

    def get(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        try:
            import qrcode
        except ImportError:
            logger.error("qrcode package missing")
            return Response({'detail': 'QR generation not available'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        img = qrcode.make(_public_pet_url(pet))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        response = HttpResponse(buf.getvalue(), content_type='image/png')
        response['Content-Disposition'] = f'attachment; filename="pawgle-{pet.animal_id}.png"'
        # Browsers cache aggressively; we don't want stale tags after a re-deploy.
        response['Cache-Control'] = 'private, max-age=3600'
        return response


class PublicPetByAnimalIdView(APIView):
    """GET /api/auth/found/<animal_id>/  -> minimal pet info for the finder.

    NO AUTH required - this is the page a stranger reaches by scanning the tag.
    Only safe-to-disclose fields go in the response; owner contact stays private
    and the finder is expected to use the in-app chat / report flow.
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_scope = 'anon'

    def get(self, request, animal_id):
        try:
            pet = Pet.objects.select_related('owner').get(animal_id=animal_id)
        except Pet.DoesNotExist:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            'animal_id': pet.animal_id,
            'name': pet.name,
            'type': pet.type,
            'breed': pet.breed,
            'category': pet.category,
            'images': pet.images or [],
            'has_owner': bool(pet.owner_id),
            # Owner contact is intentionally OMITTED - finder is funneled to
            # the chat/report flow so we can mediate (and the owner can choose
            # to share contact info in-app).
        })

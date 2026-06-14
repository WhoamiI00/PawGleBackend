"""
Geographic radius alerts: a user subscribes to "tell me about pets in this
area", and we notify them via in-app Notification + email when a matching
PetLocation report comes in.

The match logic is intentionally simple: cheap latitude-band filter first
(degrees → km is ~111 km/deg), then haversine refinement in Python. At our
scale that beats any DB-side spatial index complexity.
"""

import logging
import math
from typing import List

from django.conf import settings
from django.core.mail import send_mail
from django.utils import timezone
from rest_framework import generics, permissions, serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AlertSubscription, Notification, PetLocation

logger = logging.getLogger(__name__)


# -------- haversine --------

EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# -------- dispatcher --------

def find_matching_subscriptions(location: PetLocation) -> List[AlertSubscription]:
    """Return active subscriptions whose radius covers this location."""
    # ~degrees in 1 km. We over-include with the max possible radius (1000km)
    # if a subscription has crazy values, then haversine prunes.
    DEG_PER_KM = 1.0 / 111.0

    qs = AlertSubscription.objects.filter(is_active=True)
    if location.status in ('lost', 'found'):
        # status_filter == '' matches both; otherwise must match.
        qs = qs.filter(models_q_status(location.status))

    # We need to enumerate to apply the per-row radius. Bounding-box pre-filter
    # would help when the subscription table grows; skip for now.
    matches = []
    for sub in qs.select_related('user').iterator(chunk_size=200):
        # Quick reject by lat band.
        if abs(sub.center_lat - location.latitude) > sub.radius_km * DEG_PER_KM:
            continue
        dist = _haversine_km(
            sub.center_lat, sub.center_lon,
            location.latitude, location.longitude,
        )
        if dist <= sub.radius_km:
            matches.append(sub)
    return matches


def models_q_status(status_value: str):
    """status_filter='' OR matches the report status."""
    from django.db.models import Q
    return Q(status_filter='') | Q(status_filter=status_value)


def notify_subscribers(location_id: int) -> int:
    """Send notifications + a paging email for each matching subscription.

    Returns the count of subscriptions matched. Safe to call multiple times -
    Notification rows are append-only by design (the user gets duplicate
    reminders if a report is updated, which we accept for now).
    """
    try:
        location = PetLocation.objects.select_related('pet').get(id=location_id)
    except PetLocation.DoesNotExist:
        return 0

    if location.status not in ('lost', 'found'):
        return 0

    subs = find_matching_subscriptions(location)
    if not subs:
        return 0

    frontend = (getattr(settings, "SITE_URL", "") or "https://pawgle.neokit.app").rstrip("/")
    link = f"{frontend}/pet/map"

    pet_name = (location.pet.name if location.pet else None) or location.pet_name or "a pet"
    verb_label = "lost" if location.status == "lost" else "found"

    for sub in subs:
        # Don't ping reporter about their own report.
        if location.pet and location.pet.owner_id == sub.user_id:
            continue

        try:
            Notification.objects.create(
                recipient_id=sub.user_id,
                verb='nearby_alert',
                description=(
                    f"A {verb_label} pet ({pet_name}) was reported within "
                    f"{sub.radius_km:.0f} km of your '{sub.label or 'saved area'}'."
                ),
                target=location.pet if location.pet else None,
            )
        except Exception as e:
            logger.error(f"Failed to create nearby_alert Notification for sub {sub.id}: {e}")

        if sub.user.email:
            try:
                send_mail(
                    subject=f"PawGle: {verb_label} pet near your area",
                    message=(
                        f"A {verb_label} pet was reported in your alert zone "
                        f"({sub.label or 'saved area'}).\n\n"
                        f"See the map: {link}\n\n"
                        f"You can manage your alerts in PawGle settings."
                    ),
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[sub.user.email],
                    fail_silently=True,
                )
            except Exception as e:
                logger.error(f"Failed to send nearby_alert email for sub {sub.id}: {e}")

    return len(subs)


# -------- API -----------

class AlertSubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertSubscription
        fields = [
            'id', 'label', 'center_lat', 'center_lon', 'radius_km',
            'status_filter', 'is_active', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class AlertSubscriptionListView(generics.ListCreateAPIView):
    """GET + POST /api/auth/alerts/"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AlertSubscriptionSerializer
    throttle_scope = 'user'

    def get_queryset(self):
        return AlertSubscription.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        if self.get_queryset().count() >= 10:
            raise serializers.ValidationError(
                "You can have at most 10 alert subscriptions."
            )
        serializer.save(user=self.request.user)


class AlertSubscriptionDetailView(generics.RetrieveUpdateDestroyAPIView):
    """GET/PATCH/DELETE /api/auth/alerts/<id>/"""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = AlertSubscriptionSerializer
    throttle_scope = 'user'

    def get_queryset(self):
        return AlertSubscription.objects.filter(user=self.request.user)

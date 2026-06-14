"""
Auto-match: when a new PetLocation report gets an embedding, search Qdrant
for likely matches on the opposite side (lost <-> found) and:

  1. Record each high-confidence pairing in PetMatch (deduped by unique key).
  2. Open a Conversation between the two sides if none exists.
  3. Send a single "you have a possible match" email to the owner side
     - link only, no message body (decision Q2).

Notification is "best-effort": failures are logged and do not break the
extraction task that called us.
"""

import logging
import math
from typing import Iterable

from django.conf import settings
from django.db import IntegrityError
from django.utils import timezone

from . import qdrant_index
from .models import (
    Conversation,
    Notification,
    Pet,
    PetLocation,
    PetMatch,
)

logger = logging.getLogger(__name__)


# -------- distance --------

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) pairs, in meters."""
    R = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -------- candidate retrieval --------

def _flatten_vector(features):
    """Same convention as qdrant_index: features may be [v0...] or [[v0...]]."""
    if not features:
        return None
    first = features[0] if isinstance(features, list) else None
    if isinstance(first, list):
        return first
    return list(features)


def _search_opposite(location: PetLocation):
    """Given a new report, find candidate matches on the opposite side.

    A 'lost' report searches the registered-pets collection AND any prior
    'found' reports. A 'found' report searches registered pets AND any
    prior 'lost' reports. We search both Qdrant collections and merge.

    Returns a list of dicts: {kind, payload, score} sorted by score desc.
    """
    vector = _flatten_vector(location.features)
    if not vector:
        return []

    geo_filter = qdrant_index.geo_filter_near(
        location.latitude, location.longitude,
        settings.AUTOMATCH_MAX_DISTANCE_KM * 1000,
    )

    candidates = []

    # Registered pets - always interesting regardless of which side fired.
    pet_hits = qdrant_index.search(
        vector,
        collection=settings.QDRANT_PETS_COLLECTION,
        limit=settings.AUTOMATCH_CANDIDATE_LIMIT,
        score_threshold=settings.AUTOMATCH_MIN_SIMILARITY,
    )
    for hit in pet_hits:
        candidates.append({"kind": "pet", "payload": hit.payload or {}, "score": float(hit.score)})

    # Reports on the OPPOSITE side, geo-filtered to keep noise down.
    opposite_status = "found" if location.status == "lost" else "lost"
    report_hits = qdrant_index.search(
        vector,
        collection=settings.QDRANT_REPORTS_COLLECTION,
        limit=settings.AUTOMATCH_CANDIDATE_LIMIT,
        score_threshold=settings.AUTOMATCH_MIN_SIMILARITY,
        extra_filter=_status_filter(opposite_status, geo_filter),
    )
    for hit in report_hits:
        # Don't match a report against itself if it ends up in its own collection.
        if (hit.payload or {}).get("location_id") == location.id:
            continue
        candidates.append({"kind": "report", "payload": hit.payload or {}, "score": float(hit.score)})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[: settings.AUTOMATCH_CANDIDATE_LIMIT]


def _status_filter(status: str, geo_filter):
    """Combine a status equality filter with the geo filter."""
    try:
        from qdrant_client.http import models as qmodels
        must = [qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value=status))]
        if geo_filter is not None:
            must.extend(geo_filter.must or [])
        return qmodels.Filter(must=must)
    except Exception:
        return geo_filter


# -------- match persistence + notify --------

def _ensure_match_row(*, lost_pet, lost_report, found_report, similarity, distance_m):
    """Upsert a PetMatch. Returns (match, created)."""
    lookup = {"found_report": found_report}
    if lost_pet:
        lookup["lost_pet"] = lost_pet
    else:
        lookup["lost_report"] = lost_report

    existing = PetMatch.objects.filter(**lookup).first()
    if existing:
        return existing, False

    try:
        match = PetMatch.objects.create(
            lost_pet=lost_pet,
            lost_report=lost_report,
            found_report=found_report,
            similarity=similarity,
            distance_meters=distance_m,
        )
        return match, True
    except IntegrityError:
        # Concurrent worker beat us to it - just fetch the row.
        return PetMatch.objects.filter(**lookup).first(), False


def _open_conversation_for(match: PetMatch):
    """Get or create the Conversation tied to this match's found_report.

    For now: one chat thread per (found_report, owner-side user). We attach
    the lost-pet owner (or the lost-report reporter, by email lookup) as a
    participant.
    """
    found = match.found_report
    convo = Conversation.objects.filter(
        pet_location=found,
        participants__isnull=False,
    ).first()
    if convo:
        return convo

    convo = Conversation.objects.create(pet_location=found)

    if match.lost_pet and match.lost_pet.owner_id:
        convo.participants.add(match.lost_pet.owner_id)
    elif match.lost_report and match.lost_report.contact_email:
        # Lost-report from a non-registered reporter. Skip for now -
        # guest user creation lives in the next slice of work.
        convo.reporter_email = match.lost_report.contact_email
        convo.reporter_name = match.lost_report.contact_name or "Reporter"
        convo.save(update_fields=['reporter_email', 'reporter_name'])

    return convo


def _send_match_email(match: PetMatch, conversation: Conversation):
    """Single email, link only - no message body, no pet details beyond name."""
    from django.core.mail import send_mail

    if match.notified_at is not None:
        return

    recipient = None
    pet_name = "your pet"
    if match.lost_pet:
        recipient = match.lost_pet.owner.email if match.lost_pet.owner else None
        pet_name = match.lost_pet.name
    elif match.lost_report and match.lost_report.contact_email:
        recipient = match.lost_report.contact_email
        pet_name = match.lost_report.pet_name or pet_name

    if not recipient:
        return

    frontend = getattr(settings, "SITE_URL", "").rstrip("/") or "https://pawgle.neokit.app"
    link = f"{frontend}/chat/{conversation.id}"

    try:
        send_mail(
            subject=f"PawGle: possible match for {pet_name}",
            message=(
                f"Someone may have found {pet_name}.\n\n"
                f"Open the conversation to see the report and reply:\n{link}\n\n"
                "We sent you this email because PawGle's image match flagged a "
                "high-confidence similarity. All further messages stay inside the app."
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            fail_silently=True,
        )
        match.notified_at = timezone.now()
        match.save(update_fields=['notified_at'])
    except Exception as e:
        logger.error(f"Failed to send match email for PetMatch {match.id}: {e}")


def _create_app_notification(match: PetMatch):
    """In-app notification so the owner sees the match in their bell menu."""
    if not match.lost_pet or not match.lost_pet.owner_id:
        return
    Notification.objects.create(
        recipient_id=match.lost_pet.owner_id,
        verb='auto_match',
        description=(
            f"We found a possible match for {match.lost_pet.name}. "
            f"Open the chat to view the report."
        ),
        target=match.lost_pet,
    )


# -------- public entrypoint --------

def run_for_location(location_id: int) -> int:
    """Find matches for the just-indexed PetLocation. Returns number of new matches."""
    try:
        location = (
            PetLocation.objects
            .select_related('pet__owner')
            .get(id=location_id)
        )
    except PetLocation.DoesNotExist:
        return 0

    if location.feature_status != 'completed' or not location.features:
        return 0

    candidates = _search_opposite(location)
    if not candidates:
        return 0

    created_count = 0

    for cand in candidates:
        kind = cand["kind"]
        payload = cand["payload"]
        score = cand["score"]

        # Decide which side of the match we're building.
        # `found_report` always holds a PetLocation; `lost_pet` or
        # `lost_report` is the counterparty.
        if location.status == 'found':
            found_report = location
            lost_pet = None
            lost_report = None
            if kind == "pet":
                lost_pet = Pet.objects.filter(id=payload.get("pet_id")).first()
            else:
                lost_report = PetLocation.objects.filter(id=payload.get("location_id")).first()
        else:
            # New report is lost; the other side is a found report (or a registered pet that's
            # been spotted somewhere). Treat the candidate as the "found" side.
            if kind == "pet":
                # Candidate is a registered pet - unusual: we found a "lost" report
                # matching a registered pet's own embedding. That's the owner's own pet,
                # skip it.
                continue
            else:
                lost_pet = location.pet if location.pet else None
                lost_report = None if location.pet else location
                found_report = PetLocation.objects.filter(id=payload.get("location_id")).first()
                if not found_report:
                    continue

        # Distance between the two reports (skip if we lack coords).
        distance_m = None
        if hasattr(found_report, 'latitude') and lost_pet is None and lost_report:
            distance_m = _haversine_meters(
                lost_report.latitude, lost_report.longitude,
                found_report.latitude, found_report.longitude,
            )

        match, created = _ensure_match_row(
            lost_pet=lost_pet,
            lost_report=lost_report,
            found_report=found_report,
            similarity=score,
            distance_m=distance_m,
        )
        if not match or not created:
            continue

        created_count += 1
        try:
            convo = _open_conversation_for(match)
            _create_app_notification(match)
            _send_match_email(match, convo)
        except Exception as e:
            logger.error(f"Match-followup failed for PetMatch {match.id}: {e}")

    return created_count

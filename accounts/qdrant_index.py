"""
Qdrant integration for vector similarity search.

We keep two collections:
  - QDRANT_PETS_COLLECTION    -> embeddings of registered Pet records
  - QDRANT_REPORTS_COLLECTION -> embeddings of PetLocation (lost / found reports)

Source of truth for everything else (name, owner, images, etc.) stays in TiDB.
Qdrant only stores the vector + small payload used for filtering during search.

The client is created lazily on first use so Django can start even when
Qdrant is misconfigured. All public helpers swallow exceptions and log them
- they should never break the request/task that called them.
"""

import logging
import time
import uuid
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)

_client = None
_bootstrapped = False

# Small retry budget — Qdrant Cloud occasionally returns DNS / 5xx blips,
# especially on free tiers. Keep total wait under ~3s so we don't stall requests.
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (0.3, 0.7, 1.5)


def _retry(fn, *, op: str):
    """Run `fn()` with a tiny exponential backoff. Returns fn() result or None on failure."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt + 1 < _MAX_RETRIES:
                time.sleep(_BACKOFF_SECONDS[attempt])
    logger.error(f"Qdrant {op} failed after {_MAX_RETRIES} attempts: {last_err}")
    return None


def _get_client():
    global _client
    if _client is not None:
        return _client

    endpoint = (settings.QDRANT_CLUSTER_ENDPOINT or "").strip()
    api_key = (settings.QDRANT_API_KEY or "").strip()

    if not endpoint or not api_key:
        logger.warning("Qdrant not configured (missing endpoint or api key)")
        return None

    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        logger.error(
            f"QDRANT_CLUSTER_ENDPOINT must start with https:// — got {endpoint!r}"
        )
        return None

    try:
        from qdrant_client import QdrantClient
        _client = QdrantClient(
            url=endpoint,
            api_key=api_key,
            timeout=15,
            prefer_grpc=False,
            check_compatibility=False,
        )
        return _client
    except Exception as e:
        logger.error(f"Failed to construct Qdrant client: {e}")
        return None


def _ensure_collections():
    """Create the two collections on first use if they don't exist.

    Important: only flips `_bootstrapped` after a successful round-trip, so a
    transient network blip during the first call doesn't poison the cache.
    """
    global _bootstrapped
    if _bootstrapped:
        return

    client = _get_client()
    if client is None:
        return

    from qdrant_client.http import models as qmodels

    def _do():
        existing = {c.name for c in client.get_collections().collections}
        for name in (settings.QDRANT_PETS_COLLECTION, settings.QDRANT_REPORTS_COLLECTION):
            if name not in existing:
                client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=settings.QDRANT_VECTOR_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                client.create_payload_index(name, "owner_id", qmodels.PayloadSchemaType.INTEGER)
                client.create_payload_index(name, "kind", qmodels.PayloadSchemaType.KEYWORD)
                if name == settings.QDRANT_REPORTS_COLLECTION:
                    client.create_payload_index(name, "status", qmodels.PayloadSchemaType.KEYWORD)
                    client.create_payload_index(name, "location", qmodels.PayloadSchemaType.GEO)
        return True

    if _retry(_do, op="collection bootstrap") is True:
        _bootstrapped = True


def _point_id(prefix: str, raw_id) -> str:
    """Deterministic UUID5 so the same Pet/PetLocation always maps to the same point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pawgle:{prefix}:{raw_id}"))


def _flatten_vector(vector) -> Optional[list]:
    """The model stores features as either [v0, v1, ...] or [[v0, v1, ...]]."""
    if not vector:
        return None
    first = vector[0] if isinstance(vector, list) else None
    if isinstance(first, list):
        return first
    return list(vector)


# -------- Pets --------

def upsert_pet(pet) -> bool:
    _ensure_collections()
    client = _get_client()
    if client is None:
        return False

    vector = _flatten_vector(pet.features)
    if not vector:
        return False

    from qdrant_client.http import models as qmodels
    point = qmodels.PointStruct(
        id=_point_id("pet", pet.id),
        vector=vector,
        payload={
            "kind": "pet",
            "pet_id": pet.id,
            "animal_id": pet.animal_id,
            "owner_id": pet.owner_id,
            "type": pet.type,
            "breed": pet.breed,
            "category": pet.category,
            "is_public": pet.isPublic,
        },
    )
    result = _retry(
        lambda: client.upsert(collection_name=settings.QDRANT_PETS_COLLECTION, points=[point]),
        op=f"upsert_pet({pet.id})",
    )
    return result is not None


def delete_pet(pet_id: int) -> bool:
    client = _get_client()
    if client is None:
        return False
    from qdrant_client.http import models as qmodels
    selector = qmodels.PointIdsList(points=[_point_id("pet", pet_id)])
    result = _retry(
        lambda: client.delete(collection_name=settings.QDRANT_PETS_COLLECTION, points_selector=selector),
        op=f"delete_pet({pet_id})",
    )
    return result is not None


# -------- Reports (PetLocation) --------

def upsert_report(location) -> bool:
    _ensure_collections()
    client = _get_client()
    if client is None:
        return False

    vector = _flatten_vector(location.features)
    if not vector:
        return False

    from qdrant_client.http import models as qmodels
    payload = {
        "kind": "report",
        "location_id": location.id,
        "pet_id": location.pet_id,
        "owner_id": location.pet.owner_id if location.pet_id and location.pet else None,
        "status": location.status,
        "pet_type": location.pet_type or (location.pet.type if location.pet else ""),
        "pet_breed": location.pet_breed or (location.pet.breed if location.pet else ""),
        "location": {"lat": location.latitude, "lon": location.longitude},
    }
    point = qmodels.PointStruct(
        id=_point_id("report", location.id),
        vector=vector,
        payload=payload,
    )
    result = _retry(
        lambda: client.upsert(collection_name=settings.QDRANT_REPORTS_COLLECTION, points=[point]),
        op=f"upsert_report({location.id})",
    )
    return result is not None


def delete_report(location_id: int) -> bool:
    client = _get_client()
    if client is None:
        return False
    from qdrant_client.http import models as qmodels
    selector = qmodels.PointIdsList(points=[_point_id("report", location_id)])
    result = _retry(
        lambda: client.delete(collection_name=settings.QDRANT_REPORTS_COLLECTION, points_selector=selector),
        op=f"delete_report({location_id})",
    )
    return result is not None


# -------- Search --------

def search(
    query_vector: list,
    *,
    collection: str,
    limit: int = 20,
    score_threshold: Optional[float] = None,
    extra_filter=None,
):
    """Returns a list of ScoredPoint-like objects (with .id, .score, .payload),
    or [] if Qdrant is unavailable.

    qdrant-client removed the legacy `client.search()` method in 1.13+ in
    favor of `query_points()`, which returns a wrapped QueryResponse. We
    unwrap `.points` so callers keep the same shape they had before.
    """
    _ensure_collections()
    client = _get_client()
    if client is None or not query_vector:
        return []

    def _do_query():
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=extra_filter,
            with_payload=True,
        )
        # query_points returns QueryResponse(points=[...]); legacy .search
        # returned the list directly. Normalize to the list shape so all
        # call sites stay simple.
        return list(getattr(response, 'points', response) or [])

    result = _retry(_do_query, op=f"search({collection})")
    return result or []


def geo_filter_near(lat: float, lon: float, radius_meters: float):
    """Build a Qdrant filter that restricts results to a geo radius."""
    try:
        from qdrant_client.http import models as qmodels
        return qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="location",
                    geo_radius=qmodels.GeoRadius(
                        center=qmodels.GeoPoint(lat=lat, lon=lon),
                        radius=radius_meters,
                    ),
                )
            ]
        )
    except Exception:
        return None

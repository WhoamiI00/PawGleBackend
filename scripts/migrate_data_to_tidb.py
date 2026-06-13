"""
One-shot data migration: Supabase Postgres -> TiDB.

Reads rows from the Supabase Postgres database using psycopg2,
writes them into TiDB via Django ORM. Idempotent at the user level
(uses get_or_create on username/email); pets/locations/etc. use
their original primary keys, so re-running will raise IntegrityError
on collisions -- truncate target tables first if you need a redo.

Run from the backend directory:
    python scripts/migrate_data_to_tidb.py
"""

import os
import sys
import json
import django
from pathlib import Path

# Bootstrap Django pointed at TiDB
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "animal.settings")
django.setup()

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")

from django.contrib.auth import get_user_model  # noqa: E402
from django.db import transaction  # noqa: E402
from accounts.models import (  # noqa: E402
    Pet,
    PetLocation,
    Notification,
    Conversation,
    EditedPetImage,
)

User = get_user_model()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    print("SUPABASE_DB_URL not set; aborting.")
    sys.exit(1)


def pg_connect():
    return psycopg2.connect(SUPABASE_DB_URL)


def fetch_all(cur, table):
    cur.execute(f'SELECT * FROM "{table}"')
    return cur.fetchall()


def as_json(value, default):
    """Postgres JSON columns come back as dict/list already; coerce defensively."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def migrate_users(pg_cur):
    rows = fetch_all(pg_cur, "auth_user")
    print(f"  users in source: {len(rows)}")
    id_map = {}
    for r in rows:
        user, created = User.objects.get_or_create(
            username=r["username"],
            defaults={
                "email": r["email"] or "",
                "first_name": r["first_name"] or "",
                "last_name": r["last_name"] or "",
                "is_active": r["is_active"],
                "is_staff": r["is_staff"],
                "is_superuser": r["is_superuser"],
                "date_joined": r["date_joined"],
                "last_login": r["last_login"],
                "password": r["password"],  # hashed
            },
        )
        id_map[r["id"]] = user.id
    print(f"  users in target: {User.objects.count()}")
    return id_map


def migrate_pets(pg_cur, user_id_map):
    rows = fetch_all(pg_cur, "accounts_pet")
    print(f"  pets in source: {len(rows)}")
    for r in rows:
        owner_id = user_id_map.get(r["owner_id"])
        if owner_id is None:
            print(f"    SKIP pet {r['id']}: owner {r['owner_id']} not migrated")
            continue
        Pet.objects.update_or_create(
            id=r["id"],
            defaults={
                "name": r["name"],
                "type": r["type"],
                "category": r["category"],
                "breed": r["breed"],
                "additionalInfo": as_json(r.get("additionalInfo"), {}),
                "images": as_json(r.get("images"), []),
                "features": as_json(r.get("features"), []),
                "feature_status": r.get("feature_status") or "pending",
                "isPublic": r.get("isPublic", False),
                "owner_id": owner_id,
                "animal_id": r["animal_id"],
                "registered_at": r["registered_at"],
            },
        )
    print(f"  pets in target: {Pet.objects.count()}")


def migrate_pet_locations(pg_cur, user_id_map):
    rows = fetch_all(pg_cur, "accounts_petlocation")
    print(f"  pet_locations in source: {len(rows)}")
    for r in rows:
        # image field stores the storage key path -- preserve as-is; image migration script
        # rewrites the underlying bytes to R2 with the same key.
        PetLocation.objects.update_or_create(
            id=r["id"],
            defaults={
                "pet_id": r["pet_id"],
                "pet_name": r.get("pet_name") or "",
                "pet_type": r.get("pet_type") or "",
                "pet_breed": r.get("pet_breed") or "",
                "pet_description": r.get("pet_description") or "",
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "status": r["status"],
                "description": r.get("description") or "",
                "reported_at": r["reported_at"],
                "resolved_at": r.get("resolved_at"),
                "contact_name": r.get("contact_name") or "",
                "contact_phone": r.get("contact_phone") or "",
                "contact_email": r.get("contact_email") or "",
                "last_seen_date": r.get("last_seen_date"),
                "last_seen_time": r.get("last_seen_time"),
                "image": r.get("image") or "",
                "features": as_json(r.get("features"), []),
                "feature_status": r.get("feature_status") or "pending",
                "is_user_location": r.get("is_user_location", False),
            },
        )
    print(f"  pet_locations in target: {PetLocation.objects.count()}")


def migrate_notifications(pg_cur, user_id_map):
    rows = fetch_all(pg_cur, "accounts_notification")
    print(f"  notifications in source: {len(rows)}")
    for r in rows:
        recipient_id = user_id_map.get(r["recipient_id"])
        if recipient_id is None:
            continue
        Notification.objects.update_or_create(
            id=r["id"],
            defaults={
                "recipient_id": recipient_id,
                "verb": r["verb"],
                "description": r.get("description") or "",
                "target_id": r.get("target_id"),
                "is_read": r.get("is_read", False),
                "created_at": r["created_at"],
            },
        )
    print(f"  notifications in target: {Notification.objects.count()}")


def migrate_conversations(pg_cur):
    rows = fetch_all(pg_cur, "accounts_conversation")
    print(f"  conversations in source: {len(rows)}")
    for r in rows:
        Conversation.objects.update_or_create(
            id=r["id"],
            defaults={
                "pet_location_id": r["pet_location_id"],
                "reporter_email": r["reporter_email"],
                "reporter_name": r["reporter_name"],
                "owner_share_info": r.get("owner_share_info", False),
                "reporter_share_info": r.get("reporter_share_info", False),
                "created_at": r["created_at"],
            },
        )
    print(f"  conversations in target: {Conversation.objects.count()}")


def migrate_edited_pet_images(pg_cur, user_id_map):
    rows = fetch_all(pg_cur, "accounts_editedpetimage")
    print(f"  edited_pet_images in source: {len(rows)}")
    for r in rows:
        owner_id = user_id_map.get(r["owner_id"])
        if owner_id is None:
            continue
        EditedPetImage.objects.update_or_create(
            id=r["id"],
            defaults={
                "edited_image": r.get("edited_image") or "",
                "edit_metadata": as_json(r.get("edit_metadata"), {}),
                "created_at": r["created_at"],
                "owner_id": owner_id,
            },
        )
    print(f"  edited_pet_images in target: {EditedPetImage.objects.count()}")


def main():
    print("Connecting to Supabase Postgres...")
    pg_conn = pg_connect()
    pg_conn.set_session(readonly=True, autocommit=True)
    pg_cur = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    print("Phase 1: users")
    user_id_map = migrate_users(pg_cur)

    print("Phase 2: pets")
    with transaction.atomic():
        migrate_pets(pg_cur, user_id_map)

    print("Phase 3: pet_locations")
    with transaction.atomic():
        migrate_pet_locations(pg_cur, user_id_map)

    print("Phase 4: notifications")
    with transaction.atomic():
        migrate_notifications(pg_cur, user_id_map)

    print("Phase 5: conversations")
    with transaction.atomic():
        migrate_conversations(pg_cur)

    print("Phase 6: edited_pet_images")
    with transaction.atomic():
        migrate_edited_pet_images(pg_cur, user_id_map)

    pg_cur.close()
    pg_conn.close()
    print("Done.")


if __name__ == "__main__":
    main()

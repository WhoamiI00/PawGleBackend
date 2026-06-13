"""
One-shot image migration: Supabase Storage -> Cloudflare R2.

For every image referenced by Pet.images / PetLocation.image / EditedPetImage.edited_image:
  1. Resolve the storage key (e.g. 'pets/<uuid>.jpg').
  2. Download the bytes from Supabase public storage.
  3. Upload to R2 at the same key.
  4. For Pet.images (full URLs in a JSON list), rewrite the URL in-place to the new R2 URL.
     For PetLocation.image / EditedPetImage.edited_image (storage keys), no rewrite is needed
     -- the storage backend now resolves keys against R2.

Idempotent: if the R2 object already exists, the upload step is skipped. URL rewriting checks
for the Supabase host prefix and skips already-rewritten rows.

Run:
    python scripts/migrate_images_to_r2.py
"""

import os
import sys
import django
from pathlib import Path
from urllib.parse import urlparse

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "animal.settings")
django.setup()

import requests  # noqa: E402
import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

from django.conf import settings  # noqa: E402
from accounts.models import Pet, PetLocation, EditedPetImage  # noqa: E402

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET_NAME", "images")
SUPABASE_PUBLIC_PREFIX = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/"
# Authenticated path (service role bypasses public-egress throttling)
SUPABASE_AUTH_PREFIX = f"{SUPABASE_URL}/storage/v1/object/authenticated/{SUPABASE_BUCKET}/"
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

s3 = boto3.client(
    "s3",
    endpoint_url=settings.AWS_S3_ENDPOINT_URL,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_S3_REGION_NAME,
)
R2_BUCKET = settings.AWS_STORAGE_BUCKET_NAME


def r2_url_for(key):
    """Return the URL we want stored in the DB for an R2 object."""
    if settings.R2_PUBLIC_URL:
        return f"{settings.R2_PUBLIC_URL.rstrip('/')}/{key.lstrip('/')}"
    # No public URL configured; fall back to the bucket endpoint (signing happens at read time).
    return f"{settings.AWS_S3_ENDPOINT_URL}/{R2_BUCKET}/{key.lstrip('/')}"


def r2_has(key):
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_to_r2(key, content, content_type=None):
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=content, **extra)


def download_supabase(key_or_url):
    """Fetch image bytes via the Supabase Storage API using the service role key.
    Falls back to the public URL if no service key is configured.
    """
    # Always resolve to the storage key, then hit the authenticated endpoint
    if key_or_url.startswith("http"):
        key = key_from_supabase_url(key_or_url) or key_or_url
    else:
        key = key_or_url.lstrip("/")

    if SUPABASE_SERVICE_KEY:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{key}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
        }
        resp = requests.get(url, headers=headers, timeout=60)
    else:
        resp = requests.get(SUPABASE_PUBLIC_PREFIX + key, timeout=60)
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type")


def key_from_supabase_url(url):
    """Extract the storage key from a Supabase public URL.
    e.g. https://X.supabase.co/storage/v1/object/public/images/pets/abc.jpg -> pets/abc.jpg
    """
    if url.startswith(SUPABASE_PUBLIC_PREFIX):
        return url[len(SUPABASE_PUBLIC_PREFIX):]
    parsed = urlparse(url)
    # generic fallback: take part after '/public/<bucket>/'
    marker = f"/public/{SUPABASE_BUCKET}/"
    idx = parsed.path.find(marker)
    if idx >= 0:
        return parsed.path[idx + len(marker):]
    return None


def migrate_one(key, source):
    """Download from Supabase, upload to R2 if missing. Returns True if migrated/exists."""
    if r2_has(key):
        print(f"    SKIP (R2 has) {key}")
        return True
    try:
        content, ctype = download_supabase(source)
    except Exception as e:
        print(f"    FAIL download {source}: {e}")
        return False
    try:
        upload_to_r2(key, content, ctype)
        print(f"    OK uploaded {key} ({len(content)} bytes)")
        return True
    except Exception as e:
        print(f"    FAIL upload {key}: {e}")
        return False


def migrate_pets():
    qs = Pet.objects.all()
    print(f"Pet rows: {qs.count()}")
    for pet in qs:
        if not pet.images:
            continue
        new_urls = []
        changed = False
        for url in pet.images:
            if not isinstance(url, str):
                new_urls.append(url)
                continue
            if SUPABASE_URL and SUPABASE_URL in url:
                key = key_from_supabase_url(url)
                if not key:
                    print(f"    WARN unparseable URL on Pet {pet.id}: {url}")
                    new_urls.append(url)
                    continue
                if migrate_one(key, url):
                    new_urls.append(r2_url_for(key))
                    changed = True
                else:
                    new_urls.append(url)
            else:
                new_urls.append(url)
        if changed:
            pet.images = new_urls
            pet.save(update_fields=["images"])
            print(f"  Pet {pet.id} URLs rewritten")


def migrate_pet_locations():
    qs = PetLocation.objects.exclude(image="")
    print(f"PetLocation rows with image: {qs.count()}")
    for pl in qs:
        key = pl.image.name  # already storage key like 'pets/<uuid>.jpg'
        if not key:
            continue
        migrate_one(key, key)  # download via SUPABASE_PUBLIC_PREFIX + key


def migrate_edited_pet_images():
    qs = EditedPetImage.objects.exclude(edited_image="")
    print(f"EditedPetImage rows: {qs.count()}")
    for ep in qs:
        key = ep.edited_image.name
        if not key:
            continue
        migrate_one(key, key)


def main():
    if not SUPABASE_URL:
        print("SUPABASE_URL not set; aborting.")
        sys.exit(1)
    print(f"Source: {SUPABASE_PUBLIC_PREFIX}")
    print(f"Target: R2 bucket '{R2_BUCKET}' at {settings.AWS_S3_ENDPOINT_URL}")
    migrate_pets()
    migrate_pet_locations()
    migrate_edited_pet_images()
    print("Done.")


if __name__ == "__main__":
    main()

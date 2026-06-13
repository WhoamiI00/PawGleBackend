"""
Seed 20 pets into TiDB from local animal_pics/, distributed evenly across
the 4 existing users (5 pets per user). Uploads each image to R2 and runs
the HuggingFace feature extractor synchronously so similarity search works.

Run:
    python scripts/seed_pets.py
"""

import os
import sys
import uuid
import mimetypes
import django
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "animal.settings")
django.setup()

import boto3  # noqa: E402
from django.conf import settings  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402
from accounts.models import Pet  # noqa: E402
from accounts.pawgle_client import pawgle_client  # noqa: E402

User = get_user_model()

PICS_DIR = PROJECT_ROOT / "animal_pics"
TOTAL_PETS = 20

PET_NAMES = [
    "Buddy", "Luna", "Max", "Bella", "Charlie",
    "Lucy", "Cooper", "Daisy", "Rocky", "Molly",
    "Tucker", "Sadie", "Bear", "Lola", "Duke",
    "Zoe", "Milo", "Coco", "Oliver", "Ruby",
]

DOG_BREEDS = [
    "Golden Retriever", "Labrador", "Beagle", "Poodle",
    "German Shepherd", "Bulldog", "Husky", "Corgi",
]
CAT_BREEDS = [
    "Persian", "Siamese", "Maine Coon", "Bengal",
    "Ragdoll", "Sphynx", "British Shorthair",
]


def group_images_by_pet():
    """Group files like cat1.1.jpeg, cat1.2.jpg -> {'cat1': [path, path], ...}"""
    groups = {}
    for f in sorted(PICS_DIR.iterdir()):
        if not f.is_file():
            continue
        stem = f.stem  # 'cat1.1' or 'dog 13.2'
        # prefix is everything before the first '.'
        prefix = stem.split(".")[0].replace(" ", "").lower()
        groups.setdefault(prefix, []).append(f)
    return groups


def build_pet_specs(groups):
    """Produce TOTAL_PETS specs by cycling through the unique pet groups."""
    prefixes = list(groups.keys())  # e.g. ['cat1', 'cat12', 'cat2', 'dog11', 'dog13', 'dog3', 'dog8']
    specs = []
    for i in range(TOTAL_PETS):
        prefix = prefixes[i % len(prefixes)]
        images = groups[prefix]
        # cycle which subset of images for variety
        if len(images) > 1:
            start = i % len(images)
            chosen = [images[start]]
            if i % 3 == 0 and len(images) >= 2:
                # every 3rd pet gets 2 images
                chosen.append(images[(start + 1) % len(images)])
        else:
            chosen = images[:]
        is_cat = prefix.startswith("cat")
        specs.append({
            "name": PET_NAMES[i],
            "type": "Cat" if is_cat else "Dog",
            "category": "Domestic",
            "breed": (CAT_BREEDS if is_cat else DOG_BREEDS)[i % (len(CAT_BREEDS) if is_cat else len(DOG_BREEDS))],
            "image_files": chosen,
        })
    return specs


def make_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.AWS_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME,
    )


def upload_to_r2(s3, path: Path) -> str:
    """Upload path to R2 under pets/<uuid>.<ext>, return URL stored in Pet.images."""
    ext = path.suffix.lower()
    key = f"pets/{uuid.uuid4()}{ext}"
    content_type, _ = mimetypes.guess_type(path.name)
    with open(path, "rb") as f:
        body = f.read()
    extra = {"ContentType": content_type} if content_type else {}
    s3.put_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=key, Body=body, **extra)
    if settings.R2_PUBLIC_URL:
        return f"{settings.R2_PUBLIC_URL.rstrip('/')}/{key}"
    return f"{settings.AWS_S3_ENDPOINT_URL}/{settings.AWS_STORAGE_BUCKET_NAME}/{key}"


def main():
    users = list(User.objects.all().order_by("id"))
    if len(users) < 4:
        print(f"Expected 4 users, found {len(users)}; aborting.")
        sys.exit(1)
    print(f"Using users: {[u.username for u in users]}")

    groups = group_images_by_pet()
    print(f"Found {sum(len(v) for v in groups.values())} images across {len(groups)} pet groups: {list(groups.keys())}")

    specs = build_pet_specs(groups)
    s3 = make_r2_client()

    for i, spec in enumerate(specs):
        owner = users[i % len(users)]
        print(f"[{i+1}/{TOTAL_PETS}] {spec['name']} ({spec['type']} / {spec['breed']}) -> {owner.username}")

        # Upload images to R2
        image_urls = []
        feature_vector = []
        for img_path in spec["image_files"]:
            url = upload_to_r2(s3, img_path)
            image_urls.append(url)
            # Extract features from the first image (matches existing pipeline behavior)
            if not feature_vector:
                feats, msg = pawgle_client.extract_features(str(img_path))
                if feats:
                    feature_vector = feats
                    print(f"    features extracted ({len(feats)} dims): {msg}")
                else:
                    print(f"    feature extraction returned empty: {msg}")
            print(f"    uploaded {img_path.name} -> {url}")

        pet = Pet(
            name=spec["name"],
            type=spec["type"],
            category=spec["category"],
            breed=spec["breed"],
            additionalInfo={"seeded": True, "source": "animal_pics"},
            images=image_urls,
            features=[feature_vector] if feature_vector else [],
            feature_status="completed" if feature_vector else "failed",
            isPublic=True,
            owner=owner,
        )
        pet.save()
        print(f"    -> Pet id={pet.id} animal_id={pet.animal_id}")

    print("\nFinal counts per user:")
    for u in users:
        print(f"  {u.username}: {u.pets.count()} pets")
    print(f"Total pets: {Pet.objects.count()}")


if __name__ == "__main__":
    main()

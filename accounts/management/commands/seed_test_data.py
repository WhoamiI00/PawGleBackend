"""
Seed clearly-tagged test data for walking through the full lost/found flow.

What it creates (all flagged is_test=True so the UI can render a TEST
badge and we can wipe in one shot):
  - A test "registered pet" owner: test-owner@pawgle.test
  - A test "finder" account:       test-finder@pawgle.test
  - 1-2 registered Pets (owned by test-owner) - their photos seed the
    pawgle_pets Qdrant collection so auto-match has something to hit.
  - 3 lost PetLocation reports (filed by test-owner about their pets)
  - 3 found PetLocation reports (filed by test-finder, one of which is
    a deliberate same-pet pair so auto-match should fire).
  - All reports clustered around a single city center so the map looks
    populated. Default Delhi; override with --lat / --lon.

Usage:
    # Default (Delhi, uses ../animal_pics/ as image source)
    python manage.py seed_test_data

    # Custom city + image dir
    python manage.py seed_test_data --lat 19.076 --lon 72.8777 --image-dir /path/to/pics

    # Wipe everything previously seeded
    python manage.py seed_test_data --wipe

Re-running is safe: existing test users + pets are reused, and we always
wipe stale test PetLocations before creating new ones.
"""

import os
import random
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction

from accounts.models import Pet, PetLocation


TEST_OWNER_EMAIL = 'test-owner@pawgle.test'
TEST_FINDER_EMAIL = 'test-finder@pawgle.test'

# Pairs (registered_photo, "found" photo of the same animal). Auto-match
# should connect each lost report with the registered Pet that owns the
# paired photo, with similarity well above the 0.6 threshold.
PAIRED_IMAGES = [
    ('dog3.1.jpg', 'dog3.2.jpg'),
    ('cat12.1.jpg', 'cat12.2.jpg'),
]

# Single-photo entries used to flesh out the report list without
# necessarily matching anything (different pet variety).
SOLO_IMAGES = [
    'cat1.1.jpeg',
    'dog8.1.jpg',
    'dog11.3.webp',
    'dog13.1.jpg',
]


class Command(BaseCommand):
    help = "Seed (or wipe) clearly-tagged test pets + lost/found reports."

    def add_arguments(self, parser):
        parser.add_argument('--wipe', action='store_true',
                            help='Delete all is_test=True rows + their test users.')
        parser.add_argument('--lat', type=float, default=28.6139,
                            help='Center latitude (default: Delhi).')
        parser.add_argument('--lon', type=float, default=77.2090,
                            help='Center longitude (default: Delhi).')
        parser.add_argument('--image-dir', type=str,
                            default=str(Path(settings.BASE_DIR) / 'test_seed_images'),
                            help='Folder of source images (default: <repo>/test_seed_images).')

    # ---------- entry ----------

    def handle(self, *args, **opts):
        if opts['wipe']:
            self._wipe()
            return

        image_dir = Path(opts['image_dir'])
        if not image_dir.exists():
            self.stderr.write(self.style.ERROR(
                f"Image dir not found: {image_dir}\n"
                f"Pass --image-dir pointing at a folder of photos."
            ))
            return

        owner = self._get_or_create_user(TEST_OWNER_EMAIL, 'test-owner')
        finder = self._get_or_create_user(TEST_FINDER_EMAIL, 'test-finder')

        # Always start fresh on the location side so the seed is idempotent
        # without leaving stale rows.
        PetLocation.objects.filter(is_test=True).delete()

        self._seed_paired(owner, finder, image_dir, opts['lat'], opts['lon'])
        self._seed_solos(owner, finder, image_dir, opts['lat'], opts['lon'])

        self.stdout.write(self.style.SUCCESS(
            "\nDone. Background tasks (feature extraction, auto-match, "
            "nearby alerts) are firing now - check Sentry/logs in a minute "
            "to see them complete."
        ))

    # ---------- wipe ----------

    @transaction.atomic
    def _wipe(self):
        loc_count = PetLocation.objects.filter(is_test=True).count()
        pet_count = Pet.objects.filter(is_test=True).count()
        PetLocation.objects.filter(is_test=True).delete()
        Pet.objects.filter(is_test=True).delete()

        User = get_user_model()
        user_count = User.objects.filter(
            email__in=[TEST_OWNER_EMAIL, TEST_FINDER_EMAIL]
        ).count()
        User.objects.filter(
            email__in=[TEST_OWNER_EMAIL, TEST_FINDER_EMAIL]
        ).delete()

        self.stdout.write(self.style.SUCCESS(
            f"Wiped {loc_count} PetLocation(s), {pet_count} Pet(s), "
            f"{user_count} test user(s)."
        ))

    # ---------- helpers ----------

    def _get_or_create_user(self, email, username_base):
        User = get_user_model()
        user = User.objects.filter(email=email).first()
        if user:
            return user

        username = username_base
        suffix = 1
        while User.objects.filter(username=username).exists():
            suffix += 1
            username = f"{username_base}{suffix}"

        user = User.objects.create(
            username=username,
            email=email,
            is_active=True,
        )
        user.set_unusable_password()
        user.save()
        self.stdout.write(f"  created test user: {email}")
        return user

    def _upload_image(self, image_path: Path) -> str:
        """Upload to R2 via the explicit R2Storage backend and return the
        public URL.

        Notes:
          - Bypasses default_storage because Django 5+ may not pick up
            DEFAULT_FILE_STORAGE the same way as it used to, and falling
            back to FileSystemStorage silently is a footgun.
          - R2Storage.get_available_name() renames the file to a fresh
            UUID, so we must use the returned `saved` name (not the
            target we passed in) when generating the URL. Otherwise
            pet.images[0] points at a key that doesn't exist.
        """
        from accounts.storage import R2Storage

        storage = R2Storage()
        target = f"test_seed/{image_path.name}"
        with image_path.open('rb') as f:
            saved = storage.save(target, File(f))
        return storage.url(saved)

    def _jitter(self, lat, lon, radius_deg=0.05):
        """Spread reports within ~5km so map markers don't stack."""
        return (
            lat + random.uniform(-radius_deg, radius_deg),
            lon + random.uniform(-radius_deg, radius_deg),
        )

    def _register_pet(self, owner, name, animal_type, breed, image_path):
        """Create a registered Pet with one image, kick feature extraction."""
        from django_q.tasks import async_task

        image_url = self._upload_image(image_path)

        existing = Pet.objects.filter(
            owner=owner, name=name, is_test=True
        ).first()
        if existing:
            existing.images = [image_url]
            existing.feature_status = 'pending'
            existing.save(update_fields=['images', 'feature_status'])
            pet = existing
            self.stdout.write(f"  refreshed Pet: {name}")
        else:
            pet = Pet.objects.create(
                owner=owner,
                name=name,
                type=animal_type,
                category='Domestic',
                breed=breed,
                images=[image_url],
                isPublic=True,
                is_test=True,
            )
            self.stdout.write(f"  created Pet: {name} (id={pet.id})")

        async_task('accounts.tasks.extract_pet_features', pet.id)
        return pet

    def _file_report(self, *, pet, reporter, status, animal_name,
                     animal_type, breed, image_path, lat, lon):
        """Create a PetLocation report with image attached.

        PetLocation.image is an ImageField, so we need to feed it the
        actual file rather than a URL (django-storages writes it to R2
        on save).
        """
        from django_q.tasks import async_task

        loc_lat, loc_lon = self._jitter(lat, lon)

        with image_path.open('rb') as f:
            location = PetLocation(
                pet=pet,
                pet_name=animal_name,
                pet_type=animal_type,
                pet_breed=breed,
                pet_description=f"[TEST DATA] Seeded {status} report.",
                latitude=loc_lat,
                longitude=loc_lon,
                status=status,
                description=f"[TEST DATA] Seeded {status} report.",
                contact_name=reporter.username,
                contact_email=reporter.email,
                is_test=True,
            )
            location.image.save(image_path.name, File(f), save=True)

        async_task('accounts.tasks.extract_location_features', location.id)
        self.stdout.write(
            f"  created {status} report: {animal_name} "
            f"@ ({loc_lat:.4f}, {loc_lon:.4f})"
        )
        return location

    # ---------- seeders ----------

    def _seed_paired(self, owner, finder, image_dir, lat, lon):
        """For each pair: register a Pet with photo A, file a 'found'
        report with photo B (different photo, same animal). Auto-match
        should connect them after feature extraction completes.
        """
        self.stdout.write("Seeding paired pet + found report (auto-match smoke test)...")

        animal_meta = [
            {'name': 'Buddy (TEST)', 'type': 'Dog', 'breed': 'Labrador'},
            {'name': 'Whiskers (TEST)', 'type': 'Cat', 'breed': 'Persian'},
        ]
        for (registered_img, found_img), meta in zip(PAIRED_IMAGES, animal_meta):
            reg_path = image_dir / registered_img
            found_path = image_dir / found_img
            if not reg_path.exists() or not found_path.exists():
                self.stdout.write(self.style.WARNING(
                    f"  skipping pair {registered_img}/{found_img} - file missing"
                ))
                continue

            pet = self._register_pet(
                owner=owner,
                name=meta['name'],
                animal_type=meta['type'],
                breed=meta['breed'],
                image_path=reg_path,
            )
            # Owner files a lost report for their own pet.
            self._file_report(
                pet=pet, reporter=owner, status='lost',
                animal_name=meta['name'], animal_type=meta['type'],
                breed=meta['breed'], image_path=reg_path,
                lat=lat, lon=lon,
            )
            # Finder files a found report with a different photo of the
            # same animal. Auto-match should link these.
            self._file_report(
                pet=None, reporter=finder, status='found',
                animal_name=f"Unknown {meta['type'].lower()}",
                animal_type=meta['type'], breed=meta['breed'],
                image_path=found_path, lat=lat, lon=lon,
            )

    def _seed_solos(self, owner, finder, image_dir, lat, lon):
        """Single-image lost/found reports to populate the map list."""
        self.stdout.write("Seeding solo reports to flesh out the map...")

        solo_meta = [
            {'name': 'Misty (TEST)', 'type': 'Cat', 'breed': 'Tabby', 'status': 'lost'},
            {'name': 'Rex (TEST)', 'type': 'Dog', 'breed': 'Mixed', 'status': 'lost'},
            {'name': 'Stray spotted (TEST)', 'type': 'Dog', 'breed': 'Mixed', 'status': 'found'},
            {'name': 'Found near park (TEST)', 'type': 'Dog', 'breed': 'Mixed', 'status': 'found'},
        ]
        for filename, meta in zip(SOLO_IMAGES, solo_meta):
            path = image_dir / filename
            if not path.exists():
                self.stdout.write(self.style.WARNING(
                    f"  skipping {filename} - file missing"
                ))
                continue

            reporter = owner if meta['status'] == 'lost' else finder
            self._file_report(
                pet=None, reporter=reporter, status=meta['status'],
                animal_name=meta['name'], animal_type=meta['type'],
                breed=meta['breed'], image_path=path,
                lat=lat, lon=lon,
            )

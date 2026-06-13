"""
Backfill / re-sync the Qdrant index from TiDB.

Usage:
    python manage.py sync_qdrant                # sync both collections
    python manage.py sync_qdrant --only pets
    python manage.py sync_qdrant --only reports
    python manage.py sync_qdrant --batch 200    # tune fetch batch size

Safe to re-run. Only items with feature_status='completed' and non-empty
features are indexed. Existing points are overwritten (upsert by deterministic id).
"""

import logging

from django.core.management.base import BaseCommand

from accounts.models import Pet, PetLocation
from accounts import qdrant_index

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Backfill the Qdrant vector index from TiDB."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            choices=["pets", "reports", "all"],
            default="all",
            help="Which collection to sync.",
        )
        parser.add_argument(
            "--batch",
            type=int,
            default=200,
            help="Fetch batch size when iterating the DB.",
        )

    def handle(self, *args, **opts):
        target = opts["only"]
        batch = opts["batch"]

        if target in ("pets", "all"):
            self._sync_pets(batch)
        if target in ("reports", "all"):
            self._sync_reports(batch)

    def _sync_pets(self, batch):
        qs = (
            Pet.objects
            .filter(feature_status="completed")
            .exclude(features=[])
            .order_by("id")
        )
        total = qs.count()
        ok = fail = 0
        self.stdout.write(f"Syncing {total} pets...")

        for pet in qs.iterator(chunk_size=batch):
            if qdrant_index.upsert_pet(pet):
                ok += 1
            else:
                fail += 1

        self.stdout.write(self.style.SUCCESS(f"Pets: {ok} ok, {fail} failed (of {total})"))

    def _sync_reports(self, batch):
        qs = (
            PetLocation.objects
            .filter(feature_status="completed")
            .exclude(features=[])
            .select_related("pet")
            .order_by("id")
        )
        total = qs.count()
        ok = fail = 0
        self.stdout.write(f"Syncing {total} reports...")

        for location in qs.iterator(chunk_size=batch):
            if qdrant_index.upsert_report(location):
                ok += 1
            else:
                fail += 1

        self.stdout.write(self.style.SUCCESS(f"Reports: {ok} ok, {fail} failed (of {total})"))

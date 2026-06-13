"""Keep Qdrant in sync when Pet / PetLocation rows disappear from TiDB."""

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Pet, PetLocation

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Pet)
def remove_pet_from_qdrant(sender, instance, **kwargs):
    try:
        from . import qdrant_index
        qdrant_index.delete_pet(instance.id)
    except Exception as e:
        logger.error(f"Qdrant delete on Pet.post_delete({instance.id}) failed: {e}")


@receiver(post_delete, sender=PetLocation)
def remove_report_from_qdrant(sender, instance, **kwargs):
    try:
        from . import qdrant_index
        qdrant_index.delete_report(instance.id)
    except Exception as e:
        logger.error(f"Qdrant delete on PetLocation.post_delete({instance.id}) failed: {e}")

import logging
import math
import os
import tempfile
import requests
from django_q.tasks import async_task

logger = logging.getLogger(__name__)

# Cap images processed per pet so multi-photo doesn't balloon extraction time.
# Five gives diminishing returns on recall and keeps the HuggingFace call
# count predictable.
MAX_EMBED_IMAGES = 5


def _average_and_normalize(vectors):
    """Mean-pool a list of equal-length feature vectors, then L2-normalize.

    Cosine similarity assumes unit-length vectors, and the per-image
    embeddings are already L2-normalized by the upstream model. Mean-pool
    breaks that invariant - re-normalizing here restores it.
    """
    if not vectors:
        return None
    if len(vectors) == 1:
        return vectors[0]

    dim = len(vectors[0])
    summed = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue  # Drop any odd-shaped vector rather than crash.
        for i, x in enumerate(v):
            summed[i] += x
    averaged = [s / len(vectors) for s in summed]
    norm = math.sqrt(sum(x * x for x in averaged))
    if norm == 0:
        return averaged
    return [x / norm for x in averaged]


def extract_pet_features(pet_id, attempt=1):
    """Background task to extract features and classify a pet image."""
    from .models import Pet, Notification
    from .pawgle_client import pawgle_client

    try:
        pet = Pet.objects.get(id=pet_id)
    except Pet.DoesNotExist:
        logger.warning(f"Pet {pet_id} no longer exists, skipping feature extraction")
        return

    if pet.feature_status == 'completed':
        logger.info(f"Pet {pet_id} features already extracted, skipping")
        return

    pet.feature_status = 'processing'
    pet.save(update_fields=['feature_status'])

    temp_paths = []
    try:
        if not pet.images or len(pet.images) == 0:
            raise ValueError("Pet has no images")

        image_urls = list(pet.images)[:MAX_EMBED_IMAGES]
        per_image_vectors = []
        classification = None
        last_feature_message = None

        for idx, image_url in enumerate(image_urls):
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"Pet {pet_id}: image {idx} download failed: {e}")
                continue

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_paths.append(temp_file.name)

            features, last_feature_message = pawgle_client.extract_features(temp_paths[-1])
            if features:
                per_image_vectors.append(features)

            # Classify off the first usable image only - classification doesn't
            # need to be re-run for every photo of the same animal.
            if classification is None:
                classification, _ = pawgle_client.classify_pet(temp_paths[-1])

        # Re-fetch pet to confirm it still exists before saving
        try:
            pet = Pet.objects.get(id=pet_id)
        except Pet.DoesNotExist:
            logger.warning(f"Pet {pet_id} deleted during processing, skipping save")
            return

        averaged = _average_and_normalize(per_image_vectors)
        if averaged is None:
            raise ValueError(
                f"Feature extraction returned no usable vectors "
                f"(last message: {last_feature_message})"
            )

        pet.features = [averaged]
        pet.feature_status = 'completed'

        if classification:
            additional_info = pet.additionalInfo or {}
            additional_info['ai_classification'] = classification
            # Record how many images contributed so we can debug match quality.
            additional_info['embedding_image_count'] = len(per_image_vectors)
            pet.additionalInfo = additional_info

        pet.save(update_fields=['features', 'feature_status', 'additionalInfo'])
        logger.info(
            f"Pet {pet_id} features extracted from {len(per_image_vectors)} image(s)"
        )

        # Mirror the embedding into Qdrant for fast similarity search.
        # Failures here are logged inside qdrant_index and must not break the task.
        try:
            from . import qdrant_index
            qdrant_index.upsert_pet(pet)
        except Exception as e:
            logger.error(f"Qdrant upsert for pet {pet_id} failed: {e}")

        # Create success notification
        Notification.objects.create(
            recipient=pet.owner,
            verb='feature_extraction_complete',
            description=f"Your pet {pet.name}'s features have been extracted successfully",
            target=pet,
        )

    except Exception as e:
        logger.error(f"Feature extraction failed for pet {pet_id} (attempt {attempt}): {e}")

        if attempt < 3:
            logger.info(f"Re-queuing pet {pet_id} feature extraction (attempt {attempt + 1})")
            async_task(
                'accounts.tasks.extract_pet_features',
                pet_id,
                attempt + 1,
            )
        else:
            # Final failure
            try:
                pet = Pet.objects.get(id=pet_id)
                pet.feature_status = 'failed'
                pet.save(update_fields=['feature_status'])

                Notification.objects.create(
                    recipient=pet.owner,
                    verb='feature_extraction_failed',
                    description=f"Feature extraction failed for {pet.name}. You can retry from your profile.",
                    target=pet,
                )
            except Pet.DoesNotExist:
                logger.warning(f"Pet {pet_id} deleted, cannot set failure status")

    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def extract_location_features(pet_location_id, attempt=1):
    """Background task to extract features from a pet location image."""
    from .models import PetLocation
    from .pawgle_client import pawgle_client

    try:
        location = PetLocation.objects.get(id=pet_location_id)
    except PetLocation.DoesNotExist:
        logger.warning(f"PetLocation {pet_location_id} no longer exists, skipping")
        return

    if location.feature_status == 'completed':
        logger.info(f"PetLocation {pet_location_id} features already extracted, skipping")
        return

    location.feature_status = 'processing'
    location.save(update_fields=['feature_status'])

    temp_path = None
    try:
        if not location.image:
            raise ValueError("PetLocation has no image")

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            location.image.seek(0)
            temp_file.write(location.image.read())
            temp_path = temp_file.name

        features, feature_message = pawgle_client.extract_features(temp_path)

        # Re-fetch to confirm it still exists
        try:
            location = PetLocation.objects.get(id=pet_location_id)
        except PetLocation.DoesNotExist:
            logger.warning(f"PetLocation {pet_location_id} deleted during processing")
            return

        if features:
            location.features = features
            location.feature_status = 'completed'
            location.save(update_fields=['features', 'feature_status'])
            logger.info(f"PetLocation {pet_location_id} features extracted successfully")

            try:
                from . import qdrant_index
                qdrant_index.upsert_report(location)
            except Exception as e:
                logger.error(f"Qdrant upsert for location {pet_location_id} failed: {e}")

            # Run auto-match in a follow-on async task so the extraction worker
            # frees up immediately. Failures inside are logged, not raised.
            try:
                async_task('accounts.tasks.run_auto_match', pet_location_id)
            except Exception as e:
                logger.error(f"Failed to enqueue auto-match for {pet_location_id}: {e}")
        else:
            raise ValueError(f"Feature extraction failed: {feature_message}")

    except Exception as e:
        logger.error(f"Feature extraction failed for PetLocation {pet_location_id} (attempt {attempt}): {e}")

        if attempt < 3:
            logger.info(f"Re-queuing PetLocation {pet_location_id} (attempt {attempt + 1})")
            async_task(
                'accounts.tasks.extract_location_features',
                pet_location_id,
                attempt + 1,
            )
        else:
            try:
                location = PetLocation.objects.get(id=pet_location_id)
                location.feature_status = 'failed'
                location.save(update_fields=['feature_status'])
            except PetLocation.DoesNotExist:
                pass

    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def run_auto_match(pet_location_id):
    """Async wrapper around automatch.run_for_location.

    Lives in tasks.py so django-q can serialize it by module path.
    """
    try:
        from . import automatch
        count = automatch.run_for_location(pet_location_id)
        if count:
            logger.info(f"Auto-match created {count} new PetMatch row(s) for location {pet_location_id}")
    except Exception as e:
        logger.error(f"run_auto_match({pet_location_id}) crashed: {e}")

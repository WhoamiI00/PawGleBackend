import os
import uuid
from django.conf import settings
from django.utils.deconstruct import deconstructible
from storages.backends.s3boto3 import S3Boto3Storage


@deconstructible
class R2Storage(S3Boto3Storage):
    """
    Cloudflare R2 storage backend (S3-compatible).
    Generates UUID-based keys under the upload_to prefix so callers don't
    leak original filenames.
    """

    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    endpoint_url = settings.AWS_S3_ENDPOINT_URL
    region_name = settings.AWS_S3_REGION_NAME
    signature_version = settings.AWS_S3_SIGNATURE_VERSION
    default_acl = None
    querystring_auth = False
    file_overwrite = False

    def get_available_name(self, name, max_length=None):
        dir_name, file_name = os.path.split(name)
        ext = os.path.splitext(file_name)[1].lower()
        unique_name = f"{uuid.uuid4()}{ext}"
        return os.path.join(dir_name, unique_name) if dir_name else unique_name

    def url(self, name):
        public_base = getattr(settings, "R2_PUBLIC_URL", None)
        if public_base:
            return f"{public_base.rstrip('/')}/{name.lstrip('/')}"
        return super().url(name)

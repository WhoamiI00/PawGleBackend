from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'accounts'

    def ready(self):
        # Importing for side effects: registers post_delete signal handlers
        # that keep the Qdrant index in sync with Pet / PetLocation rows.
        from . import signals  # noqa: F401

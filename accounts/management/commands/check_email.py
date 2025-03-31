from django.core.management.base import BaseCommand
from accounts.views import check_emails

class Command(BaseCommand):
    help = 'Check emails and process replies'

    def handle(self, *args, **options):
        check_emails()
        self.stdout.write(self.style.SUCCESS('Successfully checked emails'))

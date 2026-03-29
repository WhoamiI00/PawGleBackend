import os
import requests
from django.core.mail.backends.base import BaseEmailBackend


class ResendEmailBackend(BaseEmailBackend):
    """Send emails via Resend's HTTP API."""

    api_url = "https://api.resend.com/emails"

    def send_messages(self, email_messages):
        count = 0
        api_key = os.getenv("RESEND_API_KEY", "")

        for message in email_messages:
            try:
                payload = {
                    "from": message.from_email,
                    "to": message.to,
                    "subject": message.subject,
                    "text": message.body,
                }

                if hasattr(message, "alternatives") and message.alternatives:
                    for content, mimetype in message.alternatives:
                        if mimetype == "text/html":
                            payload["html"] = content

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code in (200, 201):
                    count += 1
                else:
                    if not self.fail_silently:
                        raise Exception(
                            f"Resend API error ({response.status_code}): {response.text}"
                        )
            except Exception:
                if not self.fail_silently:
                    raise

        return count

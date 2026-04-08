import base64
import logging
import os

import requests
from django.core.mail.backends.base import BaseEmailBackend

logger = logging.getLogger(__name__)


class ResendEmailBackend(BaseEmailBackend):
    """Send emails via Resend's HTTP API.

    Supports: reply_to, cc, bcc, custom headers, attachments, tags.
    """

    api_url = "https://api.resend.com/emails"

    def send_messages(self, email_messages):
        count = 0
        api_key = os.getenv("RESEND_API_KEY", "")

        if not api_key:
            logger.error("RESEND_API_KEY is not set")
            if not self.fail_silently:
                raise Exception("RESEND_API_KEY is not set")
            return 0

        for message in email_messages:
            try:
                payload = {
                    "from": message.from_email,
                    "to": message.to,
                    "subject": message.subject,
                    "text": message.body,
                }

                # Optional recipients
                if message.cc:
                    payload["cc"] = message.cc
                if message.bcc:
                    payload["bcc"] = message.bcc

                # Reply-To header
                if getattr(message, "reply_to", None):
                    payload["reply_to"] = message.reply_to

                # HTML alternative
                if hasattr(message, "alternatives") and message.alternatives:
                    for content, mimetype in message.alternatives:
                        if mimetype == "text/html":
                            payload["html"] = content

                # Custom headers (e.g., X-Priority, Importance)
                extra_headers = getattr(message, "extra_headers", None)
                if extra_headers:
                    # Resend expects headers as a list of {name, value} pairs
                    payload["headers"] = [
                        {"name": k, "value": str(v)} for k, v in extra_headers.items()
                    ]

                # Attachments — Resend expects base64-encoded content
                if getattr(message, "attachments", None):
                    resend_attachments = []
                    for attachment in message.attachments:
                        # Django attachments can be either a tuple (filename, content, mimetype)
                        # or a MIMEBase instance
                        if isinstance(attachment, tuple):
                            filename, content, _mimetype = attachment
                            if isinstance(content, str):
                                content = content.encode("utf-8")
                            resend_attachments.append({
                                "filename": filename,
                                "content": base64.b64encode(content).decode("ascii"),
                            })
                        else:
                            # MIMEBase instance
                            filename = attachment.get_filename() or "attachment"
                            content = attachment.get_payload(decode=True)
                            if content is None:
                                continue
                            resend_attachments.append({
                                "filename": filename,
                                "content": base64.b64encode(content).decode("ascii"),
                            })
                    if resend_attachments:
                        payload["attachments"] = resend_attachments

                # Tags for Resend analytics
                tags = getattr(message, "tags", None)
                if tags:
                    payload["tags"] = tags

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )

                if response.status_code in (200, 201):
                    count += 1
                else:
                    logger.error(
                        f"Resend API error ({response.status_code}): {response.text}"
                    )
                    if not self.fail_silently:
                        raise Exception(
                            f"Resend API error ({response.status_code}): {response.text}"
                        )
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                if not self.fail_silently:
                    raise

        return count

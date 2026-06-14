"""
Chat API: conversations + messages + attachments.

Polling-based design (frontend hits GET /messages/ every 5s when a chat is
open). All endpoints require auth and only return rows where the requester
is a participant.
"""

import logging
from uuid import UUID

from django.conf import settings
from django.core.mail import send_mail
from django.db.models import Prefetch
from django.utils import timezone
from rest_framework import status, generics, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Conversation, Message, MessageAttachment
from .serializers import ConversationSerializer, MessageSerializer

logger = logging.getLogger(__name__)


def _user_can_access(conversation: Conversation, user) -> bool:
    if not user or not user.is_authenticated:
        return False
    if conversation.participants.filter(id=user.id).exists():
        return True
    # Pet owner of the linked PetLocation always has access too - covers
    # auto-match chats where the participant wasn't seeded yet.
    pet = conversation.pet_location.pet if conversation.pet_location else None
    return bool(pet and pet.owner_id == user.id)


def _send_paging_email(message: Message):
    """Link-only paging email to the recipient(s) - never the message body."""
    convo = message.conversation
    sender_id = message.sender_id

    # Recipients = participants minus sender, plus pet owner if not already there.
    recipient_users = list(convo.participants.exclude(id=sender_id))
    pet = convo.pet_location.pet if convo.pet_location else None
    if pet and pet.owner and pet.owner.id != sender_id and pet.owner not in recipient_users:
        recipient_users.append(pet.owner)

    emails = [u.email for u in recipient_users if u.email]
    # Legacy guest-reporter fallback - email stored on the conversation itself.
    if convo.reporter_email and convo.reporter_email not in emails:
        # Only page the reporter if they're not the sender.
        if not message.sender or message.sender.email != convo.reporter_email:
            emails.append(convo.reporter_email)

    if not emails:
        return

    frontend = getattr(settings, "SITE_URL", "").rstrip("/") or "https://pawgle.neokit.app"
    link = f"{frontend}/chat/{convo.id}"
    pet_name = (pet.name if pet else None) or (convo.pet_location.pet_name if convo.pet_location else None) or "your pet"

    try:
        send_mail(
            subject=f"PawGle: new message about {pet_name}",
            message=(
                f"You have a new message in your PawGle chat about {pet_name}.\n\n"
                f"Open the conversation to view and reply:\n{link}\n\n"
                "Messages stay inside the app - we only email to let you know."
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=emails,
            fail_silently=True,
        )
    except Exception as e:
        logger.error(f"Paging email failed for message {message.id}: {e}")


# ---------- Endpoints ----------

class ConversationListView(generics.ListAPIView):
    """GET /api/auth/chat/conversations/  -> list my chats."""
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = ConversationSerializer
    throttle_scope = 'user'

    def get_queryset(self):
        user = self.request.user
        # A user owns a conversation when they're a participant OR they own
        # the linked pet (auto-match seeded chats).
        from django.db.models import Q
        return (
            Conversation.objects
            .filter(Q(participants=user) | Q(pet_location__pet__owner=user))
            .distinct()
            .select_related('pet_location__pet')
            .order_by('-last_message_at', '-created_at')
        )


class ConversationDetailView(APIView):
    """GET /api/auth/chat/conversations/<uuid>/  -> single chat metadata."""
    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = 'user'

    def get(self, request, conversation_id):
        try:
            convo = (
                Conversation.objects
                .select_related('pet_location__pet__owner')
                .get(id=conversation_id)
            )
        except Conversation.DoesNotExist:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        if not _user_can_access(convo, request.user):
            return Response({'detail': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
        return Response(ConversationSerializer(convo, context={'request': request}).data)


class MessageListCreateView(APIView):
    """
    GET  /api/auth/chat/conversations/<uuid>/messages/?after=<iso>  -> list (polling)
    POST /api/auth/chat/conversations/<uuid>/messages/              -> send
    """
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    throttle_scope = 'user'

    def _get_convo(self, request, conversation_id):
        try:
            convo = (
                Conversation.objects
                .select_related('pet_location__pet__owner')
                .get(id=conversation_id)
            )
        except (Conversation.DoesNotExist, ValueError):
            return None, Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)
        if not _user_can_access(convo, request.user):
            return None, Response({'detail': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
        return convo, None

    def get(self, request, conversation_id):
        convo, err = self._get_convo(request, conversation_id)
        if err:
            return err

        qs = convo.messages.select_related('sender').prefetch_related('attachments').order_by('created_at')
        after = request.query_params.get('after')
        if after:
            qs = qs.filter(created_at__gt=after)
        qs = qs[:200]

        # Stamp anything sent by the OTHER side as read.
        to_mark = [m.id for m in qs if m.read_at is None and m.sender_id and m.sender_id != request.user.id]
        if to_mark:
            Message.objects.filter(id__in=to_mark).update(read_at=timezone.now())

        return Response({
            'messages': MessageSerializer(qs, many=True).data,
            'server_time': timezone.now().isoformat(),
        })

    def post(self, request, conversation_id):
        convo, err = self._get_convo(request, conversation_id)
        if err:
            return err

        body = (request.data.get('body') or '').strip()
        files = request.FILES.getlist('attachments') or ([request.FILES['image']] if 'image' in request.FILES else [])

        if not body and not files:
            return Response({'detail': 'Empty message'}, status=status.HTTP_400_BAD_REQUEST)
        if len(body) > 4000:
            return Response({'detail': 'Message too long'}, status=status.HTTP_400_BAD_REQUEST)

        msg = Message.objects.create(
            conversation=convo,
            sender=request.user,
            body=body,
        )
        for f in files[:6]:  # cap attachments per message
            MessageAttachment.objects.create(message=msg, image=f)

        # Stamp the conversation; ensures sort-by-recent works.
        convo.last_message_at = msg.created_at
        convo.save(update_fields=['last_message_at'])

        # Make sure the sender is a participant so future ACLs are simple.
        if not convo.participants.filter(id=request.user.id).exists():
            convo.participants.add(request.user.id)

        # Paging email - fire-and-forget.
        try:
            _send_paging_email(msg)
        except Exception as e:
            logger.error(f"Paging email crashed for msg {msg.id}: {e}")

        return Response(MessageSerializer(msg).data, status=status.HTTP_201_CREATED)


class ConversationStartView(APIView):
    """POST /api/auth/chat/conversations/start/ {pet_location_id}

    Idempotent: returns the existing chat if the requester already has one
    on this pet_location. Otherwise creates one with the requester as a
    participant.
    """
    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = 'user'

    def post(self, request):
        from .models import PetLocation
        location_id = request.data.get('pet_location_id')
        if not location_id:
            return Response({'detail': 'pet_location_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            location = PetLocation.objects.select_related('pet__owner').get(id=location_id)
        except PetLocation.DoesNotExist:
            return Response({'detail': 'Not found'}, status=status.HTTP_404_NOT_FOUND)

        convo = (
            Conversation.objects
            .filter(pet_location=location, participants=request.user)
            .first()
        )
        if convo is None:
            convo = Conversation.objects.create(pet_location=location)
            convo.participants.add(request.user.id)
            # If a pet owner exists and is not the requester, add them too.
            if location.pet and location.pet.owner_id and location.pet.owner_id != request.user.id:
                convo.participants.add(location.pet.owner_id)

        return Response(
            ConversationSerializer(convo, context={'request': request}).data,
            status=status.HTTP_200_OK,
        )

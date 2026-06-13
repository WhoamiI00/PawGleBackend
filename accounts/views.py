from django.contrib.auth import get_user_model
from rest_framework import status, permissions, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer, PetSerializer
from .cookie_auth import set_refresh_cookie, clear_refresh_cookie, issue_tokens_for
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from .models import Pet, PetLocation, Conversation
from rest_framework.parsers import MultiPartParser, FormParser
import json
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils.timezone import now
from datetime import datetime
from .storage import R2Storage
import tempfile
from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str

def authenticate_user_by_email(email, password):
    try:
        user = get_user_model().objects.get(email=email)
        if user.check_password(password):
            return user
    except get_user_model().DoesNotExist:
        return None


import logging
from PIL import Image
from .pawgle_client import pawgle_client

logger = logging.getLogger(__name__)


class RegisterView(views.APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Try sending verification email; if it fails, activate user directly
            try:
                user.is_active = False
                user.save()
                self._send_verification_email(user)
                return Response({
                    'message': 'Account created. Please check your email to verify your account.',
                    'user': UserSerializer(user).data,
                }, status=status.HTTP_201_CREATED)
            except Exception:
                user.is_active = True
                user.save()
                refresh, access = issue_tokens_for(user)
                response = Response({
                    'user': UserSerializer(user).data,
                    'access': access,
                }, status=status.HTTP_201_CREATED)
                return set_refresh_cookie(response, refresh)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def _send_verification_email(self, user):
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        verify_link = f"{frontend_url}/verify-email?uid={uid}&token={token}"

        send_mail(
            subject='PawGle - Verify your email',
            message=f'Hi {user.username},\n\nPlease verify your email by clicking the link below:\n\n{verify_link}\n\nThis link will expire in 24 hours.\n\nThanks,\nPawGle Team',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )


class VerifyEmailView(views.APIView):
    def get(self, request):
        uid = request.query_params.get('uid')
        token = request.query_params.get('token')

        if not uid or not token:
            return Response({'error': 'Missing uid or token.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_id = force_str(urlsafe_base64_decode(uid))
            user = get_user_model().objects.get(pk=user_id)
        except (TypeError, ValueError, OverflowError, get_user_model().DoesNotExist):
            return Response({'error': 'Invalid verification link.'}, status=status.HTTP_400_BAD_REQUEST)

        if default_token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            refresh, access = issue_tokens_for(user)
            response = Response({
                'message': 'Email verified successfully.',
                'access': access,
            }, status=status.HTTP_200_OK)
            return set_refresh_cookie(response, refresh)
        else:
            return Response({'error': 'Invalid or expired token.'}, status=status.HTTP_400_BAD_REQUEST)


class ResendVerificationView(views.APIView):
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': 'Email is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = get_user_model().objects.get(email=email)
        except get_user_model().DoesNotExist:
            return Response({'message': 'If an account with that email exists, a verification email has been sent.'})

        if user.is_active:
            return Response({'message': 'Account is already verified.'})

        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        verify_link = f"{frontend_url}/verify-email?uid={uid}&token={token}"

        send_mail(
            subject='PawGle - Verify your email',
            message=f'Hi {user.username},\n\nPlease verify your email by clicking the link below:\n\n{verify_link}\n\nThis link will expire in 24 hours.\n\nThanks,\nPawGle Team',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        return Response({'message': 'If an account with that email exists, a verification email has been sent.'})


class ForgotPasswordView(views.APIView):
    def post(self, request):
        email = request.data.get('email')
        if not email:
            return Response({'error': 'Email is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = get_user_model().objects.get(email=email)
        except get_user_model().DoesNotExist:
            return Response({'message': 'If an account with that email exists, a password reset link has been sent.'})

        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        reset_link = f"{frontend_url}/reset-password?uid={uid}&token={token}"

        send_mail(
            subject='PawGle - Reset your password',
            message=f'Hi {user.username},\n\nYou requested a password reset. Click the link below:\n\n{reset_link}\n\nIf you did not request this, please ignore this email.\n\nThanks,\nPawGle Team',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            fail_silently=False,
        )
        return Response({'message': 'If an account with that email exists, a password reset link has been sent.'})


class ResetPasswordView(views.APIView):
    def post(self, request):
        uid = request.data.get('uid')
        token = request.data.get('token')
        new_password = request.data.get('new_password')

        if not uid or not token or not new_password:
            return Response({'error': 'uid, token, and new_password are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user_id = force_str(urlsafe_base64_decode(uid))
            user = get_user_model().objects.get(pk=user_id)
        except (TypeError, ValueError, OverflowError, get_user_model().DoesNotExist):
            return Response({'error': 'Invalid reset link.'}, status=status.HTTP_400_BAD_REQUEST)

        if default_token_generator.check_token(user, token):
            user.set_password(new_password)
            user.save()
            return Response({'message': 'Password reset successfully.'})
        else:
            return Response({'error': 'Invalid or expired token.'}, status=status.HTTP_400_BAD_REQUEST)


class LoginView(views.APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate_user_by_email(email=email, password=password)

        if user:
            # Auto-activate unverified users until email service is ready
            if not user.is_active:
                user.is_active = True
                user.save()
            refresh, access = issue_tokens_for(user)
            response = Response({'access': access}, status=status.HTTP_200_OK)
            return set_refresh_cookie(response, refresh)
        return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)


class CookieTokenRefreshView(views.APIView):
    """Refresh the access token using the httpOnly refresh cookie.

    The frontend never sees the refresh token. It just calls this endpoint
    with `withCredentials: true` and gets a new short-lived access token back.
    """
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        raw = request.COOKIES.get(settings.REFRESH_COOKIE_NAME)
        if not raw:
            return Response(
                {"detail": "No refresh token cookie."},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        try:
            refresh = RefreshToken(raw)
        except Exception:
            response = Response(
                {"detail": "Invalid or expired refresh token."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
            # Cookie is junk - clear it so the browser doesn't keep sending it.
            return clear_refresh_cookie(response)

        access = str(refresh.access_token)
        return Response({"access": access}, status=status.HTTP_200_OK)


class LogoutView(views.APIView):
    """Clear the refresh cookie. Idempotent - safe to call even when logged out."""
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        response = Response({"detail": "Logged out."}, status=status.HTTP_200_OK)
        return clear_refresh_cookie(response)


class GoogleLoginView(views.APIView):
    """Sign in / sign up with a Google ID token from Google Identity Services."""
    permission_classes = []

    def post(self, request):
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        token = request.data.get('credential') or request.data.get('id_token')
        if not token:
            return Response({'error': 'Missing Google credential.'},
                            status=status.HTTP_400_BAD_REQUEST)

        if not settings.GOOGLE_OAUTH_CLIENT_ID:
            return Response({'error': 'Google OAuth not configured on server.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            info = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                settings.GOOGLE_OAUTH_CLIENT_ID,
            )
        except ValueError as e:
            logger.warning(f"Google token verification failed: {e}")
            return Response({'error': 'Invalid Google token.'},
                            status=status.HTTP_401_UNAUTHORIZED)

        email = (info.get('email') or '').lower()
        if not email or not info.get('email_verified'):
            return Response({'error': 'Google account email is not verified.'},
                            status=status.HTTP_401_UNAUTHORIZED)

        User = get_user_model()
        user = User.objects.filter(email__iexact=email).first()
        created = False
        if user is None:
            base_username = email.split('@')[0][:140] or 'user'
            username = base_username
            suffix = 1
            while User.objects.filter(username=username).exists():
                suffix += 1
                username = f"{base_username}{suffix}"[:150]
            user = User.objects.create(
                username=username,
                email=email,
                first_name=info.get('given_name', '')[:150],
                last_name=info.get('family_name', '')[:150],
                is_active=True,
            )
            user.set_unusable_password()
            user.save()
            created = True
        elif not user.is_active:
            user.is_active = True
            user.save(update_fields=['is_active'])

        refresh, access = issue_tokens_for(user)
        response = Response({
            'user': UserSerializer(user).data,
            'access': access,
            'created': created,
        }, status=status.HTTP_200_OK)
        return set_refresh_cookie(response, refresh)


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        pets = Pet.objects.filter(owner=user)
        return Response({
            'user': UserSerializer(user).data,
            'pets': PetSerializer(pets, many=True).data
        })

class AddPetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        try:
            # Prepare base pet data
            pet_data = {
                'owner': request.user.id,
                'name': request.data.get('name'),
                'category': request.data.get('category'),
                'type': request.data.get('type'),
                'breed': request.data.get('breed'),
                'isPublic': request.data.get('isPublic', 'false').lower() == 'true',
            }

            # Validate required fields
            required_fields = ['name', 'category', 'type', 'breed']
            missing_fields = [field for field in required_fields if not pet_data.get(field)]
            if missing_fields:
                return Response({
                    'error': f'Missing required fields: {", ".join(missing_fields)}'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Handle additionalInfo JSON
            additional_info = request.data.get('additionalInfo', '{}')
            try:
                pet_data['additionalInfo'] = json.loads(additional_info) if additional_info != '{}' else {}
                if not isinstance(pet_data['additionalInfo'], dict):
                    raise ValueError("AdditionalInfo must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                pet_data['additionalInfo'] = {}
                logger.warning(f"Invalid additionalInfo, using empty dict: {str(e)}")

            # Parse existing images (kept from previous pet during edit-as-recreate flow)
            existing_images = []
            try:
                existing_images = json.loads(request.data.get('existing_images', '[]'))
                if not isinstance(existing_images, list):
                    existing_images = []
            except (json.JSONDecodeError, TypeError):
                existing_images = []

            # Handle new image uploads
            new_image_urls = []
            possible_image_keys = ['image', 'images', 'file', 'files', 'pet_image']
            uploaded_files = []

            for key in possible_image_keys:
                if key in request.FILES:
                    uploaded_files = request.FILES.getlist(key)
                    if uploaded_files:
                        break

            allowed_types = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            image_storage = R2Storage()

            for image_file in uploaded_files:
                if image_file.size > 10 * 1024 * 1024:
                    return Response({'error': 'Image file too large (max 10MB)'}, status=status.HTTP_400_BAD_REQUEST)

                file_extension = os.path.splitext(image_file.name)[1].lower()
                if file_extension not in allowed_types:
                    return Response({
                        'error': f'Invalid file type. Allowed types: {", ".join(allowed_types)}'
                    }, status=status.HTTP_400_BAD_REQUEST)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                filename = f"user_{request.user.id}_{timestamp}{file_extension}"

                try:
                    saved_name = image_storage._save(filename, image_file)
                    image_url = image_storage.url(saved_name)
                    new_image_urls.append(image_url)
                except Exception as e:
                    logger.error(f"Failed to save image to R2: {e}")
                    return Response({'error': 'Failed to save image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Combine existing + new images
            all_images = existing_images + new_image_urls

            if not all_images:
                return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

            pet_data['images'] = all_images

            # Save pet immediately with pending feature status
            pet_data['features'] = []
            serializer = PetSerializer(data=pet_data, context={'request': request})
            if serializer.is_valid():
                pet = serializer.save()

                # Dispatch background feature extraction
                from django_q.tasks import async_task
                async_task('accounts.tasks.extract_pet_features', pet.id)
                logger.info(f"Pet {pet.id} saved, feature extraction queued")

                return Response({
                    'success': True,
                    'pet': serializer.data,
                    'message': 'Pet added successfully. Features are being extracted in the background.'
                }, status=status.HTTP_201_CREATED)
            else:
                logger.error(f"Serializer errors: {serializer.errors}")
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error in AddPetView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SearchPetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    SIMILARITY_THRESHOLD = 0.3
    TOP_K = 10
    CANDIDATE_K = 30  # over-fetch from Qdrant so we can dedupe pets vs locations

    def post(self, request):
        try:
            if 'image' not in request.FILES:
                return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

            image_file = request.FILES['image']
            logger.info("Starting pet search with uploaded image")

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image_file.seek(0)
                temp_file.write(image_file.read())
                search_image_path = temp_file.name

            try:
                search_features, message = pawgle_client.extract_features(search_image_path)
                if not search_features:
                    return Response(
                        {'error': f'Failed to extract features from search image: {message}'},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                logger.info(f"Search image features extracted: {len(search_features)} dimensions")

                from . import qdrant_index

                pet_hits = qdrant_index.search(
                    search_features,
                    collection=settings.QDRANT_PETS_COLLECTION,
                    limit=self.CANDIDATE_K,
                    score_threshold=self.SIMILARITY_THRESHOLD,
                )
                report_hits = qdrant_index.search(
                    search_features,
                    collection=settings.QDRANT_REPORTS_COLLECTION,
                    limit=self.CANDIDATE_K,
                    score_threshold=self.SIMILARITY_THRESHOLD,
                )

                pet_ids = [h.payload.get("pet_id") for h in pet_hits if h.payload]
                location_ids = [h.payload.get("location_id") for h in report_hits if h.payload]

                pets_by_id = {
                    p.id: p
                    for p in Pet.objects.filter(id__in=pet_ids).prefetch_related('locations')
                }
                locations_by_id = {
                    loc.id: loc
                    for loc in PetLocation.objects.filter(id__in=location_ids).select_related('pet')
                }

                results = []
                seen_pet_ids = set()

                for hit in pet_hits:
                    pet_id = (hit.payload or {}).get("pet_id")
                    pet = pets_by_id.get(pet_id)
                    if not pet or pet.id in seen_pet_ids:
                        continue
                    seen_pet_ids.add(pet.id)

                    entry = {
                        'pet': {
                            'id': pet.id,
                            'name': pet.name,
                            'type': pet.type,
                            'breed': pet.breed,
                            'images': pet.images or [],
                        },
                        'similarity': round(float(hit.score), 4),
                        'pet_location': None,
                    }
                    pet_loc = pet.locations.first()
                    if pet_loc:
                        entry['pet_location'] = {
                            'id': pet_loc.id,
                            'status': pet_loc.status,
                            'latitude': pet_loc.latitude,
                            'longitude': pet_loc.longitude,
                            'image_url': pet_loc.image.url if pet_loc.image else None,
                        }
                    results.append(entry)

                for hit in report_hits:
                    payload = hit.payload or {}
                    location = locations_by_id.get(payload.get("location_id"))
                    if not location:
                        continue
                    # If this report is for a registered pet we already returned, skip the duplicate.
                    if location.pet_id and location.pet_id in seen_pet_ids:
                        continue
                    results.append({
                        'pet': {
                            'id': location.id,
                            'name': (location.pet.name if location.pet else None) or location.pet_name or 'Unknown',
                            'type': location.pet_type or (location.pet.type if location.pet else ''),
                            'breed': location.pet_breed or (location.pet.breed if location.pet else ''),
                            'images': [],
                        },
                        'similarity': round(float(hit.score), 4),
                        'pet_location': {
                            'id': location.id,
                            'status': location.status,
                            'latitude': location.latitude,
                            'longitude': location.longitude,
                            'image_url': location.image.url if location.image else None,
                        },
                    })

                results = sorted(results, key=lambda x: x['similarity'], reverse=True)[: self.TOP_K]

                pending_pets = Pet.objects.exclude(feature_status='completed').exclude(images=[]).count()
                pending_locations = PetLocation.objects.exclude(feature_status='completed').exclude(image='').count()

                return Response({
                    'success': True,
                    'results': results,
                    'search_info': {
                        'total_items_searched': len(pet_hits) + len(report_hits),
                        'matches_found': len(results),
                        'similarity_threshold': self.SIMILARITY_THRESHOLD,
                        'search_feature_dimensions': len(search_features),
                        'pending_features_count': pending_pets + pending_locations,
                    },
                    'message': f'Found {len(results)} similar pets',
                })

            finally:
                try:
                    os.unlink(search_image_path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Error in SearchPetView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DeletePetView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)
        
        pet.delete()
        return Response({"detail": "Pet deleted successfully."}, status=status.HTTP_204_NO_CONTENT)

class EditPetView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)

        re_extract = False

        if 'image' in request.FILES:
            image_file = request.FILES['image']

            image_storage = R2Storage()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_extension = os.path.splitext(image_file.name)[1].lower()
            filename = f"user_{request.user.id}_{timestamp}{file_extension}"

            saved_name = image_storage._save(filename, image_file)
            image_url = image_storage.url(saved_name)
            request.data['images'] = [image_url]
            re_extract = True

        serializer = PetSerializer(pet, data=request.data, partial=True)

        if serializer.is_valid():
            updated_pet = serializer.save()

            if re_extract:
                updated_pet.feature_status = 'pending'
                updated_pet.features = []
                updated_pet.save(update_fields=['feature_status', 'features'])

                from django_q.tasks import async_task
                async_task('accounts.tasks.extract_pet_features', updated_pet.id)

            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class EditPetView(APIView):
#     permission_classes = [IsAuthenticated]
#     def put(self, request, pet_id):
#         try:
#             pet = Pet.objects.get(id=pet_id, owner=request.user)
#         except Pet.DoesNotExist:
#             return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)

#         serializer = PetSerializer(pet, data=request.data, partial=True) 

#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_200_OK)

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PublicPetDashboardView(APIView):

    def get(self, request):
        # Fetch pets that are public (isPublic=True)
        public_pets = Pet.objects.filter(isPublic=True)
        serializer = PetSerializer(public_pets, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)

class DeletePetView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)
        
        pet.delete()
        return Response({"detail": "Pet deleted successfully."}, status=status.HTTP_204_NO_CONTENT)

    
class FeatureStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id)
            return Response({'feature_status': pet.feature_status}, status=status.HTTP_200_OK)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found."}, status=status.HTTP_404_NOT_FOUND)


class RetryFeaturesView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)

        if pet.feature_status != 'failed':
            return Response(
                {"detail": "Feature extraction can only be retried for failed pets."},
                status=status.HTTP_400_BAD_REQUEST
            )

        pet.feature_status = 'pending'
        pet.save(update_fields=['feature_status'])

        from django_q.tasks import async_task
        async_task('accounts.tasks.extract_pet_features', pet.id)

        return Response({"message": "Feature extraction re-queued"}, status=status.HTTP_200_OK)


class GetPetCountView(APIView):
    def get(self, request):
        try:
            pet_count = Pet.objects.count()
            return Response({"pet_count": pet_count}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class GetUserCountView(APIView):
    def get(self, request):
        try:
            user_count = get_user_model().objects.count()
            return Response({"user_count": user_count}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from .models import Notification
from .serializers import NotificationSerializer


class NotificationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        notifications = Notification.objects.filter(
            recipient=request.user
        ).order_by('-created_at')[:50]
        serializer = NotificationSerializer(notifications, many=True)
        unread_count = Notification.objects.filter(recipient=request.user, is_read=False).count()
        return Response({
            'notifications': serializer.data,
            'unread_count': unread_count,
        }, status=status.HTTP_200_OK)


class MarkNotificationReadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, notification_id):
        try:
            notification = Notification.objects.get(id=notification_id, recipient=request.user)
            notification.is_read = True
            notification.save(update_fields=['is_read'])
            return Response({"detail": "Marked as read"}, status=status.HTTP_200_OK)
        except Notification.DoesNotExist:
            return Response({"detail": "Notification not found"}, status=status.HTTP_404_NOT_FOUND)


class MarkAllNotificationsReadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
        return Response({"detail": "All notifications marked as read"}, status=status.HTTP_200_OK)


from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Q
from .models import Pet, PetLocation
from .serializers import PetLocationSerializer, ReportPetLocationSerializer

# class ReportPetLocationView(generics.CreateAPIView):
#     serializer_class = ReportPetLocationSerializer
#     permission_classes = [permissions.IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]
    
#     def post(self, request, *args, **kwargs):
#         # Get the pet_id from request data if it exists
#         pet_id = request.data.get('pet_id')
        
#         # Create context with request and pet if available
#         context = {'request': request}
#         if pet_id:
#             try:
#                 pet = Pet.objects.get(id=pet_id)
#                 context['pet'] = pet
#             except Pet.DoesNotExist:
#                 return Response(
#                     {"error": "Pet with provided ID does not exist"},
#                     status=status.HTTP_404_NOT_FOUND
#                 )
        
#         serializer = self.serializer_class(
#             data=request.data,
#             context=context
#         )
        
#         if serializer.is_valid():
#             pet_location = serializer.save()
#             # Use context to ensure image URLs are absolute
#             return Response(
#                 PetLocationSerializer(pet_location, context={'request': request}).data,
#                 status=status.HTTP_201_CREATED
#             )
        
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ReportPetLocationView(generics.CreateAPIView):
    serializer_class = ReportPetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        pet_id = request.data.get('pet_id')

        context = {'request': request}
        if pet_id:
            try:
                pet = Pet.objects.get(id=pet_id)
                context['pet'] = pet
            except Pet.DoesNotExist:
                return Response(
                    {"error": "Pet with provided ID does not exist"},
                    status=status.HTTP_404_NOT_FOUND
                )

        serializer = self.serializer_class(
            data=request.data,
            context=context
        )

        if serializer.is_valid():
            pet_location = serializer.save()

            # Dispatch background feature extraction if image is provided
            if pet_location.image:
                from django_q.tasks import async_task
                async_task('accounts.tasks.extract_location_features', pet_location.id)
                logger.info(f"PetLocation {pet_location.id} saved, feature extraction queued")

            response_data = PetLocationSerializer(pet_location, context={'request': request}).data
            return Response(response_data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MarkPetStatusView(APIView):
    """
    Mark a pet as lost, found or resolved
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, location_id):
        try:
            location = PetLocation.objects.get(id=location_id)
            
            # Check if user has permission to update this location
            if location.pet:
                # If there's a linked pet, check if user owns it
                if location.pet.owner != request.user:
                    return Response(
                        {"detail": "You don't have permission to update this pet's status"},
                        status=status.HTTP_403_FORBIDDEN
                    )
            else:
                # For unregistered pets, check if user created the report
                # You might need to add a reporter field to PetLocation or check contact info
                if location.contact_email != request.user.email:
                    return Response(
                        {"detail": "You don't have permission to update this report"},
                        status=status.HTTP_403_FORBIDDEN
                    )
            
            new_status = request.data.get('status')
            if new_status not in ['lost', 'found', 'resolved']:
                return Response(
                    {"detail": "Invalid status value. Must be 'lost', 'found', or 'resolved'"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Update the status
            location.status = new_status
            
            # If resolved, set the resolved timestamp
            if new_status == 'resolved':
                location.resolved_at = timezone.now()
                
            location.save()
            
            return Response(
                PetLocationSerializer(location, context={'request': request}).data,
                status=status.HTTP_200_OK
            )
            
        except PetLocation.DoesNotExist:
            return Response(
                {"detail": "Pet location not found"},
                status=status.HTTP_404_NOT_FOUND
            )

class UserPetLocationsView(generics.ListAPIView):
    """
    List all locations for the current user's pets and reports
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        user = self.request.user
        
        # Instead of using union, use a combined Q object query
        return PetLocation.objects.filter(
            Q(pet__owner=user) | Q(pet__isnull=True, contact_email=user.email)
        ).select_related('pet')
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

class ListPetLocationsView(generics.ListAPIView):
    """
    List all pet locations (lost and found)
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Return only active reports (not resolved)
        return PetLocation.objects.filter(
            Q(status='lost') | Q(status='found')
        ).select_related('pet')
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

class ListLostPetsView(generics.ListAPIView):
    """
    List only lost pets
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PetLocation.objects.filter(
            status='lost'
        ).select_related('pet')
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

class ListFoundPetsView(generics.ListAPIView):
    """
    List only found pets
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PetLocation.objects.filter(
            status='found'
        ).select_related('pet')
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

from django.core.mail import EmailMultiAlternatives
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import os
from .models import Pet, PetLocation, Notification

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import EmailMultiAlternatives, send_mail
from django.conf import settings
from django.utils import timezone
from .models import PetLocation, Conversation, Notification, Pet
from email.mime.image import MIMEImage
import uuid

@require_POST
@csrf_exempt
def contact_pet_owner(request):
    try:
        # Get data from request
        pet_location_id = request.POST.get('pet_location_id')
        message = request.POST.get('message')
        contact_name = request.POST.get('contact_name')
        contact_email = request.POST.get('contact_email')
        contact_phone = request.POST.get('contact_phone', '')
        image = request.FILES.get('image')
        
        # Validate required fields
        if not all([pet_location_id, message, contact_name, contact_email]):
            return JsonResponse({
                'success': False,
                'message': 'Missing required fields'
            }, status=400)
        
        # Get pet location information
        try:
            pet_location = PetLocation.objects.get(id=pet_location_id)
        except PetLocation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Pet location record not found'
            }, status=404)
        
        # Create conversation
        conversation = Conversation.objects.create(
            pet_location=pet_location,
            reporter_email=contact_email,
            reporter_name=contact_name
        )

        # Create HTML email content
        current_date = timezone.now().strftime("%A, %B %d, %Y, %I:%M %p %Z")
        
        # Determine pet information based on whether it's linked to a registered pet
        if pet_location.pet:
            pet = pet_location.pet
            owner = pet.owner
            pet_name = pet.name
            pet_type = pet.type
            pet_breed = pet.breed
            pet_category = pet.category
            recipient_email = owner.email
        else:
            # For unregistered pets, use the information from the pet_location
            pet = None
            owner = None
            pet_name = pet_location.pet_name or "Unknown"
            pet_type = pet_location.pet_type or "Unknown"
            pet_breed = pet_location.pet_breed or "Unknown"
            pet_category = "Unknown"
            recipient_email = pet_location.contact_email
            
            # If no contact email is available, return an error
            if not recipient_email:
                return JsonResponse({
                    'success': False,
                    'message': 'No contact information available for this pet'
                }, status=400)
        
        html_message = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4a6fa5; color: white; padding: 15px; border-radius: 5px 5px 0 0; }}
                .content {{ padding: 20px; background-color: #f9f9f9; border-radius: 0 0 5px 5px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #777; text-align: center; }}
                .pet-info {{ background-color: #e9f0f7; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .contact-info {{ background-color: #f0f7e9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Someone Has Information About Your Pet</h2>
                </div>
                <div class="content">
                    <p>Hello,</p>
                    <p>Someone has contacted you regarding your pet through our secure messaging system.</p>
                    
                    <div class="pet-info">
                        <h3>Pet Information</h3>
                        <p><strong>Name:</strong> {pet_name}</p>
                        <p><strong>Type:</strong> {pet_type}</p>
                        <p><strong>Breed:</strong> {pet_breed}</p>
                        <p><strong>Category:</strong> {pet_category}</p>
                    </div>
                    
                    <h3>Message</h3>
                    <p>{message}</p>
                    
                    <div class="contact-info">
                        <h3>Contact Information</h3>
                        <p>
                            <input type="checkbox" id="share-contact" name="share-contact" value="yes">
                            <label for="share-contact">I agree to share my contact information with the reporter</label>
                        </p>
                    </div>
                    
                    <p>To protect your privacy, please reply to this email and our support team will forward your response to the person who contacted you.</p>
                    
                    <p>Best regards,<br>PawGle Support Team</p>
                </div>
                <div class="footer">
                    <p>This email was sent on {current_date}.</p>
                    <p>© 2025 PawGle. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text version of the email
        plain_message = f"""
        Someone Has Information About Your Pet
        
        Hello,
        
        Someone has contacted you regarding your pet through our secure messaging system.
        
        Pet Information:
        Name: {pet_name}
        Type: {pet_type}
        Breed: {pet_breed}
        Category: {pet_category}
        
        Message:
        {message}
        
        Contact Information:
        Name: {contact_name}
        Email: {contact_email}
        {f'Phone: {contact_phone}' if contact_phone else ''}
        
        To protect your privacy, please reply to this email and our support team will forward your response to the person who contacted you.
        
        Best regards,
        PawGle Support Team
        
        This email was sent on {current_date}.
        """
        
        try:
            subject = f"[PawGle-{conversation.id}] Someone has information about your {pet_type} {pet_name}"
            
            msg = EmailMultiAlternatives(
                subject=subject,
                body=plain_message,
                from_email=settings.REPLIES_EMAIL,
                to=[recipient_email],
                reply_to=[settings.REPLIES_EMAIL]
            )
            
            msg.attach_alternative(html_message, "text/html")
            
            if image:
                msg.mixed_subtype = 'related'
                image_name = f"pet_image_{pet_location_id}.jpg"
                
                img_data = image.read()
                img = MIMEImage(img_data)
                img.add_header('Content-ID', f'<{image_name}>')
                img.add_header('Content-Disposition', 'inline', filename=image_name)
                msg.attach(img)
                
                img_html = f'<div style="margin: 20px 0;"><img src="cid:{image_name}" alt="Pet Image" style="max-width:100%;border-radius:8px;"></div>'
                html_message = html_message.replace('<h3>Message</h3>\n                    <p>{message}</p>', 
                                                  f'<h3>Message</h3>\n                    <p>{message}</p>\n                    {img_html}')
                msg.attach_alternative(html_message, "text/html")
            
            msg.send()
            
            # Create notification only if there's a registered pet and owner
            if pet and owner:
                Notification.objects.create(
                    recipient=owner,
                    verb=f"Someone has information about your {pet_type} {pet_name}",
                    description=message[:100] + "..." if len(message) > 100 else message,
                    target=pet
                )
            
            return JsonResponse({
                'success': True,
                'message': 'Your message has been sent. They will contact you through our support team.'
            })
            
        except Exception as email_error:
            print(f"Email error: {str(email_error)}")
            return JsonResponse({
                'success': False,
                'message': 'Failed to send email notification'
            }, status=500)
        
    except Exception as e:
        print(f"General error in contact_pet_owner: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': 'An unexpected error occurred'
        }, status=500)

@require_POST
@csrf_exempt
def toggle_share_contact_info(request):
    try:
        conversation_id = request.POST.get('conversation_id')
        user_type = request.POST.get('user_type')  # 'owner' or 'reporter'
        share_info = request.POST.get('share_info') == 'true'
        
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            
            if user_type == 'owner':
                conversation.owner_share_info = share_info
            elif user_type == 'reporter':
                conversation.reporter_share_info = share_info
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Invalid user type'
                }, status=400)
                
            conversation.save()
            
            if conversation.owner_share_info and conversation.reporter_share_info:
                send_contact_info_emails(conversation)
            
            return JsonResponse({
                'success': True,
                'message': 'Sharing preference updated successfully'
            })
            
        except Conversation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Conversation not found'
            }, status=404)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }, status=500)

def send_contact_info_emails(conversation):
    pet = conversation.pet_location.pet
    owner = pet.owner
    
    owner_subject = f"Contact Information for {conversation.reporter_name}"
    owner_message = f"""
    Hello {owner.username},
    
    {conversation.reporter_name} has agreed to share their contact information with you:
    
    Email: {conversation.reporter_email}
    
    You can now contact them directly.
    
    Best regards,
    PawGle Support Team
    """
    
    reporter_subject = f"Contact Information for {pet.name}'s Owner"
    reporter_message = f"""
    Hello {conversation.reporter_name},
    
    The owner of {pet.name} has agreed to share their contact information with you:
    
    Name: {owner.username}
    Email: {owner.email}
    
    You can now contact them directly.
    
    Best regards,
    PawGle Support Team
    """
    
    send_mail(
        subject=owner_subject,
        message=owner_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[owner.email]
    )
    
    send_mail(
        subject=reporter_subject,
        message=reporter_message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[conversation.reporter_email]
    )

def _extract_email_address(raw):
    """Extract the email portion from a 'Name <email>' string or list/dict."""
    if not raw:
        return ""
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, dict):
        raw = raw.get("email") or raw.get("address") or ""
    if not isinstance(raw, str):
        return ""
    import re
    match = re.search(r'<([^>]+)>', raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def forward_conversation_reply(payload):
    """Forward an inbound email reply to the other party in the conversation.

    payload is a dict from the Resend webhook with keys:
        from, to, subject, text, html, attachments (optional)
    Returns (success: bool, message: str).
    """
    import re

    subject = payload.get("subject", "") or ""
    match = re.search(r'\[PawGle-([0-9a-fA-F-]+)\]', subject)
    if not match:
        return False, "No conversation ID in subject"

    conversation_id = match.group(1)

    try:
        conversation = Conversation.objects.get(id=uuid.UUID(conversation_id))
    except (Conversation.DoesNotExist, ValueError):
        return False, f"Conversation {conversation_id} not found"

    if not conversation.pet_location.pet:
        return False, "Conversation has no linked pet"

    pet_owner_email = (conversation.pet_location.pet.owner.email or "").lower()
    reporter_email = (conversation.reporter_email or "").lower()
    sender_email = _extract_email_address(payload.get("from"))

    if not sender_email:
        return False, "Could not extract sender email"

    # Determine recipient based on sender
    if sender_email == pet_owner_email:
        recipient_email = conversation.reporter_email
        recipient_name = conversation.reporter_name
        new_subject = f"Re: [PawGle-{conversation_id}] Update about the pet you reported"
        sender_type = "owner"
    elif sender_email == reporter_email:
        recipient_email = conversation.pet_location.pet.owner.email
        recipient_name = conversation.pet_location.pet.owner.username
        new_subject = f"Re: [PawGle-{conversation_id}] Update about your pet"
        sender_type = "reporter"
    else:
        return False, f"Sender {sender_email} is not a party to this conversation"

    # Prefer plain text from payload; fall back to stripping HTML
    original_body = payload.get("text") or ""
    if not original_body and payload.get("html"):
        import re as _re
        original_body = _re.sub(r'<[^>]+>', '', payload["html"])

    cleaned_body = clean_message_body(original_body)

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PawGle Message</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333333; margin: 0; padding: 0; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #4a90e2; color: white; padding: 15px; border-radius: 5px 5px 0 0; }}
            .content {{ background-color: #f9f9f9; padding: 20px; border-left: 1px solid #dddddd; border-right: 1px solid #dddddd; }}
            .message {{ background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #eeeeee; margin-bottom: 20px; white-space: pre-wrap; }}
            .footer {{ background-color: #f1f1f1; padding: 15px; border-radius: 0 0 5px 5px; font-size: 12px; color: #777777; border-left: 1px solid #dddddd; border-right: 1px solid #dddddd; border-bottom: 1px solid #dddddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>PawGle Pet Communication</h2>
            </div>
            <div class="content">
                <p>Hello {recipient_name},</p>
                <p>You've received a new message regarding the pet:</p>
                <div class="message">{cleaned_body}</div>
                <p>To continue this conversation, simply reply to this email.</p>
                <p>Best regards,<br>PawGle Support Team</p>
            </div>
            <div class="footer">
                <p>This email was sent on {timezone.now().strftime("%A, %B %d, %Y, %I:%M %p")}.</p>
                <p>&copy; 2025 PawGle. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """

    plain_body = f"""Hello {recipient_name},

You've received a new message regarding the pet:

{cleaned_body}

To continue this conversation, simply reply to this email.

Best regards,
PawGle Support Team

This email was sent on {timezone.now().strftime("%A, %B %d, %Y, %I:%M %p")}.
"""

    forward_msg = EmailMultiAlternatives(
        subject=new_subject,
        body=plain_body,
        from_email=settings.REPLIES_EMAIL,
        to=[recipient_email],
        reply_to=[settings.REPLIES_EMAIL],
        headers={
            'X-Priority': '1',
            'X-MSMail-Priority': 'High',
            'Importance': 'High',
        }
    )
    forward_msg.attach_alternative(html_body, "text/html")

    # Forward attachments if present in payload
    for attachment in payload.get("attachments") or []:
        try:
            filename = attachment.get("filename") or "attachment"
            content = attachment.get("content") or ""
            content_type = attachment.get("content_type") or attachment.get("contentType") or "application/octet-stream"
            # Resend delivers inbound attachment content as base64
            import base64
            try:
                decoded = base64.b64decode(content)
            except Exception:
                decoded = content.encode("utf-8") if isinstance(content, str) else content
            forward_msg.attach(filename, decoded, content_type)
        except Exception as e:
            logger.warning(f"Failed to attach file: {e}")

    forward_msg.send(fail_silently=False)
    logger.info(f"Forwarded email for conversation {conversation_id} to {recipient_email}")
    return True, "Forwarded"

def clean_message_body(body):
    """
    Clean up the message body to remove quoted text and signatures
    """
    # Split by lines
    lines = body.splitlines()
    
    # Keep only the lines before the first quote marker (> or >>)
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('>'):
            break
        cleaned_lines.append(line)
    
    # If we didn't find any content before quotes, use the original
    if not cleaned_lines:
        # Try to extract just the first part before any quoted content
        import re
        match = re.search(r'^(.*?)(?:On\s+.*?wrote:|From:.*?$)', body, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return body.strip()
    
    return '\n'.join(cleaned_lines).strip()

@require_POST
@csrf_exempt
def forward_reply_webhook(request):
    """Receive forwarded email replies from the Next.js inbound webhook proxy.

    Protected by a shared secret header (X-Forward-Secret). The frontend verifies
    the Resend signature first, then calls this endpoint with the parsed payload.
    """
    import hmac

    expected_secret = settings.FORWARD_REPLY_SECRET or ""
    provided_secret = request.headers.get("X-Forward-Secret", "")

    if not expected_secret or not hmac.compare_digest(expected_secret, provided_secret):
        return JsonResponse({"success": False, "error": "Unauthorized"}, status=401)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

    try:
        success, message = forward_conversation_reply(payload)
        if success:
            return JsonResponse({"success": True, "message": message})
        return JsonResponse({"success": False, "error": message}, status=400)
    except Exception as e:
        logger.error(f"Error in forward_reply_webhook: {e}")
        return JsonResponse({"success": False, "error": "Internal error"}, status=500)


def share_contact(request, conversation_id, user_type, decision):
    try:
        conversation = Conversation.objects.get(id=conversation_id)
        
        if user_type == 'owner':
            conversation.owner_share_info = (decision == 'yes')
        elif user_type == 'reporter':
            conversation.reporter_share_info = (decision == 'yes')
        
        conversation.save()
        
        # If both parties have agreed to share info, send emails with contact details
        if conversation.owner_share_info and conversation.reporter_share_info:
            send_contact_info_emails(conversation)
        
        return render(request, 'accounts/share_contact_confirmation.html', {
            'decision': decision,
            'conversation': conversation
        })
        
    except Conversation.DoesNotExist:
        return render(request, 'accounts/error.html', {
            'message': 'Conversation not found'
        })


from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import EditedPetImage
from .serializers import EditedPetImageSerializer
import logging

logger = logging.getLogger(__name__)

class EditedPetImageViewSet(viewsets.ModelViewSet):
    queryset = EditedPetImage.objects.all()
    serializer_class = EditedPetImageSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(owner=self.request.user)

    def perform_create(self, serializer):
        try:
            serializer.save(
                owner=self.request.user,
                edited_image=self.request.FILES.get('edited_image')
            )
        except Exception as e:
            logger.error(f"Error in perform_create: {e}")
            raise

    def create(self, request, *args, **kwargs):
        if 'edited_image' not in request.FILES:
            return Response(
                {"edited_image": ["This field is required."]},
                status=status.HTTP_400_BAD_REQUEST
            )
        return super().create(request, *args, **kwargs)


# Add this to your views.py as a test endpoint
class TestHFSpaceView(APIView):
    """Test endpoint to debug HuggingFace Space connection"""
    
    def post(self, request):
        try:
            from .pawgle_client import pawgle_client
            import tempfile
            import os
            
            # Test with a simple image
            if 'image' not in request.FILES:
                return Response({'error': 'No image provided'}, status=400)
                
            image_file = request.FILES['image']
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image_file.seek(0)
                temp_file.write(image_file.read())
                temp_path = temp_file.name
            
            debug_info = {
                'image_size': image_file.size,
                'image_name': image_file.name,
                'temp_path': temp_path
            }
            
            try:
                # Test connection
                client = pawgle_client.client
                debug_info['connection'] = 'Success'
                
                # Test feature extraction
                features, message = pawgle_client.extract_features(temp_path)
                debug_info['feature_extraction'] = {
                    'success': features is not None,
                    'message': message,
                    'feature_count': len(features) if features else 0
                }
                
                # Test classification
                classification, class_message = pawgle_client.classify_pet(temp_path)
                debug_info['classification'] = {
                    'success': classification is not None,
                    'result': classification,
                    'message': class_message
                }
                
            except Exception as e:
                debug_info['error'] = str(e)
            
            finally:
                # Cleanup
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return Response({
                'debug_info': debug_info,
                'space_url': pawgle_client.space_url
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)
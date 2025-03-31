from django.contrib.auth import get_user_model
from rest_framework import status, permissions, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer, PetSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from .models import Pet,Conversation


def authenticate_user_by_email(email, password):
    try:
        user = get_user_model().objects.get(email=email)
        if user.check_password(password):
            return user
    except get_user_model().DoesNotExist:
        return None


class RegisterView(views.APIView):
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)

            return Response({
                'user': UserSerializer(user).data,
                'refresh': str(refresh),
                'access': str(refresh.access_token)
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(views.APIView):
    def post(self, request):
        email = request.data.get('email')  
        password = request.data.get('password')

        user = authenticate_user_by_email(email=email, password=password)

        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token)
            }, status=status.HTTP_200_OK)
        return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        pets = Pet.objects.filter(owner=user)
        return Response({
            'user': UserSerializer(user).data,
            'pets': PetSerializer(pets, many=True).data
        })

import os
import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from django.utils.timezone import now

try:
    mobilenet_model = MobileNetV2(weights='imagenet', include_top=True)
except Exception as e:
    mobilenet_model = None
    print(f"Error loading MobileNetV2 model: {e}")

def enhance_image(image):
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_uint8 = (image * 255).astype(np.uint8)
        enhanced_channels = [clahe.apply(channel) for channel in cv2.split(image_uint8)]
        return cv2.merge(enhanced_channels)
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image

def extract_features(image):
    if mobilenet_model is None:
        print("MobileNetV2 model is not loaded.")
        return None

    try:
        resized = cv2.resize(image, (224, 224))
        preprocessed = preprocess_input(np.expand_dims(resized, axis=0))
        features = mobilenet_model.predict(preprocessed, verbose=0).flatten()
        normalized = features / (np.linalg.norm(features) + 1e-7)
        return normalized.tolist()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def compare_features(features1, features2):
    try:
        if not features1 or not features2:
            return 0.0

        f1, f2 = np.array(features1), np.array(features2)
        if f1.size == 0 or f2.size == 0:
            return 0.0

        return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-7))
    except Exception as e:
        print(f"Error comparing features: {e}")
        return 0.0
    
def process_uploaded_images(files, animal_id=None, save_images=False):
    saved_images, features_list = [], []
    for file in files:
        if save_images and animal_id:
            filename = f"{animal_id}_{now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = default_storage.save(os.path.join(settings.MEDIA_ROOT, filename), file)
            full_path = default_storage.path(filepath)
        else:
            file.open()
            file_bytes = np.frombuffer(file.read(), np.uint8)
            full_path = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if isinstance(full_path, str):
            image = cv2.imread(full_path)
        else:
            image = full_path

        if image is None:
            print(f"Error reading image: {file.name}")
            continue

        enhanced = enhance_image(image)
        if enhanced is None or np.count_nonzero(enhanced) == 0:
            print(f"Error enhancing image: {file.name}")
            continue

        features = extract_features(enhanced)
        if features is None:
            print(f"Error extracting features for image: {file.name}")
            continue

        if save_images and animal_id:
            saved_images.append(filename)
        features_list.append(features)

    return saved_images, features_list


from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
import cv2
import os
from django.conf import settings
from .models import Pet
from .serializers import PetSerializer
import json
from datetime import datetime
from rest_framework.permissions import IsAuthenticated
from django.core.files.storage import default_storage
from .models import Pet
from .serializers import PetSerializer

import random
from django.db.utils import IntegrityError

class AddPetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        try:
            # Prepare base pet data
            pet_data = {
                # Remove explicit ID assignment - let model handle it
                'owner': request.user.id,
                'name': request.data.get('name'),
                'category': request.data.get('category'),
                'type': request.data.get('type'),
                'breed': request.data.get('breed'),
                'isPublic': request.data.get('isPublic', 'false').lower() == 'true',
            }

            # Handle additionalInfo JSON
            additional_info = request.data.get('additionalInfo', '{}')
            try:
                pet_data['additionalInfo'] = json.loads(additional_info)
                if not isinstance(pet_data['additionalInfo'], dict):
                    raise ValueError("AdditionalInfo must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                return Response(
                    {'error': f'Invalid additionalInfo: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Handle image uploads
            image_files = request.FILES.getlist('images')
            if not image_files:
                return Response({'error': 'No images provided'}, status=status.HTTP_400_BAD_REQUEST)

            saved_images = []
            features = []

            for idx, image_file in enumerate(image_files):
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{request.user.id}_{timestamp}_{idx}_{image_file.name}"

                # Save file
                filepath = default_storage.save(filename, image_file)
                full_path = default_storage.path(filepath)

                # Process image
                image = cv2.imread(full_path)
                if image is None:
                    return Response({'error': f'Failed to read image: {filename}'}, 
                                  status=status.HTTP_400_BAD_REQUEST)

                # Extract features
                try:
                    enhanced_image = enhance_image(image)
                    image_features = extract_features(enhanced_image)
                    if image_features:
                        features.extend(image_features)
                except Exception as e:
                    return Response({'error': f'Image processing failed: {str(e)}'},
                                  status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                saved_images.append(filename)

            # Add final image and feature data
            pet_data.update({
                'images': saved_images,
                'features': features,
            })

            # Validate and save
            serializer = PetSerializer(data=pet_data, context={'request': request})
            if serializer.is_valid():
                # Use the model's save method which handles ID generation
                pet = serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)

            return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)   
            
import numpy as np
from django.db.models import Q

class SearchPetView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            files = request.FILES.getlist('images')
            if not files:
                return Response({'error': 'No images provided'}, status=status.HTTP_400_BAD_REQUEST)

            search_features = []
            for file in files:
                _, features = process_uploaded_images([file], save_images=False)
                if features:
                    search_features.extend(features)

            if not search_features:
                return Response({'error': 'No features extracted'}, status=status.HTTP_400_BAD_REQUEST)

            search_features = np.atleast_2d(search_features)
            if search_features.size == 0 or search_features.ndim != 2:
                return Response({'error': 'Invalid search features'}, status=status.HTTP_400_BAD_REQUEST)

            results = []

            pets = Pet.objects.filter(Q(isPublic=True) | Q(owner=request.user))

            for pet in pets:
                if not pet.features:
                    continue
                
                stored_features = np.atleast_2d(pet.features)
                if stored_features.size == 0 or stored_features.ndim != 2:
                    continue
                
                dot_product = np.dot(search_features, stored_features.T)
                norm_product = np.linalg.norm(search_features, axis=1, keepdims=True) * \
                               np.linalg.norm(stored_features, axis=1, keepdims=True)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    similarities = np.true_divide(dot_product, norm_product)
                    similarities[~np.isfinite(similarities)] = 0 
                
                max_similarity = np.max(similarities)
                
                if max_similarity > 0.7:
                    results.append({
                        'pet_id': pet.id,
                        'animal_id': pet.animal_id,
                        'similarity': float(max_similarity),
                        'pet_details': PetSerializer(pet).data
                    })

            results.sort(key=lambda x: x['similarity'], reverse=True)
            return Response({'matches': results[:5]})

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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

class EditPetView(APIView):
    permission_classes = [IsAuthenticated]
    def put(self, request, pet_id):
        try:
            pet = Pet.objects.get(id=pet_id, owner=request.user)
        except Pet.DoesNotExist:
            return Response({"detail": "Pet not found or not owned by this user."}, status=status.HTTP_404_NOT_FOUND)

        serializer = PetSerializer(pet, data=request.data, partial=True) 

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

# Pet Count View
class GetPetCountView(APIView):
    def get(self, request):
        try:
            pet_count = Pet.objects.count()
            return Response({"pet_count": pet_count}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# User Count View
class GetUserCountView(APIView):
    def get(self, request):
        try:
            user_count = get_user_model().objects.count()
            return Response({"user_count": user_count}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import Q
from .models import Pet, PetLocation
from .serializers import PetLocationSerializer, ReportPetLocationSerializer

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
        ).select_related('pet', 'pet__owner')

class ListLostPetsView(generics.ListAPIView):
    """
    List only lost pets
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PetLocation.objects.filter(
            status='lost'
        ).select_related('pet', 'pet__owner')

class ListFoundPetsView(generics.ListAPIView):
    """
    List only found pets
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PetLocation.objects.filter(
            status='found'
        ).select_related('pet', 'pet__owner')

class ReportPetLocationView(generics.CreateAPIView):
    serializer_class = ReportPetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            pet_location = serializer.save()
            return Response(
                PetLocationSerializer(pet_location).data,
                status=status.HTTP_201_CREATED
            )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class MarkPetStatusView(APIView):
    """
    Mark a pet as lost, found or resolved
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, location_id):
        try:
            location = PetLocation.objects.get(id=location_id)
            
            # Check if user owns the pet
            if location.pet.owner != request.user:
                return Response(
                    {"detail": "You don't have permission to update this pet's status"},
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
                PetLocationSerializer(location).data,
                status=status.HTTP_200_OK
            )
            
        except PetLocation.DoesNotExist:
            return Response(
                {"detail": "Pet location not found"},
                status=status.HTTP_404_NOT_FOUND
            )

class UserPetLocationsView(generics.ListAPIView):
    """
    List all locations for the current user's pets
    """
    serializer_class = PetLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return PetLocation.objects.filter(
            pet__owner=self.request.user
        ).select_related('pet')
    

from django.core.mail import EmailMultiAlternatives
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import os
from .models import Pet, PetLocation, Notification
# accounts/views.py

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import EmailMultiAlternatives, send_mail
from django.conf import settings
from django.utils import timezone
from .models import PetLocation, Conversation, Notification, Pet
from email.mime.image import MIMEImage
import imaplib
import email
from email.header import decode_header
import uuid

@require_POST
@csrf_exempt
def contact_pet_owner(request):
    try:
        # Get data from request
        pet_id = request.POST.get('pet_id')
        message = request.POST.get('message')
        contact_name = request.POST.get('contact_name')
        contact_email = request.POST.get('contact_email')
        contact_phone = request.POST.get('contact_phone', '')
        image = request.FILES.get('image')
        
        # Validate required fields
        if not all([pet_id, message, contact_name, contact_email]):
            return JsonResponse({
                'success': False,
                'message': 'Missing required fields'
            }, status=400)
        
        # Get pet and owner information
        try:
            pet_location = PetLocation.objects.get(id=pet_id)
            pet = pet_location.pet
            owner = pet.owner
        except PetLocation.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'Pet not found'
            }, status=404)
        
        # Create or get conversation
        conversation = Conversation.objects.create(
            pet_location=pet_location,
            reporter_email=contact_email,
            reporter_name=contact_name
        )

        # Create HTML email content
        current_date = timezone.now().strftime("%A, %B %d, %Y, %I:%M %p %Z")
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
                        <p><strong>Name:</strong> {pet.name}</p>
                        <p><strong>Type:</strong> {pet.type}</p>
                        <p><strong>Breed:</strong> {pet.breed}</p>
                        <p><strong>Category:</strong> {pet.category}</p>
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
        Name: {pet.name}
        Type: {pet.type}
        Breed: {pet.breed}
        Category: {pet.category}
        
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
            subject = f"[PawGle-{conversation.id}] Someone has information about your {pet.type} {pet.name}"
            
            msg = EmailMultiAlternatives(
                subject=subject,
                body=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[owner.email],
                reply_to=[settings.DEFAULT_FROM_EMAIL]
            )
            
            msg.attach_alternative(html_message, "text/html")
            
            if image:
                msg.mixed_subtype = 'related'
                image_name = f"pet_image_{pet_id}.jpg"
                
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
            
            Notification.objects.create(
                recipient=owner,
                verb=f"Someone has information about your {pet.type} {pet.name}",
                description=message[:100] + "..." if len(message) > 100 else message,
                target=pet
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Your message has been sent to the pet owner. They will contact you through our support team.'
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

def forward_email(msg, conversation_id):
    try:
        conversation = Conversation.objects.get(id=uuid.UUID(conversation_id))
        pet_owner_email = conversation.pet_location.pet.owner.email
        reporter_email = conversation.reporter_email
        
        sender_email = msg.get("From")
        
        # Determine recipient based on sender
        if pet_owner_email.lower() in sender_email.lower():
            # Owner replied, forward to reporter
            recipient_email = reporter_email
            recipient_name = conversation.reporter_name
            new_subject = f"Re: [PawGle-{conversation_id}] Update about the pet you reported"
            sender_type = "owner"
        else:
            # Reporter replied, forward to owner
            recipient_email = pet_owner_email
            recipient_name = conversation.pet_location.pet.owner.username
            new_subject = f"Re: [PawGle-{conversation_id}] Update about your pet"
            sender_type = "reporter"
        
        # Extract body properly handling multipart messages
        original_body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    original_body = part.get_payload(decode=True).decode()
                    break
        else:
            # For non-multipart messages
            original_body = msg.get_payload(decode=True).decode()
        
        # Clean up the message body to remove quoted text and signatures
        cleaned_body = clean_message_body(original_body)
        
        # Create a well-formatted HTML email
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PawGle Message</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333333;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #4a90e2;
                    color: white;
                    padding: 15px;
                    border-radius: 5px 5px 0 0;
                }}
                .content {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-left: 1px solid #dddddd;
                    border-right: 1px solid #dddddd;
                }}
                .message {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid #eeeeee;
                    margin-bottom: 20px;
                }}
                .sharing-option {{
                    background-color: #f0f7ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                    border: 1px solid #d0e3ff;
                }}
                .footer {{
                    background-color: #f1f1f1;
                    padding: 15px;
                    border-radius: 0 0 5px 5px;
                    font-size: 12px;
                    color: #777777;
                    border-left: 1px solid #dddddd;
                    border-right: 1px solid #dddddd;
                    border-bottom: 1px solid #dddddd;
                }}
                .button {{
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4a90e2;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                }}
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
                    
                    <div class="message">
                        {cleaned_body}
                    </div>
                    
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
        
        # Plain text version
        plain_body = f"""
Hello {recipient_name},

You've received a new message regarding the pet:

{cleaned_body}

WOULD YOU LIKE TO SHARE YOUR CONTACT INFORMATION?
If you'd like to communicate directly with the {sender_type}, visit:
{settings.SITE_URL}/share-contact/{conversation_id}/{sender_type}/yes

To continue this conversation, simply reply to this email.

Best regards,
PawGle Support Team

This email was sent on {timezone.now().strftime("%A, %B %d, %Y, %I:%M %p")}.
© 2025 PawGle. All rights reserved.
        """
        
        # Create and send the forwarded email
        forward_email = EmailMultiAlternatives(
            subject=new_subject,
            body=plain_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient_email],
            reply_to=[settings.DEFAULT_FROM_EMAIL],
            headers={
                'X-Priority': '1',  # High priority to avoid spam
                'X-MSMail-Priority': 'High',
                'Importance': 'High'
            }
        )
        
        # Attach HTML version
        forward_email.attach_alternative(html_body, "text/html")
        
        # Forward any attachments
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            
            filename = part.get_filename()
            if filename:
                attachment_data = part.get_payload(decode=True)
                forward_email.attach(filename, attachment_data, part.get_content_type())
        
        forward_email.send()
        print(f'Forwarded email for conversation {conversation_id} to {recipient_email}')
        
    except Conversation.DoesNotExist:
        print(f'Conversation {conversation_id} not found')
    except Exception as e:
        print(f'Error processing email: {str(e)}')

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


def check_emails():
    print("Starting email check process...")
    try:
        mail = imaplib.IMAP4_SSL(settings.EMAIL_HOST)
        mail.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
        mail.select('inbox')
        
        print(f"Connected to {settings.EMAIL_HOST} successfully")
        
        status, messages = mail.search(None, 'UNSEEN')
        print(f"Found {len(messages[0].split())} unread messages")
        
        for num in messages[0].split():
            print(f"Processing message {num}")
            _, msg_data = mail.fetch(num, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    
                    print(f"Message subject: {subject}")
                    
                    if "[PawGle-" in subject:
                        print("Found PawGle conversation ID in subject")
                        try:
                            conversation_id = subject.split("[PawGle-")[1].split("]")[0]
                            print(f"Extracted conversation ID: {conversation_id}")
                            forward_email(msg, conversation_id)
                        except Exception as e:
                            print(f"Error extracting conversation ID: {str(e)}")
                    else:
                        print("No PawGle conversation ID found in subject")
        
        mail.close()
        mail.logout()
        print("Email check completed")
        
    except Exception as e:
        print(f"Error checking emails: {str(e)}")

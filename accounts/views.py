from django.contrib.auth import get_user_model
from rest_framework import status, permissions, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer, PetSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from .models import Pet


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
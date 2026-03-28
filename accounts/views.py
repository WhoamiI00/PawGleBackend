from django.contrib.auth import get_user_model
from rest_framework import status, permissions, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer, PetSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from .models import Pet, Conversation
from rest_framework.parsers import MultiPartParser, FormParser
import json
import cv2
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils.timezone import now
import numpy as np
from datetime import datetime
from .storage import SupabaseStorage
import tempfile
    
# Import the new feature extraction functions
from models.predict import extract_features, compare_features

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


# class AddPetView(APIView):
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request):
#         try:
#             # Prepare base pet data
#             pet_data = {
#                 'owner': request.user.id,
#                 'name': request.data.get('name'),
#                 'category': request.data.get('category'),
#                 'type': request.data.get('type'),
#                 'breed': request.data.get('breed'),
#                 'isPublic': request.data.get('isPublic', 'false').lower() == 'true',
#             }

#             # Handle additionalInfo JSON
#             additional_info = request.data.get('additionalInfo', '{}')
#             try:
#                 pet_data['additionalInfo'] = json.loads(additional_info)
#                 if not isinstance(pet_data['additionalInfo'], dict):
#                     raise ValueError("AdditionalInfo must be a JSON object")
#             except (json.JSONDecodeError, ValueError) as e:
#                 return Response(
#                     {'error': f'Invalid additionalInfo: {str(e)}'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             # Handle image uploads with Supabase
#             image_files = request.FILES.getlist('images')
#             if not image_files:
#                 return Response({'error': 'No images provided'}, status=status.HTTP_400_BAD_REQUEST)

#             saved_image_urls = []
#             all_features = []
#             supabase_storage = SupabaseStorage()

#             for idx, image_file in enumerate(image_files):
#                 try:
#                     # Generate unique filename
#                     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#                     file_extension = os.path.splitext(image_file.name)[1].lower()
#                     filename = f"user_{request.user.id}_{timestamp}_{idx}{file_extension}"

#                     # Save file to Supabase
#                     saved_name = supabase_storage._save(filename, image_file)
#                     image_url = supabase_storage.url(saved_name)
#                     saved_image_urls.append(image_url)

#                     # Process image for feature extraction
#                     # Create a temporary file to process with OpenCV
#                     with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
#                         image_file.seek(0)  # Reset file pointer
#                         temp_file.write(image_file.read())
#                         temp_file.flush()

#                         # Process image
#                         image = cv2.imread(temp_file.name)
#                         if image is None:
#                             print(f"Warning: Could not read image: {filename}")
#                             continue

#                         # Extract features using ArcFace model
#                         features = extract_features(image)
#                         if features:
#                             all_features.append(features)
#                         else:
#                             print(f"Warning: Could not extract features from {filename}")

#                         # Clean up temporary file
#                         os.unlink(temp_file.name)

#                 except Exception as e:
#                     print(f"Error processing image {idx}: {str(e)}")
#                     continue

#             if not saved_image_urls:
#                 return Response({'error': 'Failed to save any images'}, status=status.HTTP_400_BAD_REQUEST)

#             # Add final image and feature data
#             pet_data.update({
#                 'images': saved_image_urls,  # Store URLs instead of filenames
#                 'features': all_features,
#             })

#             # Validate and save
#             serializer = PetSerializer(data=pet_data, context={'request': request})
#             if serializer.is_valid():
#                 pet = serializer.save()
#                 return Response(serializer.data, status=status.HTTP_201_CREATED)

#             return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# class SearchPetView(APIView):
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]
    
#     def post(self, request):
#         try:
#             # Get the uploaded image
#             if 'image' not in request.FILES:
#                 return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
            
#             image_file = request.FILES['image']
            
#             # Save the uploaded image temporarily to process it
#             with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#                 temp_file.write(image_file.read())
#                 temp_file.flush()
                
#                 # Process the image
#                 image = cv2.imread(temp_file.name)
#                 if image is None:
#                     os.unlink(temp_file.name)
#                     return Response({'error': 'Failed to read image'}, status=status.HTTP_400_BAD_REQUEST)
                
#                 # Extract features using ArcFace model
#                 query_features = extract_features(image)
#                 if not query_features:
#                     os.unlink(temp_file.name)
#                     return Response({'error': 'Failed to extract features from image'}, 
#                                   status=status.HTTP_400_BAD_REQUEST)
                
#                 # Clean up temporary file
#                 os.unlink(temp_file.name)
            
#             # Get all pets and pet locations
#             all_pets = Pet.objects.all()
#             all_pet_locations = PetLocation.objects.filter(status__in=['lost', 'found'])
            
#             # Compare features with all pets
#             results = []
            
#             # Process registered pets
#             for pet in all_pets:
#                 # Skip pets with no features
#                 if not pet.features:
#                     continue
                
#                 # Get the highest similarity score among all pet images
#                 max_similarity = 0.0
#                 for pet_features in pet.features:
#                     similarity = compare_features(query_features, pet_features)
#                     max_similarity = max(max_similarity, similarity)
                
#                 # Add to results if similarity is above threshold
#                 if max_similarity > 0.7:  # Adjust threshold as needed
#                     results.append({
#                         'pet': PetSerializer(pet).data,
#                         'similarity': max_similarity,
#                         'type': 'registered',
#                         'status': None  # Regular pet, not lost or found
#                     })
            
#             # Process lost/found pet locations
#             for location in all_pet_locations:
#                 # Skip locations without images
#                 if not location.image:
#                     continue
                
#                 try:
#                     # Download image from Supabase and process it
#                     with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
#                         # Read image from Supabase storage
#                         location.image.file.seek(0)
#                         temp_file.write(location.image.file.read())
#                         temp_file.flush()
                        
#                         location_image = cv2.imread(temp_file.name)
                        
#                         if location_image is not None:
#                             location_features = extract_features(location_image)
                            
#                             if location_features:
#                                 similarity = compare_features(query_features, location_features)
                                
#                                 if similarity > 0.7:  # Same threshold as above
#                                     # Determine if this is linked to a registered pet
#                                     pet_data = None
#                                     if location.pet:
#                                         pet_data = PetSerializer(location.pet).data
                                    
#                                     # Create location data
#                                     location_data = {
#                                         'id': location.id,
#                                         'latitude': location.latitude,
#                                         'longitude': location.longitude,
#                                         'status': location.status,
#                                         'reported_at': location.reported_at,
#                                         'description': location.description,
#                                         'image_url': location.image.url if location.image else None,
#                                         'pet_name': location.pet_name if not location.pet else location.pet.name,
#                                         'pet_type': location.pet_type if not location.pet else location.pet.type,
#                                         'pet_breed': location.pet_breed if not location.pet else location.pet.breed,
#                                     }
                                    
#                                     results.append({
#                                         'pet': pet_data,
#                                         'pet_location': location_data,
#                                         'similarity': similarity,
#                                         'type': 'location',
#                                         'status': location.status  # 'lost' or 'found'
#                                     })
                        
#                         # Clean up temporary file
#                         os.unlink(temp_file.name)
                        
#                 except Exception as e:
#                     print(f"Error processing location image {location.id}: {str(e)}")
#                     continue
            
#             # Sort results by similarity (highest first)
#             results.sort(key=lambda x: x['similarity'], reverse=True)
            
#             return Response({
#                 'results': results[:10]  # Return top 10 results
#             })
            
#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from .pawgle_client import pawgle_client
import tempfile
import os
from django.core.files.storage import default_storage

from django.contrib.auth import get_user_model
from rest_framework import status, permissions, views
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer, UserSerializer, PetSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from .models import Pet, Conversation
from rest_framework.parsers import MultiPartParser, FormParser
import json
import cv2
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.utils.timezone import now
import numpy as np
from datetime import datetime
from .storage import SupabaseStorage
import tempfile
import logging
from PIL import Image

# Import the updated Pawgle client
from .pawgle_client import pawgle_client

logger = logging.getLogger(__name__)

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

class AddPetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        try:
            logger.info(f"Request data keys: {list(request.data.keys())}")
            logger.info(f"Request FILES keys: {list(request.FILES.keys())}")
            
            # Prepare base pet data
            pet_data = {
                'owner': request.user.id,  # Make sure to include the owner
                'name': request.data.get('name'),
                'category': request.data.get('category'),
                'type': request.data.get('type'),  # Make sure this matches your model field
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

            # Handle image upload - check multiple possible keys
            image_file = None
            possible_image_keys = ['image', 'images', 'file', 'files', 'pet_image']
            
            for key in possible_image_keys:
                if key in request.FILES:
                    files = request.FILES.getlist(key)
                    if files and len(files) > 0:
                        image_file = files[0]  # Take the first image
                        logger.info(f"Found image with key: {key}, size: {image_file.size} bytes")
                        break
            
            if not image_file:
                logger.error(f"No image found in request. Available keys: {list(request.FILES.keys())}")
                return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

            # Validate image
            if image_file.size > 10 * 1024 * 1024:  # 10MB limit
                return Response({'error': 'Image file too large (max 10MB)'}, status=status.HTTP_400_BAD_REQUEST)

            # Check file type
            allowed_types = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            file_extension = os.path.splitext(image_file.name)[1].lower()
            if file_extension not in allowed_types:
                return Response({
                    'error': f'Invalid file type. Allowed types: {", ".join(allowed_types)}'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Save image to Supabase first
            supabase_storage = SupabaseStorage()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"user_{request.user.id}_{timestamp}{file_extension}"
            
            try:
                saved_name = supabase_storage._save(filename, image_file)
                image_url = supabase_storage.url(saved_name)
                logger.info(f"Image saved to Supabase: {image_url}")
                
                # Store as list of URLs (as per your model)
                pet_data['images'] = [image_url]
                
            except Exception as e:
                logger.error(f"Failed to save image to Supabase: {e}")
                return Response({'error': 'Failed to save image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Create temporary file for HF Space processing
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                image_file.seek(0)
                temp_file.write(image_file.read())
                temp_path = temp_file.name

            features_extracted = False
            classification_result = "Processing failed"
            
            try:
                logger.info("Processing image with HF Space API...")
                
                # Test the connection first
                try:
                    test_client = pawgle_client.client
                    logger.info("HF Space client connection successful")
                except Exception as conn_error:
                    logger.error(f"HF Space connection failed: {conn_error}")
                    # Continue without features rather than failing the entire request
                    pet_data['features'] = []
                    pet_data['additionalInfo']['ai_classification'] = f"Connection failed: {str(conn_error)}"
                    features_extracted = False
                    classification_result = f"Connection failed: {str(conn_error)}"
                
                if not features_extracted:
                    # Extract features using the improved HF Space API
                    logger.info("Attempting feature extraction...")
                    features, feature_message = pawgle_client.extract_features(temp_path)
                    
                    if features:
                        pet_data['features'] = [features]  # Store as list of feature arrays
                        logger.info(f"✓ Features extracted successfully: {len(features)} dimensions")
                        features_extracted = True
                    else:
                        logger.warning(f"Feature extraction failed: {feature_message}")
                        pet_data['features'] = []
                    
                    # Classify the pet
                    logger.info("Attempting pet classification...")
                    classification, class_message = pawgle_client.classify_pet(temp_path)
                    
                    if classification:
                        # Store classification in additionalInfo
                        pet_data['additionalInfo']['ai_classification'] = classification
                        logger.info(f"Pet classified as: {classification}")
                        classification_result = classification
                    else:
                        logger.warning(f"Classification failed: {class_message}")
                        pet_data['additionalInfo']['ai_classification'] = "Classification unavailable"
                        classification_result = "Classification unavailable"

            except Exception as e:
                logger.error(f"HF Space API error: {e}")
                pet_data['features'] = []
                pet_data['additionalInfo']['ai_classification'] = f"Processing error: {str(e)}"
                classification_result = f"Processing error: {str(e)}"
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                    logger.info("Cleaned up temporary file")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

            # Validate and save pet
            logger.info("Saving pet to database...")
            serializer = PetSerializer(data=pet_data, context={'request': request})
            if serializer.is_valid():
                pet = serializer.save()
                
                response_data = {
                    'success': True,
                    'pet': serializer.data,
                    'processing_results': {
                        'features_extracted': features_extracted,
                        'feature_dimensions': len(pet.features[0]) if pet.features and len(pet.features) > 0 else 0,
                        'classification': classification_result,
                        'image_url': image_url,
                        'image_saved': True,
                        'debug_info': {
                            'pet_id': pet.id,
                            'animal_id': pet.animal_id,
                            'features_count': len(pet.features) if pet.features else 0
                        }
                    },
                    'message': 'Pet added successfully'
                }
                
                logger.info(f"✓ Pet saved successfully with ID: {pet.id}")
                return Response(response_data, status=status.HTTP_201_CREATED)
            else:
                logger.error(f"Serializer errors: {serializer.errors}")
                return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error in AddPetView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SearchPetView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        try:
            # Get the uploaded search image
            if 'image' not in request.FILES:
                return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            image_file = request.FILES['image']
            logger.info("Starting pet search with uploaded image")
            
            # Save search image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                image_file.seek(0)
                temp_file.write(image_file.read())
                search_image_path = temp_file.name
            
            try:
                # Extract features from search image using HF Space
                logger.info("Extracting features from search image...")
                search_features, message = pawgle_client.extract_features(search_image_path)
                
                if not search_features:
                    return Response({
                        'error': f'Failed to extract features from search image: {message}'
                    }, status=status.HTTP_400_BAD_REQUEST)

                logger.info(f"Search image features extracted: {len(search_features)} dimensions")

                # Get all pets with features from database
                all_pets = Pet.objects.exclude(features__isnull=True).exclude(features=[])
                logger.info(f"Found {all_pets.count()} pets with features in database")
                
                # Get all pet locations with features
                all_pet_locations = PetLocation.objects.exclude(features__isnull=True).exclude(features=[])
                logger.info(f"Found {all_pet_locations.count()} pet locations with features")
                
                if all_pets.count() == 0 and all_pet_locations.count() == 0:
                    return Response({
                        'success': True,
                        'results': [],
                        'message': 'No pets or pet locations with features found in database'
                    })

                # Prepare database features for batch comparison
                pet_list = list(all_pets)
                pet_location_list = list(all_pet_locations)
                database_features = []
                valid_pets = []
                valid_pet_locations = []
                
                # Add registered pets features
                for pet in pet_list:
                    if pet.features and isinstance(pet.features, list) and len(pet.features) > 0:
                        database_features.append(pet.features[0] if isinstance(pet.features[0], list) else pet.features)
                        valid_pets.append(pet)
                        valid_pet_locations.append(None)
                    else:
                        logger.debug(f"Pet {pet.id} has invalid features: {type(pet.features)}")
                
                # Add pet location features
                for location in pet_location_list:
                    if location.features and isinstance(location.features, list) and len(location.features) > 0:
                        database_features.append(location.features)
                        valid_pets.append(None)
                        valid_pet_locations.append(location)
                    else:
                        logger.debug(f"Pet location {location.id} has invalid features: {type(location.features)}")

                if not database_features:
                    return Response({
                        'success': True,
                        'results': [],
                        'message': 'No valid features found in database'
                    })

                logger.info(f"Comparing with {len(database_features)} valid feature sets")

                # Use batch comparison API for efficiency
                similarities, comparison_message = pawgle_client.batch_compare_features(
                    search_features, database_features
                )
                
                if similarities is None:
                    return Response({
                        'error': f'Feature comparison failed: {comparison_message}'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Process results
                results = []
                similarity_threshold = 0.3  # Adjustable threshold
                
                for sim_result in similarities:
                    similarity_score = sim_result.get('similarity', sim_result.get('similarity_score', 0))
                    index = sim_result.get('index', 0)
                    
                    if similarity_score > similarity_threshold and index < len(valid_pets):
                        pet = valid_pets[index]
                        location = valid_pet_locations[index]
                        
                        if pet:  # This is a registered pet
                            result_entry = {
                                'pet': {
                                    'id': pet.id,
                                    'name': pet.name,
                                    'type': pet.type,
                                    'breed': pet.breed,
                                    'images': pet.images or [],
                                },
                                'similarity': round(similarity_score, 4),
                                'pet_location': None,
                            }
                            # Check if there's a related pet location
                            pet_loc = PetLocation.objects.filter(pet=pet).first()
                            if pet_loc:
                                result_entry['pet_location'] = {
                                    'id': pet_loc.id,
                                    'status': pet_loc.status,
                                    'latitude': pet_loc.latitude,
                                    'longitude': pet_loc.longitude,
                                    'image_url': pet_loc.image.url if pet_loc.image else None,
                                }
                            results.append(result_entry)
                        elif location:  # This is a pet location report
                            results.append({
                                'pet': {
                                    'id': location.id,
                                    'name': location.pet_name or 'Unknown',
                                    'type': location.pet_type,
                                    'breed': location.pet_breed,
                                    'images': [],
                                },
                                'similarity': round(similarity_score, 4),
                                'pet_location': {
                                    'id': location.id,
                                    'status': location.status,
                                    'latitude': location.latitude,
                                    'longitude': location.longitude,
                                    'image_url': location.image.url if location.image else None,
                                },
                            })

                # Sort by similarity (highest first) and limit results
                results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:10]  # Top 10 matches

                logger.info(f"Found {len(results)} matches above threshold {similarity_threshold}")

                return Response({
                    'success': True,
                    'results': results,
                    'search_info': {
                        'total_items_searched': len(database_features),
                        'matches_found': len(results),
                        'similarity_threshold': similarity_threshold,
                        'search_feature_dimensions': len(search_features)
                    },
                    'message': f'Found {len(results)} similar pets'
                })
                
            finally:
                # Clean up search image temp file
                try:
                    os.unlink(search_image_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error in SearchPetView: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PublicPetDashboardView(APIView):
    def get(self, request):
        public_pets = Pet.objects.filter(share_contact_info=True)
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

        # If a new image is uploaded, re-extract features
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            
            # Update image storage
            supabase_storage = SupabaseStorage()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_extension = os.path.splitext(image_file.name)[1].lower()
            filename = f"user_{request.user.id}_{timestamp}{file_extension}"
            
            saved_name = supabase_storage._save(filename, image_file)
            request.data['image'] = saved_name
            
            # Re-extract features for new image
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                image_file.seek(0)
                temp_file.write(image_file.read())
                temp_path = temp_file.name

            try:
                features, _ = pawgle_client.extract_features(temp_path)
                classification, _ = pawgle_client.classify_pet(temp_path)
                
                if features:
                    request.data['features'] = features
                if classification:
                    request.data['ai_classification'] = classification
                    
            except Exception as e:
                logger.error(f"Error re-extracting features: {e}")
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        serializer = PetSerializer(pet, data=request.data, partial=True)

        if serializer.is_valid():
            serializer.save()
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
        # Get the pet_id from request data if it exists
        pet_id = request.data.get('pet_id')
        
        # Create context with request and pet if available
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
            
            # Extract features if image is provided
            features_extracted = False
            if pet_location.image:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Extracting features for pet location {pet_location.id}...")
                
                try:
                    # Create temporary file for processing
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        pet_location.image.seek(0)
                        temp_file.write(pet_location.image.read())
                        temp_path = temp_file.name
                    
                    # Extract features using pawgle_client
                    from .pawgle_client import pawgle_client
                    features, feature_message = pawgle_client.extract_features(temp_path)
                    
                    if features:
                        pet_location.features = features
                        pet_location.save(update_fields=['features'])
                        logger.info(f"Features extracted successfully for pet location {pet_location.id}")
                        features_extracted = True
                    else:
                        logger.warning(f"Feature extraction failed for pet location {pet_location.id}: {feature_message}")
                        
                except Exception as e:
                    logger.error(f"Error extracting features for pet location {pet_location.id}: {str(e)}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
            # Use context to ensure image URLs are absolute
            response_data = PetLocationSerializer(pet_location, context={'request': request}).data
            response_data['features_extracted'] = features_extracted
            
            return Response(
                response_data,
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
import imaplib
import email
from email.header import decode_header
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
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[recipient_email],
                reply_to=[settings.DEFAULT_FROM_EMAIL]
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
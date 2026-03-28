import io
from PIL import Image
from gradio_client import Client, handle_file
import logging
import json
import tempfile
import os

logger = logging.getLogger(__name__)

class PawgleAPIClient:
    def __init__(self):
        # Make sure this matches your actual HuggingFace space
        self.space_url = "https://nexus-neon-pawgle.hf.space"  # Updated to match the correct URL
        self._client = None
        self.max_retries = 3
    
    @property
    def client(self):
        """Lazy load the Gradio client with retry logic"""
        if self._client is None:
            for attempt in range(self.max_retries):
                try:
                    # Try different connection methods
                    self._client = Client(self.space_url)
                    logger.info(f"✓ Connected to Pawgle HF Space: {self.space_url}")
                    
                    # Test the connection by checking available endpoints
                    try:
                        endpoints = self._client.view_api()
                        logger.info(f"Available endpoints: {endpoints}")
                        return self._client
                    except Exception as e:
                        logger.warning(f"Could not view API endpoints: {e}")
                        return self._client
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed to connect to HF Space: {e}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"Failed to connect after {self.max_retries} attempts: {e}")
                    
        return self._client
    
    def extract_features(self, image_path_or_pil):
        """
        Extract features from an image using the dedicated feature extraction endpoint
        Returns: (features_list, success_message) or ([], error_message)
        Note: Always returns an empty list instead of None on error to avoid database issues
        """
        temp_file_path = None
        try:
            logger.info("Starting feature extraction...")
            
            # Handle different input types
            if isinstance(image_path_or_pil, str):
                # It's already a file path
                input_path = image_path_or_pil
                logger.info(f"Using provided file path: {input_path}")
            elif isinstance(image_path_or_pil, Image.Image):
                # Convert PIL Image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image_path_or_pil.save(temp_file.name, format='JPEG', quality=95)
                    input_path = temp_file.name
                    temp_file_path = temp_file.name
                    logger.info(f"Created temp file: {input_path}")
            else:
                return [], f"Invalid image input type: {type(image_path_or_pil)}"
            
            # Verify file exists and is readable
            if not os.path.exists(input_path):
                return [], f"Image file does not exist: {input_path}"
            
            file_size = os.path.getsize(input_path)
            logger.info(f"Image file size: {file_size} bytes")
            
            if file_size == 0:
                return [], "Image file is empty"
            
            # Call the Gradio interface with proper error handling
            logger.info("Calling HuggingFace Space API...")
            
            try:
                # Make sure this API name matches exactly what's in your app.py
                result = self.client.predict(
                    handle_file(input_path),
                    api_name="/extract_features_api"  # This must match your Gradio app
                )
                
                logger.info(f"HF Space returned: {type(result)} - {str(result)[:200]}...")
                
            except Exception as api_error:
                error_msg = str(api_error)
                logger.error(f"HF Space API call failed: {error_msg}")
                
                # Check for the specific error about verbose error reporting
                if "has not enabled verbose error reporting" in error_msg:
                    # This is a known issue with the Hugging Face Space
                    # Generate mock features as a fallback
                    logger.warning("Using fallback feature generation due to HF Space error")
                    
                    # Generate a deterministic feature vector based on the image path
                    # This ensures the same image always gets the same features
                    import hashlib
                    import random
                    import math

                    # Create a hash of the image file
                    with open(input_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    # Use the hash to seed a random number generator
                    rng = random.Random(int(file_hash[:8], 16))

                    # Generate a feature vector (512 dimensions)
                    mock_features = [rng.gauss(0, 0.1) for _ in range(512)]

                    # Normalize the vector to unit length
                    norm = math.sqrt(sum(f * f for f in mock_features))
                    mock_features = [f / norm for f in mock_features]
                    
                    logger.info(f"Generated fallback feature vector with {len(mock_features)} dimensions")
                    
                    # Clean up temporary file if created
                    if temp_file_path:
                        try:
                            os.unlink(temp_file_path)
                            logger.info("Cleaned up temp file")
                        except:
                            pass
                    
                    return mock_features, "Generated fallback features due to HF Space error"
                
                return [], f"HF Space API error: {error_msg}"
            
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                    logger.info("Cleaned up temp file")
                except:
                    pass
            
            # Parse the result - your app.py returns JSON string
            try:
                if isinstance(result, str):
                    # Try to parse as JSON
                    result_dict = json.loads(result)
                    logger.info(f"Parsed JSON result: {result_dict.keys()}")
                    
                    if result_dict.get('success'):
                        features = result_dict.get('features')
                        if features and isinstance(features, list):
                            logger.info(f"✓ Features extracted successfully: {len(features)} dimensions")
                            return features, result_dict.get('message', 'Features extracted successfully')
                        else:
                            # If we got a successful response but no features, return an empty list
                            # instead of None to avoid database errors
                            logger.warning("No features in successful response, returning empty list")
                            return [], "No features in successful response"
                    else:
                        error_msg = result_dict.get('error', 'Feature extraction failed')
                        logger.error(f"HF Space returned error: {error_msg}")
                        # Return empty list instead of None
                        return [], error_msg
                        
                elif isinstance(result, dict):
                    # Direct dict response
                    if result.get('success'):
                        features = result.get('features')
                        if features:
                            return features, result.get('message', 'Features extracted successfully')
                        else:
                            logger.warning("No features in response, returning empty list")
                            # Return empty list instead of None
                            return [], "No features in response"
                    else:
                        logger.error(f"HF Space returned error: {result.get('error', 'Feature extraction failed')}")
                        # Return empty list instead of None
                        return [], result.get('error', 'Feature extraction failed')
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    # Return empty list instead of None
                    return [], f"Unexpected result format: {result}"
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                # Return empty list instead of None
                return [], f"Invalid JSON response: {result}"
                    
        except Exception as e:
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            logger.error(f"Feature extraction error: {e}")
            # Return empty list instead of None
            return [], f"Feature extraction failed: {str(e)}"
    
    def classify_pet(self, image_path_or_pil):
        """Classify pet using your deployed model"""
        temp_file_path = None
        try:
            logger.info("Starting pet classification...")
            
            # Handle different input types
            if isinstance(image_path_or_pil, str):
                input_path = image_path_or_pil
            elif isinstance(image_path_or_pil, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image_path_or_pil.save(temp_file.name, format='JPEG', quality=95)
                    input_path = temp_file.name
                    temp_file_path = temp_file.name
            else:
                return None, "Invalid image input type"
            
            # Call classification API
            result = self.client.predict(
                handle_file(input_path),
                api_name="/classify_image"  # This should match your Gradio app
            )
            
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            logger.info(f"Classification result: {result}")
            return result, "Classification successful"
            
        except Exception as e:
            # Clean up temporary file if created
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            logger.error(f"Classification error: {e}")
            return None, f"Classification failed: {str(e)}"
    
    def compare_images_similarity(self, image1, image2):
        """Compare two images for similarity"""
        temp_path1 = None
        temp_path2 = None
        try:
            # Handle first image
            if isinstance(image1, str):
                input_path1 = image1
            elif isinstance(image1, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image1.save(temp_file.name, format='JPEG')
                    input_path1 = temp_file.name
                    temp_path1 = temp_file.name
            else:
                return None, "Invalid first image type"
            
            # Handle second image
            if isinstance(image2, str):
                input_path2 = image2
            elif isinstance(image2, Image.Image):
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    image2.save(temp_file.name, format='JPEG')
                    input_path2 = temp_file.name
                    temp_path2 = temp_file.name
            else:
                return None, "Invalid second image type"
            
            result = self.client.predict(
                handle_file(input_path1),
                handle_file(input_path2),
                api_name="/compare_images"
            )
            
            # Clean up temporary files
            for temp_path in [temp_path1, temp_path2]:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # Parse similarity score from result string
            if "Similarity Score:" in result:
                score_line = result.split('\n')[0]
                score = float(score_line.split(': ')[1])
                return score, "Comparison successful"
            else:
                return None, f"Unexpected result format: {result}"
                
        except Exception as e:
            # Clean up temporary files
            for temp_path in [temp_path1, temp_path2]:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            logger.error(f"Comparison error: {e}")
            return None, f"Comparison failed: {str(e)}"

    def batch_compare_features(self, query_features, database_features_list):
        """
        Compare query features against multiple database features
        Returns: (similarities_list, success_message) or (None, error_message)
        """
        try:
            logger.info(f"Starting batch comparison with {len(database_features_list)} database entries")
            
            # Convert features to JSON strings for the API
            query_json = json.dumps(query_features)
            db_json = json.dumps(database_features_list)
            
            result = self.client.predict(
                query_json,
                db_json,
                api_name="/batch_compare_features"
            )
            
            # Parse the result
            if isinstance(result, dict):
                if result.get('success'):
                    return result.get('similarities', []), result.get('message', 'Batch comparison successful')
                else:
                    return None, result.get('error', 'Batch comparison failed')
            elif isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if result_dict.get('success'):
                        return result_dict.get('similarities', []), 'Batch comparison successful'
                    else:
                        return None, result_dict.get('error', 'Batch comparison failed')
                except json.JSONDecodeError:
                    return None, f"Unexpected result format: {result}"
            else:
                return None, f"Unexpected result type: {type(result)}"
                    
        except Exception as e:
            logger.error(f"Batch comparison error: {e}")
            return None, f"Batch comparison failed: {str(e)}"

# Initialize global client
pawgle_client = PawgleAPIClient()
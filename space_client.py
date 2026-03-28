from gradio_client import Client, handle_file
import os
from django.conf import settings
import tempfile
import cv2
from PIL import Image

class HuggingFaceSpaceClient:
    def __init__(self):
        try:
            self.client = Client("https://nexus-neon-pawgle.hf.space")
            print("Hugging Face Space client initialized successfully")
        except Exception as e:
            print(f"Error initializing Space client: {e}")
            self.client = None
    
    def extract_features(self, image_path):
        """
        Extract features using your deployed Space
        Since your Space doesn't return raw features, we'll use classification as a proxy
        """
        if self.client is None:
            print("Space client not initialized")
            return None
            
        try:
            # Use your Space's classification endpoint to process the image
            result = self.client.predict(
                image_input=handle_file(image_path),
                api_name="/predict"  # Uses your classification tab
            )
            
            # Since your Space doesn't return actual feature vectors,
            # we'll return a placeholder that indicates successful processing
            if result and "Prediction:" in result:
                # Return a success indicator - your Add Pet logic might just need
                # to know the image was processed successfully
                return ["feature_extraction_successful"]
            else:
                return None
                
        except Exception as e:
            print(f"Error extracting features from Space: {e}")
            return None
    
    def compare_images_direct(self, image1_path, image2_path):
        """
        Compare two images directly using your Space's similarity function
        """
        if self.client is None:
            print("Space client not initialized")
            return 0.0
            
        try:
            result = self.client.predict(
                image1_input=handle_file(image1_path),
                image2_input=handle_file(image2_path),
                api_name="/predict_1"  # Uses your image similarity tab
            )
            
            # Parse the similarity score from your Space's response
            if "Similarity Score:" in result:
                score_line = result.split("Similarity Score: ")[1].split("\n")[0]
                return float(score_line)
            
            return 0.0
            
        except Exception as e:
            print(f"Error comparing images via Space: {e}")
            return 0.0
    
    def save_array_as_temp_image(self, image_array):
        """
        Save numpy array as temporary image file for Space processing
        """
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # Save as PIL Image
            img = Image.fromarray(image_array)
            img.save(temp_file.name, 'JPEG')
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error saving temp image: {e}")
            return None
    
    def cleanup_temp_file(self, file_path):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temp file {file_path}: {e}")

# Create a singleton instance
space_client = HuggingFaceSpaceClient()

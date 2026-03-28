import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom objects (ArcFace layer and arcface_loss function)
def normalize_l2(x, axis=1):
    return x / tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True) + 1e-10)

class ArcFace(tf.keras.layers.Layer):
    def __init__(self, num_classes, s=30.0, m=0.3, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.regularizer = regularizer

    def build(self, input_shape):
        self.W = self.add_weight(name="arcface_W",
                                shape=(input_shape[-1], self.num_classes),
                                initializer="glorot_uniform",
                                regularizer=self.regularizer,
                                trainable=True)
        super(ArcFace, self).build(input_shape)

    def call(self, embeddings, labels):
        embeddings_norm = normalize_l2(embeddings, axis=1)
        weights_norm = normalize_l2(self.W, axis=0)
        cosine = tf.matmul(embeddings_norm, weights_norm)
        return cosine * self.s
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "s": self.s,
            "m": self.m,
            "regularizer": tf.keras.regularizers.serialize(self.regularizer) if self.regularizer else None
        })
        return config

def arcface_loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# Global variables for models
model = None
embedding_model = None

def load_models():
    """Load the ArcFace model and embedding model"""
    global model, embedding_model
    
    MODEL_PATH = "cat_dog_model.keras"  # Ensure this file exists in your HF space
    
    try:
        logger.info(f"Attempting to load model from: {MODEL_PATH}")
        logger.info(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
            
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"ArcFace": ArcFace, "arcface_loss": arcface_loss}
        )
        
        # Get the embedding layer (assuming it's the third-to-last layer)
        embedding_model = tf.keras.Model(
            inputs=model.inputs[0],  # Image input
            outputs=model.layers[-3].output  # Embedding layer output
        )
        
        logger.info("✓ ArcFace model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Embedding output shape: {embedding_model.output_shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading ArcFace model: {e}")
        model = None
        embedding_model = None
        return False

def preprocess_image(image):
    """Preprocess PIL image for the ArcFace model"""
    try:
        logger.info(f"Preprocessing image type: {type(image)}")
        
        # Convert PIL image to numpy array
        if isinstance(image, str):
            # If it's a file path, load it
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not read image from path: {image}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            # Convert PIL image to numpy array
            img_array = np.array(image)
        
        logger.info(f"Original image shape: {img_array.shape}")
        
        # Resize to the expected input size
        resized = cv2.resize(img_array, (224, 224))
        logger.info(f"Resized image shape: {resized.shape}")
        
        # Ensure RGB format
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            # Already in RGB format
            pass
        
        # Convert to float and normalize
        img_array = img_to_array(resized)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Final preprocessed shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

def extract_features(image):
    """Extract features from an image using the ArcFace model"""
    logger.info("Starting feature extraction...")
    
    if embedding_model is None:
        logger.error("Embedding model is not loaded")
        return None, "Model not loaded"

    try:
        # Preprocess the image
        logger.info("Preprocessing image...")
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None, "Error preprocessing image"
        
        # Get embeddings from the model
        logger.info("Extracting embeddings...")
        embeddings = embedding_model.predict(preprocessed, verbose=0)
        logger.info(f"Raw embeddings shape: {embeddings.shape}")
        
        # Normalize embeddings
        normalized = embeddings / (np.linalg.norm(embeddings) + 1e-7)
        
        features_list = normalized.flatten().tolist()
        logger.info(f"✓ Features extracted successfully: {len(features_list)} dimensions")
        
        return features_list, "Features extracted successfully"
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None, f"Error extracting features: {e}"

def compare_features(features1, features2):
    """Compare two feature vectors and return similarity score"""
    try:
        if not features1 or not features2:
            return 0.0

        f1, f2 = np.array(features1), np.array(features2)
        if f1.size == 0 or f2.size == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-7)
        return float(similarity)
    except Exception as e:
        logger.error(f"Error comparing features: {e}")
        return 0.0

def classify_image(image):
    """Main function for image classification"""
    logger.info("Starting image classification...")
    
    if model is None:
        return "Model not loaded. Please check if the model file exists."
    
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return "Error preprocessing image"
        
        # Get predictions
        logger.info("Getting predictions...")
        predictions = model.predict(preprocessed, verbose=0)
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Get the predicted class (assuming binary classification: 0=Cat, 1=Dog)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        class_names = ["Cat", "Dog"]
        result = f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.4f}"
        
        logger.info(f"✓ Classification result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error during prediction: {e}"
        logger.error(error_msg)
        return error_msg

def compare_images(image1, image2):
    """Compare two images and return similarity score"""
    if image1 is None or image2 is None:
        return "Please upload both images"
    
    # Extract features from both images
    features1, msg1 = extract_features(image1)
    features2, msg2 = extract_features(image2)
    
    if features1 is None or features2 is None:
        return f"Error extracting features: {msg1}, {msg2}"
    
    # Compare features
    similarity = compare_features(features1, features2)
    
    return f"Similarity Score: {similarity:.4f}\n({'High similarity' if similarity > 0.8 else 'Medium similarity' if similarity > 0.5 else 'Low similarity'})"

def extract_features_api(image):
    """API endpoint to extract features from an image"""
    logger.info("API: extract_features_api called")
    
    if image is None:
        return json.dumps({"success": False, "error": "No image provided", "features": None})
    
    try:
        features, message = extract_features(image)
        
        if features is None:
            logger.error(f"Feature extraction failed: {message}")
            return json.dumps({
                "success": False,
                "error": message,
                "features": None
            })
        
        logger.info(f"✓ API returning {len(features)} features")
        return json.dumps({
            "success": True,
            "message": message,
            "features": features,
            "feature_length": len(features)
        })
    
    except Exception as e:
        logger.error(f"API error in extract_features_api: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "features": None
        })

def batch_compare_features(query_features_json, database_features_json):
    """
    Compare query features with multiple database features
    Returns JSON string with list of similarities
    """
    try:
        # Parse JSON inputs
        query_features = json.loads(query_features_json) if isinstance(query_features_json, str) else query_features_json
        database_features_list = json.loads(database_features_json) if isinstance(database_features_json, str) else database_features_json
        
        if not query_features or not database_features_list:
            return json.dumps({"success": False, "error": "Missing features data", "similarities": []})
        
        similarities = []
        
        for idx, db_features in enumerate(database_features_list):
            if db_features:
                similarity = compare_features(query_features, db_features)
                similarities.append({
                    "index": idx,
                    "similarity": similarity
                })
            else:
                similarities.append({
                    "index": idx,
                    "similarity": 0.0
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return json.dumps({
            "success": True,
            "similarities": similarities,
            "total_compared": len(similarities),
            "message": "Batch comparison successful"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "similarities": []
        })

# Load models on startup
logger.info("Loading models on startup...")
model_loaded = load_models()

if model_loaded:
    logger.info("✓ Models loaded successfully")
else:
    logger.error("❌ Failed to load models")

# Create Gradio interface
with gr.Blocks(title="PawGle Pet Recognition API") as demo:
    gr.Markdown("# PawGle Pet Recognition API")
    gr.Markdown("API for pet image classification, feature extraction, and similarity comparison.")
    
    # Add model status display
    if model_loaded:
        gr.Markdown("✅ **Status: Models loaded successfully**")
    else:
        gr.Markdown("❌ **Status: Models failed to load**")
    
    with gr.Tab("Image Classification"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                classify_btn = gr.Button("Classify Image")
            with gr.Column():
                classification_output = gr.Textbox(label="Classification Result", lines=3)
        
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=classification_output,
            api_name="classify_image"
        )
    
    with gr.Tab("Feature Extraction"):
        gr.Markdown("Extract feature vectors from images for similarity matching")
        with gr.Row():
            with gr.Column():
                feature_image_input = gr.Image(type="pil", label="Upload Image")
                extract_btn = gr.Button("Extract Features")
            with gr.Column():
                feature_output = gr.Textbox(label="Feature Extraction Result (JSON)", lines=10)
        
        extract_btn.click(
            fn=extract_features_api,
            inputs=feature_image_input,
            outputs=feature_output,
            api_name="extract_features_api"
        )
    
    with gr.Tab("Image Similarity"):
        with gr.Row():
            with gr.Column():
                image1_input = gr.Image(type="pil", label="Upload First Image")
                image2_input = gr.Image(type="pil", label="Upload Second Image")
                compare_btn = gr.Button("Compare Images")
            with gr.Column():
                similarity_output = gr.Textbox(label="Similarity Result", lines=3)
        
        compare_btn.click(
            fn=compare_images,
            inputs=[image1_input, image2_input],
            outputs=similarity_output,
            api_name="compare_images"
        )
    
    with gr.Tab("Batch Feature Comparison"):
        gr.Markdown("Compare one set of features against multiple database features")
        with gr.Row():
            with gr.Column():
                query_features_input = gr.Textbox(
                    label="Query Features (JSON array)", 
                    placeholder='[0.1, 0.2, 0.3, ...]'
                )
                db_features_input = gr.Textbox(
                    label="Database Features (JSON array of arrays)", 
                    placeholder='[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]'
                )
                batch_compare_btn = gr.Button("Compare Features")
            with gr.Column():
                batch_output = gr.Textbox(label="Batch Comparison Result (JSON)", lines=10)
        
        batch_compare_btn.click(
            fn=batch_compare_features,
            inputs=[query_features_input, db_features_input],
            outputs=batch_output,
            api_name="batch_compare_features"
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
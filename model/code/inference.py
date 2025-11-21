import os
import json
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

CLASS_LABELS = ['Buffalo', 'Cow', 'Goat', 'Sheep']

def model_fn(model_dir):
    """
    Load the model for inference
    """
    # The name of the model file is hardcoded here based on app.py.
    # Ensure 'inceptionNetV3_50e_v2v3_v1_final_TRIAL2.h5' is in the model archive.
    model_file = os.path.join(model_dir, 'inceptionNetV3_50e_v2v3_v1_final_TRIAL2.h5')
    model = load_model(model_file)
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess the input data.
    """
    if request_content_type in ['image/jpeg', 'image/png', 'application/octet-stream']:
        try:
            img = Image.open(io.BytesIO(request_body))
            
            # Reusing preprocessing logic from app.py for InceptionV3
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            target_size = (299, 299)
            aspect_ratio = img.width / img.height
            
            if aspect_ratio > 1:
                new_width = target_size[0]
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_size[1]
                new_width = int(new_height * aspect_ratio)
                
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            x = image.img_to_array(new_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            return x
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Run prediction on the preprocessed data.
    """
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    """
    if response_content_type == 'application/json':
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = CLASS_LABELS[predicted_class_index]
        confidence = float(np.max(prediction[0]))
        
        response = {
            'prediction': predicted_class,
            'confidence': confidence
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import base64

app = Flask(__name__)

# Configuration
CLASSES = ['A', 'B', 'C', 'Hello', 'Help', 'Thank_You']
IMG_SIZE = 128
MODEL_PATH = 'isl_cnn_model.h5'

# Global variables
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model '{MODEL_PATH}' not found. Please train first.")

# Initialize model
load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 500
        
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided.'}), 400
            
        # Extract base64 image data (remove 'data:image/jpeg;base64,' prefix)
        image_data = data['image'].split(',')[1]
        
        # Decode base64 to image array
        img_bytes = base64.b64decode(image_data)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # The frontend sends the entire frame. We need to extract the ROI.
        # Ensure these coordinates match where the user puts their hand in the UI.
        # Frontend canvas is sending a 640x480 image (typical webcam).
        # We'll use the same ROI: (300, 100) to (550, 350)
        # Note: If frontend video size differs, this might crop incorrectly. 
        # But for standard 640x480, this works.
        roi = frame[100:350, 300:550]
        
        if roi.size == 0:
            return jsonify({'prediction': 'Waiting...', 'confidence': '0%'}), 200

        # Preprocess ROI
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blur_roi, 50, 150)
        resized_roi = cv2.resize(edges, (IMG_SIZE, IMG_SIZE))
        
        # Normalize & Predict
        normalized_roi = resized_roi.astype('float32') / 255.0
        reshaped_roi = np.reshape(normalized_roi, (1, IMG_SIZE, IMG_SIZE, 1))
        
        prediction = model.predict(reshaped_roi, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
                prediction = model.predict(reshaped_roi, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        predicted_class = CLASSES[predicted_class_idx]
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence * 100:.1f}%"
        })
        
    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({'error': str(e)}), 500


       

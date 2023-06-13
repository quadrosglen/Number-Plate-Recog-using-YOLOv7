import streamlit as st
import cv2
import requests
import numpy as np
from flask import Flask, request, jsonify

URL = 'https://inf-76370045-724e-413b-960f-6e28fa989274-no4xvrhsfq-uc.a.run.app/detect'  # Theos API URL
OCR_MODEL = 'large'
OCR_CLASS = 'license-plate'
FOLDER_PATH = 'license-plates'
seconds_to_wait = 2

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def process_upload():
    uploaded_file = request.files['image']
    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Perform license plate detection
        license_plates = detect_license_plates(image)

        if license_plates:
            # Return the detected license plates as the API response
            return jsonify({"license_plates": license_plates})
        else:
            return jsonify({"message": "No license plates detected."})
    else:
        return jsonify({"message": "No image file received."})

def detect_license_plates(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    image_bytes = img_encoded.tobytes()

    response = requests.post(URL, data={'ocr_model': OCR_MODEL, 'ocr_classes': OCR_CLASS}, files={'image': image_bytes})

    if response.status_code == 200:
        data = response.json()
        license_plates = []

        for detection in data:
            if detection['class'] == OCR_CLASS and detection['text']:
                license_plates.append(detection['text'].upper())

        return license_plates

    return []

if __name__ == "__main__":
    app.run()

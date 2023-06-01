import os
os.environ['DISPLAY'] = ':0'

import streamlit as st
import cv2
import requests
import numpy as np

URL = 'https://inf-76370045-724e-413b-960f-6e28fa989274-no4xvrhsfq-uc.a.run.app/detect'  # Theos API URL
OCR_MODEL = 'large'
OCR_CLASS = 'license-plate'
FOLDER_PATH = 'license-plates'
seconds_to_wait = 2

def main():
    st.title("License Plate Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, channels="BGR")

        # Perform license plate detection
        license_plates = detect_license_plates(image)

        # Display the detected license plates
        if license_plates:
            st.write("Detected license plates:")
            st.write(", ".join(license_plates))
        else:
            st.write("No license plates detected.")

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
    main()

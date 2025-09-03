import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained CNN model
model = load_model("CNNmodel.h5")

# Define class labels (change based on your dataset)
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("🤟 Sign Language Detection")
st.write("Upload an image or use webcam to detect the sign")

# Upload image option
uploaded_file = st.file_uploader("Upload a sign image", type=["jpg", "jpeg", "png"])

# Webcam capture
camera_option = st.checkbox("Use Webcam")

if uploaded_file is not None:
    # Process uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img = image.resize((64, 64))  # Adjust to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.subheader(f"Prediction: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")

elif camera_option:
    st.write("Click 'Start' to capture image")
    camera_input = st.camera_input("Take a picture")

    if camera_input:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        img = image.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.subheader(f"Prediction: {class_labels[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}")

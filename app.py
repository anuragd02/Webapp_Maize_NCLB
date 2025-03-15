import streamlit as st

# Set the page configuration at the very beginning
st.set_page_config(layout="wide", page_title="Maize Leaf Disease Classification (NCLB)")

import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive file ID for the maize NCLB disease model
MODEL_FILE_ID = "168SpEP2L1X8DgItbqbuot0rD9kvlJJjT"
MODEL_PATH = "nclb_vgg_net16.h5"

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):  # Download only if the model is not already present
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = download_and_load_model()

# Define class labels
CLASS_NAMES = ["Unhealthy", "Healthy"]

st.title("Maize Leaf Disease Classification Dashboard (NCLB)")

# Creating two columns for a split-screen layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        img = image.resize((224, 224))  # Ensure this matches the model's expected input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence >= 50:
            final_prediction = predicted_class + ": Northern Corn Leaf Blight Disease"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            final_prediction = "Healthy"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {100 - confidence:.2f}%")


with col2:
    st.subheader("Northern Corn Leaf Blight (NCLB)")
    st.markdown(
        """
        **Pathogen:**
        - Caused by *Exserohilum turcicum*.
        - Characterized by long, elliptical, grayish-green lesions on leaves.

        **Management Strategies:**
        - Use resistant maize hybrids.
        - Apply fungicides like *Azoxystrobin* or *Propiconazole*.
        - Ensure proper crop rotation and field sanitation.
        """
    )

st.markdown("---")
st.write("Developed by Anurag using Streamlit❤️")

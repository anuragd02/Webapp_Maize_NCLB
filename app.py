import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Page configuration
st.set_page_config(layout="wide",
                   page_title="Maize Leaf Disease Classification (NCLB)",
                   page_icon="")

# Google Drive model info
MODEL_FILE_ID = "168SpEP2L1X8DgItbqbuot0rD9kvlJJjT"
MODEL_PATH = "nclb_vgg_net16.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = download_and_load_model()

CLASS_NAMES = ["Unhealthy", "Healthy"]

# App Title
st.title("Maize Leaf Disease Classification Dashboard (NCLB)")

# Two columns: left = prediction, right = advisory
col1, col2 = st.columns([1, 1])

# --- Left Column : Upload & Prediction ---
with col1:
    st.subheader("📸 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess & Predict
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence >= 50:
            final_prediction = predicted_class + ": Northern Corn Leaf Blight Disease"
            st.success(f"**Prediction:** {final_prediction}")
            st.info(f"**Confidence:** {confidence:.2f}%")
        else:
            st.success("**Prediction:** Healthy")
            st.info(f"**Confidence:** {100 - confidence:.2f}%")

# --- Right Column : ONLY Fungicide Advisory ---
with col2:
    st.subheader("Northern Corn Leaf Blight – Fungicide Advisory")

    st.markdown(
        """
        **General Guidelines**  
        * Monitor fields from knee-high stage to tasseling; NCLB spreads fast in humid 20–27 °C conditions.  
        * Make spray decisions based on **disease severity** and crop stage.  
        * Ensure **500 L water/ha** spray coverage with a knapsack or motorized sprayer.
        """
    )

    st.markdown("#### Severity-Based Spraying")
    st.table({
        "Disease Severity (PDI)": ["Low (≤10%)", "Moderate (10–20%)", "High (>20%)"],
        "Crop Stage": [
            "Knee-high (30–35 DAS)",
            "Pre-tasseling (45–55 DAS)",
            "Tasseling–grain filling (60–80 DAS)"
        ],
        "Spray Recommendation": [
            "Preventive: Carbendazim + Mancozeb OR Zineb",
            "Azoxystrobin + Cyproconazole OR Pyraclostrobin + Epoxiconazole",
            "Azoxystrobin + Difenoconazole (repeat after 15–20 days if needed)"
        ]
    })

    st.markdown(
        """
        **Advisory Highlights**  
        * **First Spray:** At first symptoms or knee-high stage.  
        * **Second Spray:** Pre-tasseling (45–55 DAS), depending on severity.  
        * **Third Spray:** Tasseling to grain-filling stage (60–80 DAS) under humid conditions.  
        * **Resistance Management:** Rotate fungicides with different FRAC codes to avoid resistance.  
        * **Most Effective:** **Azoxystrobin + Difenoconazole**; good alternatives include **Azoxystrobin + Cyproconazole** and **Pyraclostrobin + Epoxiconazole**
        """
    )

st.markdown("---")
st.markdown(
    """
    <style>
    .developed-by {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .person {
        font-size: 16px;
        margin-bottom: 2px;
    }
    </style>
    <div class="developed-by">Developed by</div>
    <div class="person"><b>Anurag Dhole</b> - Researcher at MIT, Manipal</div>
    <div class="person"><b>Dr. Jadesha G</b> - Assistant Professor at GKVK, UAS, Bangalore</div>
    <div class="person"><b>Dr. Deepak D.</b> - Professor at MIT, Manipal</div>
    """,
    unsafe_allow_html=True
)


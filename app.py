# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# ===== CONFIG =====
IMG_HEIGHT = 64
IMG_WIDTH = 64
RESCALE_FACTOR = 1.0 / 255.0

CLASS_DETAILS = {
    "AnnualCrop": {"icon": "🌾", "desc": "Annual Crop Land", "color": "#FFD700"},
    "Forest": {"icon": "🌲", "desc": "Forest Terrain", "color": "#228B22"},
    "HerbaceousVegetation": {"icon": "🌿", "desc": "Herbaceous Vegetation", "color": "#90EE90"},
    "Highway": {"icon": "🛣️", "desc": "Highway / Road", "color": "#696969"},
    "Industrial": {"icon": "🏭", "desc": "Industrial Area", "color": "#A0522D"},
    "Pasture": {"icon": "🐄", "desc": "Pastureland", "color": "#9AFF9A"},
    "PermanentCrop": {"icon": "🍇", "desc": "Permanent Crop Zone", "color": "#DDA0DD"},
    "Residential": {"icon": "🏠", "desc": "Residential Area", "color": "#FFB6C1"},
    "River": {"icon": "🏞️", "desc": "River/Water Channel", "color": "#87CEEB"},
    "SeaLake": {"icon": "🌊", "desc": "Sea or Lake", "color": "#4169E1"},
}

CLASS_NAMES = list(CLASS_DETAILS.keys())

# Local model path
LOCAL_MODEL_PATH = r"C:\Users\91989\OneDrive\Desktop\LAND_COVER_CLASSIFICATION\lulc_2_epoch"

# ===== UTILITY FUNCTIONS =====
@st.cache_resource
def load_model_from_local():
    """Load model from local folder."""
    try:
        tf.keras.backend.clear_session()
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"Local model folder not found: {LOCAL_MODEL_PATH}")
        
        model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
        return model, None
    except Exception as e:
        return None, f"❌ Model loading failed: {str(e)}"


def preprocess_image(image: Image.Image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) * RESCALE_FACTOR
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def get_top_predictions(predictions, top_k=3):
    pred_probs = predictions[0]
    top_indices = np.argsort(pred_probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        class_name = CLASS_NAMES[idx]
        confidence = float(pred_probs[idx])
        results.append({
            'class': class_name,
            'confidence': confidence,
            'details': CLASS_DETAILS[class_name]
        })
    return results

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="HRRS Land Cover Classifier", 
    page_icon="🌍", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===== MAIN APP =====
def main():
    st.markdown('<h1 style="text-align:center;">🌍 HRRS Land Cover Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-style:italic;">Upload your satellite image for instant classification!</p>', unsafe_allow_html=True)
    
    # Load model from local folder
    model, error = load_model_from_local()
    if error:
        st.error(error)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 System Information")
        st.info(f"**TensorFlow:** {tf.__version__}")
        st.info(f"**Streamlit:** {st.__version__}")
        st.info(f"**Model Classes:** {len(CLASS_NAMES)}")
        st.info(f"**Input Size:** {IMG_WIDTH}×{IMG_HEIGHT} pixels")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Your Satellite Image", 
        type=["jpg", "jpeg", "png", "tiff", "tif"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Satellite Image", use_container_width=True)
        
        processed_image, prep_error = preprocess_image(image)
        if prep_error:
            st.error(prep_error)
            st.stop()
        
        if st.button("🚀 Analyze Land Cover"):
            predictions = model.predict(processed_image, verbose=0)
            top_predictions = get_top_predictions(predictions, top_k=3)
            
            top_pred = top_predictions[0]
            confidence_pct = top_pred['confidence'] * 100
            st.subheader(f"🎯 Prediction: {top_pred['class'].replace('_', ' ')} ({confidence_pct:.1f}%)")
            st.write(top_pred['details']['desc'])
            
            if len(top_predictions) > 1:
                st.subheader("🔍 Alternative Classifications")
                for pred in top_predictions[1:]:
                    st.write(f"{pred['class'].replace('_', ' ')}: {pred['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()

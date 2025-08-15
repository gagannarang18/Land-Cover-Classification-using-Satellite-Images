# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
from pathlib import Path
import gdown
import shutil

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

# Google Drive Configuration - Your exact folder
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Hp2pF1OtUazdGGRM9cywQbJYLmDLAWs4"
MODEL_FOLDER_NAME = "lulc_2_epoch"
TEMP_MODEL_PATH = f"./temp_{MODEL_FOLDER_NAME}"

# ===== UTILITY FUNCTIONS =====
@st.cache_resource
def download_and_load_model_from_gdrive():
    """Download model directly from Google Drive and load it."""
    try:
        # Close any previous TF sessions before deleting old temp folder
        tf.keras.backend.clear_session()

        if os.path.exists(TEMP_MODEL_PATH):
            try:
                shutil.rmtree(TEMP_MODEL_PATH)
            except PermissionError:
                # Ignore locked files (Windows/OneDrive issue)
                print(f"[WARNING] Could not delete some files in {TEMP_MODEL_PATH}, skipping locked files...")
                shutil.rmtree(TEMP_MODEL_PATH, ignore_errors=True)

        with st.status("📥 Loading model from Google Drive...", expanded=True) as status:
            st.write("🔗 Connecting to Google Drive...")
            time.sleep(0.5)
            st.write("⬇️ Downloading model files...")

            # Download from Google Drive
            gdown.download_folder(
                url=GOOGLE_DRIVE_FOLDER_URL,
                output=TEMP_MODEL_PATH,
                quiet=False,
                use_cookies=False
            )

            # Detect model path
            model_path = None
            if os.path.exists(os.path.join(TEMP_MODEL_PATH, MODEL_FOLDER_NAME)):
                model_path = os.path.join(TEMP_MODEL_PATH, MODEL_FOLDER_NAME)
            elif any(f.endswith('.pb') or f == 'saved_model.pb' for f in os.listdir(TEMP_MODEL_PATH)):
                model_path = TEMP_MODEL_PATH
            else:
                for item in os.listdir(TEMP_MODEL_PATH):
                    item_path = os.path.join(TEMP_MODEL_PATH, item)
                    if os.path.isdir(item_path) and any(
                        f.endswith('.pb') or f == 'saved_model.pb' for f in os.listdir(item_path)
                    ):
                        model_path = item_path
                        break

            if not model_path:
                raise Exception("Could not locate TensorFlow model files in downloaded folder")

            st.write("🧠 Loading TensorFlow model...")
            model = tf.keras.models.load_model(model_path, compile=False)
            st.write("✅ Model loaded successfully!")
            status.update(label="✅ Model ready from Google Drive!", state="complete", expanded=False)

            return model, None

    except Exception as e:
        # Cleanup on error
        tf.keras.backend.clear_session()
        if os.path.exists(TEMP_MODEL_PATH):
            shutil.rmtree(TEMP_MODEL_PATH, ignore_errors=True)
        return None, f"❌ Model loading failed: {str(e)}"


def preprocess_image(image: Image.Image):
    """Preprocess the uploaded image for model prediction."""
    try:
        # Convert to RGB if needed (PIL 11.3.0 compatible)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        
        # Convert to numpy array (compatible with numpy>=1.26.0)
        img_array = np.array(image, dtype=np.float32) * RESCALE_FACTOR
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def get_top_predictions(predictions, top_k=3):
    """Get top K predictions with confidence scores."""
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

# ===== CUSTOM CSS =====
custom_css = """
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .header {
        font-size: 42px !important;
        font-weight: bold !important;
        color: #2E86AB !important;
        text-align: center;
        margin: 0 0 15px 0;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subheader {
        font-size: 18px !important;
        color: #367588 !important;
        text-align: center;
        margin-bottom: 25px;
        font-style: italic;
        line-height: 1.4;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-box {
        border: 3px dashed #ffffff;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 40px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        color: white;
    }
    
    .result-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        color: white;
    }
    
    .prediction-card {
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #00d4aa, #00d4ff);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .big-icon {
        font-size: 48px;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .class-title {
        font-size: 24px !important;
        font-weight: 600;
        margin: 10px 0 5px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .class-desc {
        font-size: 16px !important;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    
    .confidence-text {
        font-size: 18px !important;
        font-weight: 500;
        margin: 5px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #ffffff;
    }
    
    .drive-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideIn 0.6s ease-out;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ===== MAIN APP =====
def main():
    # Header
    st.markdown('<div class="header">🌍 HRRS Land Cover Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Advanced AI-powered satellite image analysis • Model streamed directly from Google Drive • Upload your satellite image for instant classification!</div>', unsafe_allow_html=True)
    
    # Display Google Drive connection info
    st.markdown('<div class="drive-info">', unsafe_allow_html=True)
    st.markdown("☁️ **Model Source:** Google Drive Cloud Storage")
    st.markdown(f"📁 **Folder:** {MODEL_FOLDER_NAME}")
    st.markdown("🔄 **Status:** Real-time download and processing")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model directly from Google Drive
    model, error = download_and_load_model_from_gdrive()
    
    if error:
        st.error(error)
        
        with st.expander("🔧 Troubleshooting Guide"):
            st.markdown("""
            **Common Solutions:**
            
            1. **Google Drive Access Issues:**
               - Ensure your folder link is publicly accessible
               - Right-click folder → Share → "Anyone with the link can view"
            
            2. **Package Issues:**
               ```
               pip install --upgrade gdown
               pip install tensorflow==2.20.0 streamlit==1.48.1
               ```
            
            3. **Network Issues:**
               - Check internet connection
               - Try refreshing the page
               - Some networks may block Google Drive access
            
            4. **Model Format Issues:**
               - Ensure your model is in TensorFlow SavedModel format
               - Check that all model files are present in the folder
            """)
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("🤖 System Information")
        st.info(f"**TensorFlow:** {tf.__version__}")
        st.info(f"**Streamlit:** {st.__version__}")
        st.info(f"**Model Classes:** {len(CLASS_NAMES)}")
        st.info(f"**Input Size:** {IMG_WIDTH}×{IMG_HEIGHT} pixels")
        
        st.header("🎯 Land Cover Classes")
        for class_name, details in CLASS_DETAILS.items():
            st.write(f"{details['icon']} **{class_name.replace('_', ' ')}**")
            st.caption(details['desc'])
        
        st.header("☁️ Google Drive Features")
        st.success("✅ Direct cloud loading")
        st.success("✅ No local storage needed")
        st.success("✅ Always up-to-date model")
        st.success("✅ Automatic error handling")
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Upload Your Satellite Image", 
        type=["jpg", "jpeg", "png", "tiff", "tif"], 
        help="Supported formats: JPG, JPEG, PNG, TIFF • Recommended: High-resolution satellite imagery"
    )
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Satellite Image", use_container_width=True)
        
        # Image metadata
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write(f"**📏 Dimensions:** {image.size[0]} × {image.size[1]} pixels")
        st.write(f"**🎨 Color Mode:** {image.mode}")
        st.write(f"**💾 File Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
        st.write(f"**📊 Format:** {image.format if image.format else 'Unknown'}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🚀 Analyze Land Cover", type="primary", use_container_width=True):
            with st.status("🔍 Analyzing satellite imagery...", expanded=True) as status:
                st.write("🔄 Preprocessing image for AI analysis...")
                time.sleep(0.5)
                
                # Preprocess image
                processed_image, prep_error = preprocess_image(image)
                if prep_error:
                    st.error(f"Image preprocessing failed: {prep_error}")
                    st.stop()
                
                st.write("🧠 Running deep learning classification...")
                time.sleep(0.5)
                
                # Make prediction using TensorFlow 2.20.0
                try:
                    predictions = model.predict(processed_image, verbose=0)
                    top_predictions = get_top_predictions(predictions, top_k=3)
                    
                    st.write("✅ Classification analysis complete!")
                    status.update(label="🎯 Land cover identified successfully!", state="complete", expanded=False)
                    
                except Exception as e:
                    st.error(f"AI prediction failed: {str(e)}")
                    st.stop()
            
            # Display results with animation
            st.markdown('<div class="result-container animate-in">', unsafe_allow_html=True)
            
            # Primary prediction result
            top_pred = top_predictions[0]
            confidence_pct = top_pred['confidence'] * 100
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f'<div class="big-icon">{top_pred["details"]["icon"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="class-title">{top_pred["class"].replace("_", " ")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="class-desc">{top_pred["details"]["desc"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-text">Confidence: {confidence_pct:.1f}%</div>', unsafe_allow_html=True)
                
                # Animated confidence bar
                st.markdown(f'''
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alternative predictions
            if len(top_predictions) > 1:
                st.subheader("🔍 Alternative Classifications")
                
                for pred in top_predictions[1:]:
                    confidence_pct = pred['confidence'] * 100
                    
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.markdown(f"<div style='font-size: 24px;'>{pred['details']['icon']}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.write(f"**{pred['class'].replace('_', ' ')}**")
                        st.caption(pred['details']['desc'])
                    
                    with col3:
                        st.write(f"**{confidence_pct:.1f}%**")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Contextual environmental insights
            predicted_class = top_pred['class']
            if predicted_class == "Forest":
                st.success("🌳 **Forest Ecosystem Detected!** This area is crucial for carbon sequestration, biodiversity conservation, and climate regulation. Important for environmental sustainability.")
            elif predicted_class == "SeaLake":
                st.info("🌊 **Water Body Identified!** Critical for local hydrology, ecosystem services, water resource management, and biodiversity support.")
            elif predicted_class == "Residential":
                st.warning("🏘️ **Urban Residential Area!** Important for urban planning, population density analysis, and sustainable development monitoring.")
            elif predicted_class == "Industrial":
                st.info("🏭 **Industrial Zone Detected!** Key area for economic analysis, environmental impact assessment, and emission monitoring.")
            elif predicted_class == "Highway":
                st.info("🛣️ **Transportation Infrastructure!** Critical corridor for connectivity, economic development, and logistics planning.")
            elif predicted_class == "AnnualCrop":
                st.success("🌾 **Agricultural Land Detected!** Essential for food security, rural economy, and sustainable farming practice analysis.")
            elif predicted_class == "PermanentCrop":
                st.success("🍇 **Permanent Crop Zone!** Sustainable agricultural area important for long-term food production and soil conservation.")

if __name__ == "__main__":
    main()

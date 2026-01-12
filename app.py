import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from model import OilSpillCNN
import io

# Page config
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

# Title and description
st.title("🛢️ Oil Spill Detection System")
st.markdown("Upload a SAR satellite image to detect oil spills using AI")

# Sidebar for model upload
st.sidebar.header("Model Configuration")
st.sidebar.info("Upload your trained model file (.pth)")

model_file = st.sidebar.file_uploader("Upload Model File", type=['pth'])

# Initialize model
@st.cache_resource
def load_model(model_file):
    model = OilSpillCNN()
    model_bytes = model_file.read()
    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 128x128
    image = cv2.resize(image, (128, 128))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image

# Main content
if model_file is not None:
    model = load_model(model_file)
    st.sidebar.success("✅ Model loaded successfully!")
    
    # File uploader for images
    st.header("Upload SAR Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Make prediction
        if st.button("🔍 Detect Oil Spill", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess
                image_np = np.array(image)
                processed_image = preprocess_image(image_np)
                
                # Predict
                with torch.no_grad():
                    output = model(processed_image)
                    probability = output.item()
                
                # Display results
                with col2:
                    st.subheader("Detection Result")
                    
                    if probability > 0.5:
                        st.error(f"🛢️ **Oil Spill Detected!**")
                        st.metric("Confidence", f"{probability*100:.2f}%")
                    else:
                        st.success(f"✅ **No Oil Spill Detected**")
                        st.metric("Confidence", f"{(1-probability)*100:.2f}%")
                    
                    # Progress bar
                    st.progress(probability)
else:
    st.warning("⚠️ Please upload a trained model file (.pth) in the sidebar to begin.")
    st.info("""
    ### Instructions:
    1. Upload your trained model file using the sidebar
    2. Upload a SAR satellite image
    3. Click 'Detect Oil Spill' to get predictions
    """)

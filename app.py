import streamlit as st
import torch
import cv2
import numpy as np
from model import OilSpillCNN


st.set_page_config(page_title="Oil Spill Detection")


st.title("Oil Spill Detection using SAR Images")


# Load model
model = OilSpillCNN()
model.load_state_dict(torch.load("oil_spill_model.pth", map_location=torch.device('cpu')))
model.eval()


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)


uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    st.image(image, caption="Uploaded Image", use_column_width=True)


    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)


    if output.item() > 0.5:
        st.success("Prediction: Oil Spill Detected")
    else:
        st.success("Prediction: No Oil Spill Detected")

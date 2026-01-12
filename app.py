import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np

# ---------------- UI ----------------
st.title("Oil Spill Detection using SAR Images")
st.write("Upload a SAR image to detect oil spill")

# ---------------- MODEL DEFINITION ----------------
# ⚠️ COPY THIS CLASS EXACTLY FROM Final_Training.ipynb
class OilSpillCNN(nn.Module):
    def __init__(self):
        super(OilSpillCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))
        return x

# ---------------- LOAD MODEL ----------------
model = OilSpillCNN()
model.load_state_dict(
    torch.load("oil_spill_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload SAR Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(img_tensor)

    if output.item() > 0.5:
        st.success("Prediction: Oil Spill Detected")
    else:
        st.success("Prediction: No Oil Spill Detected")

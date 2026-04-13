import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download

# -------------------------------
# MODEL DOWNLOAD CONFIG
# -------------------------------
MODEL_PATH = "best_model.pth"
HF_REPO_ID = "aditya-kumar04/breed-identification"

def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_PATH,
                local_dir="."
            )
    return MODEL_PATH

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Buffalo AI",
    page_icon="🐃",
    layout="wide"
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Settings")
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image to identify buffalo breed.")

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# CLASSES
# -------------------------------
classes = [
    "Jaffrabadi", "Mehsana", "Murrah",
    "Nagori", "Red_Sindhi", "Sahiwal",
    "Surti", "Tharparkar", "Toda"
]

# -------------------------------
# BREED INFO
# -------------------------------
breed_info = {
    "Murrah": "High milk yield, black coat, Haryana",
    "Sahiwal": "Heat tolerant, reddish-brown, Punjab",
    "Jaffrabadi": "Heavy build, curved horns, Gujarat",
    "Mehsana": "Hybrid dairy breed",
}

# -------------------------------
# LOAD MODEL (FIXED)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = get_model()  # ✅ correct usage

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 9)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# GRAD-CAM
# -------------------------------
def generate_gradcam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]

    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_score = output[0, target_class]

    model.zero_grad()
    class_score.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    fwd.remove()
    bwd.remove()

    return cam

def overlay_heatmap(image, cam):
    image = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay = heatmap + np.float32(image) / 255
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)

# -------------------------------
# PREDICTION
# -------------------------------
def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    confidence, pred = torch.max(probs, 1)

    return {
        "class_idx": pred.item(),
        "prediction": classes[pred.item()],
        "confidence": float(confidence.item()),
        "probs": probs,
        "tensor": image_tensor
    }

# -------------------------------
# UI
# -------------------------------
st.title("🐃 Buffalo Breed Identification System")
st.markdown("Upload an image to classify the breed using AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            result = predict(image)

        st.subheader("Prediction Result")
        st.success(result["prediction"])

        st.progress(result["confidence"])
        st.write(f"Confidence: {result['confidence']:.4f}")

        if result["prediction"] in breed_info:
            st.info(breed_info[result["prediction"]])

        if result["confidence"] > 0.95:
            st.warning("⚠️ Model may be overconfident")

    # ---------------------------
    # Grad-CAM
    # ---------------------------
    if show_gradcam:
        st.markdown("---")
        st.subheader("🔍 Model Attention (Grad-CAM)")

        cam = generate_gradcam(
            model,
            result["tensor"],
            result["class_idx"]
        )

        heatmap = overlay_heatmap(image, cam)

        col3, col4 = st.columns(2)

        with col3:
            st.image(image, caption="Original")

        with col4:
            st.image(heatmap, caption="Model Focus")
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

# ---------------------------
# LOAD MODEL
# ---------------------------
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 14)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()

# ---------------------------
# CLASS NAMES
# ---------------------------
CLASS_NAMES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# UI
# ---------------------------
st.title("🩺 Chest X-ray Disease Predictor")

uploaded_file = st.file_uploader("Upload X-ray Image")

# ---------------------------
# PREDICTION BLOCK
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output)[0]

    # ---------------------------
    # TOP PREDICTIONS
    # ---------------------------
    st.subheader("🩺 Detected Conditions (Top Predictions)")

    found = False
    for i, prob in enumerate(probs):
        if prob > 0.1:
            st.write(f"🔹 {CLASS_NAMES[i]}: {prob:.4f}")
            found = True

    if not found:
        st.write("✅ No strong disease detected")

    # ---------------------------
    # CONFIDENCE BARS (PRO UI)
    # ---------------------------
    st.subheader("📊 Prediction Confidence")

    for i, prob in enumerate(probs):
        st.progress(float(prob))
        st.write(f"{CLASS_NAMES[i]}: {prob:.4f}")
import torch
from PIL import Image
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = "data/raw/images/00000001_000.png"  # change if needed

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# LOAD MODEL
# ---------------------------
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 14)

model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------
# LOAD IMAGE
# ---------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# ---------------------------
# PREDICT
# ---------------------------
with torch.no_grad():
    outputs = model(image)
    probs = torch.sigmoid(outputs).cpu().numpy()[0]

# ---------------------------
# PRINT RESULTS
# ---------------------------
print("\n🔍 Prediction Results:\n")

for i, prob in enumerate(probs):
    print(f"{CLASS_NAMES[i]}: {prob:.4f}")

print("\n✅ Done")
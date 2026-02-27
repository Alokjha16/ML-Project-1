import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# =========================
# CONFIG
# =========================
IMAGE_SIZE = (252, 448)   # ⚠️ SAME AS TRAINING
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

model_path = "latest_checkpoint.pth"
test_image_dir = r"C:\Users\alokj\OneDrive\Desktop\Falcon_Hackathon\Offroad_Segmentation_Training_Dataset\train\Color_Images"
save_dir = "submissions/FinalVariant"

os.makedirs(save_dir, exist_ok=True)

# =========================
# LOAD BACKBONE (DINOv2)
# =========================
print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval()

# =========================
# MODEL
# =========================
class DinoSegmentationModel(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        x = features["x_norm_patchtokens"]

        B, N, C = x.shape

        # Auto patch grid detection
        H = int(np.sqrt(N))
        W = H

        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = torch.nn.functional.interpolate(
            x,
            size=IMAGE_SIZE,
            mode="bilinear",
            align_corners=False
        )

        x = self.decoder(x)
        return x

# =========================
# LOAD MODEL
# =========================
model = DinoSegmentationModel(backbone, NUM_CLASSES).to(DEVICE)

print("Loading checkpoint...")
checkpoint = torch.load(model_path, map_location=DEVICE)
print("Checkpoint keys:", checkpoint.keys())

# 🔥 REMOVE "decoder." PREFIX
clean_state_dict = {}
for k, v in checkpoint["model"].items():
    if k.startswith("decoder."):
        new_key = k.replace("decoder.", "")
        clean_state_dict[new_key] = v

# Load decoder weights correctly
model.decoder.load_state_dict(clean_state_dict)

model.eval()
print("Model loaded successfully!")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# =========================
# GENERATE PREDICTIONS
# =========================
print("Generating Predictions...")

if not os.path.exists(test_image_dir):
    raise FileNotFoundError(f"Folder not found: {test_image_dir}")

for filename in os.listdir(test_image_dir):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(test_image_dir, filename)
    image = Image.open(img_path).convert("RGB")
    original_width, original_height = image.size

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.argmax(output, dim=1)

        pred_mask = output.squeeze(0).cpu().numpy().astype(np.uint8)

        pred_mask = cv2.resize(
            pred_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, pred_mask)

print("✅ Submission Generated Successfully!")
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Model Definition (MATCH TRAINING)
# ===============================
class DinoSegmentationModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        print("Loading DINOv2 backbone...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14"
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Decoder (must match training architecture)
        self.decoder = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        x = features["x_norm_patchtokens"]

        B, N, C = x.shape
        H = W = int(np.sqrt(N))

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear")
        return x


# ===============================
# Load Model
# ===============================
model = DinoSegmentationModel(num_classes=10).to(device)

checkpoint = torch.load("best_model_epoch5.pth", map_location=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.eval()
print("Checkpoint loaded successfully with strict=False")

# ===============================
# Load Image Automatically
# ===============================
folder_name = "visual_results"   # change if needed

image_list = glob.glob(os.path.join(folder_name, "*.png"))

if len(image_list) == 0:
    print("\nDEBUG INFO:")
    print("Current directory:", os.getcwd())
    print("Available files/folders:", os.listdir())
    raise Exception(f"No PNG images found inside '{folder_name}'")

IMAGE_PATH = image_list[0]
print("Using image:", IMAGE_PATH)

# ===============================
# Preprocess Image
# ===============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_tensor = transform(image).unsqueeze(0).to(device)

# ===============================
# Inference
# ===============================
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# ===============================
# Resize Prediction to Original Size
# ===============================
h, w, _ = image.shape

prediction_resized = cv2.resize(
    prediction.astype(np.uint8),
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ===============================
# Create Colored Mask
# ===============================
colored_mask = cv2.applyColorMap(
    (prediction_resized * 25).astype(np.uint8),
    cv2.COLORMAP_JET
)

# ===============================
# Overlay
# ===============================
overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

# ===============================
# Visualization
# ===============================
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(prediction_resized, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
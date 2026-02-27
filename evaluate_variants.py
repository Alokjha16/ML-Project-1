import torch
import torch.nn.functional as F
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from PIL import Image
import cv2

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "latest_checkpoint.pth"

IMAGE_WIDTH = 448
IMAGE_HEIGHT = 252

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

NUM_CLASSES = len(value_map)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Mask Conversion
# ============================================================

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

# ============================================================
# IoU Function
# ============================================================

def compute_iou(pred, target, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious) if len(ious) > 0 else 0

# ============================================================
# Segmentation Head (SAME AS TRAINING)
# ============================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)

# ============================================================
# Load Backbone + Model
# ============================================================

print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(device)

# Get embedding size
dummy = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
with torch.no_grad():
    tokens = backbone.forward_features(dummy)["x_norm_patchtokens"]

embedding_dim = tokens.shape[2]

classifier = SegmentationHead(
    embedding_dim,
    NUM_CLASSES,
    IMAGE_WIDTH // 14,
    IMAGE_HEIGHT // 14
).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
classifier.load_state_dict(checkpoint["model"])
classifier.eval()

print("Model loaded successfully!")

# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# Prediction Function
# ============================================================

def predict(image, tta=False, smoothing=False):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = backbone.forward_features(img)["x_norm_patchtokens"]
        logits = classifier(tokens)
        logits = F.interpolate(logits, size=img.shape[2:], mode="bilinear", align_corners=False)

        # TTA
        if tta:
            flipped = torch.flip(img, dims=[3])
            tokens_flip = backbone.forward_features(flipped)["x_norm_patchtokens"]
            logits_flip = classifier(tokens_flip)
            logits_flip = F.interpolate(logits_flip, size=img.shape[2:], mode="bilinear", align_corners=False)
            logits_flip = torch.flip(logits_flip, dims=[3])
            logits = (logits + logits_flip) / 2

        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

        # Smoothing
        if smoothing:
            pred = cv2.medianBlur(pred.astype(np.uint8), 5)

    return pred

# ============================================================
# Evaluate Variants
# ============================================================

VAL_DIR = r"C:\Users\alokj\OneDrive\Desktop\Falcon_Hackathon\Offroad_Segmentation_Training_Dataset\val"

image_dir = os.path.join(VAL_DIR, "Color_Images")
mask_dir = os.path.join(VAL_DIR, "Segmentation")

variants = [
    {"name": "baseline", "tta": False, "smoothing": False},
    {"name": "tta", "tta": True, "smoothing": False},
    {"name": "smoothing", "tta": False, "smoothing": True},
    {"name": "tta_smoothing", "tta": True, "smoothing": True},
]

for variant in variants:
    print(f"\nEvaluating {variant['name']}...")
    ious = []

    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        # IMPORTANT FIX: Resize mask to match prediction size
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

        pred = predict(
            image,
            tta=variant["tta"],
            smoothing=variant["smoothing"]
        )

        iou = compute_iou(pred, mask, NUM_CLASSES)
        ious.append(iou)

    print(f"{variant['name']} IoU: {np.mean(ious):.4f}")

print("\nEvaluation complete!")
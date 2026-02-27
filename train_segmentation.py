import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from tqdm import tqdm

# =========================
# CONFIG
# =========================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 7   # IMPORTANT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

IMAGE_DIR = r"C:\Users\alokj\OneDrive\Desktop\Falcon_Hackathon\Offroad_Segmentation_Training_Dataset\train\Color_Images"
MASK_DIR = r"C:\Users\alokj\OneDrive\Desktop\Falcon_Hackathon\Offroad_Segmentation_Training_Dataset\train\Segmentation"

# Fixed label mapping
label_map = {
    200: 0,
    300: 1,
    500: 2,
    550: 3,
    800: 4,
    7100: 5,
    10000: 6
}

# =========================
# DATASET
# =========================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        mask = Image.open(mask_path)
        mask = mask.resize(IMAGE_SIZE, Image.NEAREST)
        mask = np.array(mask)

        # Convert label values safely
        new_mask = np.zeros_like(mask)

        for k in label_map:
            new_mask[mask == k] = label_map[k]

        mask = torch.tensor(new_mask, dtype=torch.long)

        return image, mask


# =========================
# MODEL (DINOv2 + Seg Head)
# =========================

print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.to(DEVICE)

class DinoSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.conv = nn.Conv2d(384, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        x = features["x_norm_patchtokens"]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
        return x


model = DinoSegmentationModel(backbone, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# DATA LOADER
# =========================

dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# TRAINING LOOP
# =========================

print("Starting Training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(loader)

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "segmentation_model.pth")
print("Model saved successfully!")
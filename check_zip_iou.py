import os
import numpy as np
import cv2
from tqdm import tqdm

# ===== CHANGE THIS PATH IF NEEDED =====
GT_DIR = "../Offroad_Segmentation_Training_Dataset/val/Segmentation"

NUM_CLASSES = 10

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


def evaluate_folder(pred_dir):
    ious = []
    total_images = 0
    resized_count = 0

    for filename in tqdm(os.listdir(pred_dir)):
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(GT_DIR, filename)

        if not os.path.exists(gt_path):
            continue

        pred = cv2.imread(pred_path, 0)
        gt = cv2.imread(gt_path, 0)

        if pred is None or gt is None:
            continue

        total_images += 1

        # ✅ Resize prediction if shape mismatch
        if pred.shape != gt.shape:
            print(f"Resizing {filename}: {pred.shape} -> {gt.shape}")
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),  # width, height
                interpolation=cv2.INTER_NEAREST
            )
            resized_count += 1

        iou = compute_iou(pred, gt, NUM_CLASSES)
        ious.append(iou)

    mean_iou = np.mean(ious) if len(ious) > 0 else 0

    print(f"\nTotal images evaluated: {total_images}")
    print(f"Images resized: {resized_count}")

    return mean_iou


folders = [
    "submissions/Variant1",
    "submissions/Variant2",
    "submissions/Variant3",
    "submissions/Variant4"
]

for folder in folders:
    print(f"\nChecking {folder} ...")
    mean_iou = evaluate_folder(folder)
    print(f"{folder} Mean IoU: {mean_iou:.4f}")

print("\nDone!")
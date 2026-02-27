# 🚀 Ignitia Hackathon – Baseline Variant

## 📌 Project Overview
This repository contains the **Baseline implementation** of the DINOv2-based semantic segmentation model.

The model uses:
- Pretrained DINOv2 backbone
- Custom CNN Decoder
- Standard inference pipeline

---

## 🏗️ Model Architecture
- **Backbone:** DINOv2 (dinov2_vits14)
- **Decoder:** 3-layer CNN segmentation head
- **Input Size:** 252 x 448
- **Loss Function:** (Mention from training script)
- **Classes:** 10

---

## 📊 Validation Results

| Variant | IoU |
|----------|--------|
| Baseline | **0.4559** |

---

## 🖼️ Sample Predictions

Screenshots of predictions are available in:

`visual_results/`

<img width="1000" height="500" alt="cc0000745" src="https://github.com/user-attachments/assets/38cae410-a37f-4224-bdbd-1033e586085b" />



---

## ▶️ How to Run

```bash
python train_segmentation.py
python generate_submissions.py
python evaluate_variants.py

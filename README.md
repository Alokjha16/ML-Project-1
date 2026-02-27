# 🚀 Ignitia Hackathon – Baseline Variant

## 📌 Project Overview
This repository contains the **Baseline implementation** of a DINOv2-based semantic segmentation model.

The model performs pixel-level classification into 10 classes using a pretrained Vision Transformer backbone and a custom CNN decoder.

The model uses:
- Pretrained DINOv2 backbone
- Custom CNN Decoder
- Standard inference pipeline

---

## 🏗️ Model Architecture
- Backbone: DINOv2 (dinov2_vits14)
- Decoder: Custom CNN Head
- Input Size: 252 × 448
- Output Classes: 10
- Framework: PyTorch

---

## 📊 Validation Results

| Variant | IoU |
|----------|--------|
| Baseline | **0.4559** |

---

## 🖼️ Sample Prediction


<img width="1000" height="500" alt="cc0000745" src="https://github.com/user-attachments/assets/38cae410-a37f-4224-bdbd-1033e586085b" />



---

## ▶️ How to Run

```bash
python train_segmentation.py
python generate_submissions.py
python evaluate_variants.py

# DeepLearningAndRobotPereception_Project
Progetto d' Esame di DEEP LEARNING AND ROBOT PERCEPTION- Università degli Studi di Perugia

# 🧠 Monocular Depth Estimation

Deep Learning project for **monocular depth estimation**, developed using **PyTorch**.  
The goal is to predict a **depth map** (distance of each pixel from the camera) from a **single RGB image**.

👨‍🎓 **Author:** Daniele Angeloni  
🏫 **University:** Università degli Studi di Perugia  
📚 **Course:** Deep Learning and Robot Perception - CHALLENGE

---

## 🚀 Project Overview

This project implements a **supervised deep learning pipeline** for depth estimation using an **encoder–decoder neural network**.

Main features:

- 🧩 **Encoder:** ResNet-50 pretrained on ImageNet  
- 🔄 **Decoder:** custom upsampling network with **bilinear upsampling + skip connections**  
- 🖼 **Output:** dense **depth map prediction**

The architecture is inspired by **U-Net** and is designed to preserve spatial details during reconstruction.

---

## 📂 Dataset

Dataset used: **DepthEstimationUnreal**

Characteristics:

- RGB images → **JPEG**
- Depth maps → **NumPy arrays**
- Resolution → **144 × 256**
- Depth range → **0 – 20 meters**

Dataset split:

```
dataset/
├── rgb/
│ ├── train
│ ├── val
│ └── test
└── depth/
├── train
├── val
└── test
```

---

## 🏗 Model Architecture

Pipeline:


RGB Image
↓
ResNet50 Encoder
↓
Multi-scale Features
↓
Decoder with Upsampling
↓
Predicted Depth Map


Key techniques:

- transfer learning
- skip connections
- bilinear upsampling

---

## 📉 Loss Function

Training optimizes a combination of:

- 📏 **MSE (Mean Squared Error)**
- 🧠 **SSIM (Structural Similarity Index)**


Loss = MSE + SSIM


This improves both **numerical accuracy** and **perceptual quality** of the predicted depth maps.

---

## 📊 Results

| Metric | Value |
|------|------|
| RMSE | **2.05** |
| SSIM | **0.68** |

The model produces **consistent and structurally accurate depth predictions**.

---

## 📁 Repository Structure

project/
├── model.py # neural network architecture
├── solver.py # training, validation and testing
├── dataset.py # dataset loader
├── utils.py # metrics and visualization
├── main.py # experiment execution
└── README.md


---

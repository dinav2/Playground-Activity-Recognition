# MP-GCN Playground Group Activity Recognition  
Multi-Person Graph Convolutional Network for Safety-Oriented Activity Classification

This repository contains a full pipeline for **Group Activity Recognition (GAR)** in playground environments using **2D skeletons**, **object context**, and a **Multi-Person Graph Convolutional Network (MP-GCN)**.  
The system takes raw videos, extracts poses, annotates scenes in CVAT, builds panoramic human–object graphs, and trains an MP-GCN model to classify scenes into:

- **Transit**
- **Play_Object_Normal**
- **Play_Object_Risk**

---

## 🚀 Project Overview

Playground environments are complex: multiple people interact simultaneously with static or moving structures (slides, swings, ramps), often with occlusions or irregular motion.  
This project builds an **end-to-end automated pipeline** that transforms raw videos into graph-structured tensors suitable for deep learning models.

The model is based on the MP-GCN architecture proposed by *Li et al., 2024*, adapted to safety-focused playground scenes.

---

## 📦 Repository Structure

├── data/
│ ├── npz/ # Final graph tensors for training
│ ├── intermediate/ # JSON structured pose/object annotations
│ ├── cvat_exports/ # Raw CVAT annotation files
│ └── stats/ # Dataset statistics & plots
│
├── scripts/
│ ├── script.py # Single-video full pipeline: frames → CVAT → poses → JSON
│ ├── run_batch_pipeline.py# Batch processing for multiple videos
│ ├── extract_job_annotations.py
│ ├── cvat_to_intermediate.py
│ ├── intermediate_to_npz.py
│ └── dataset_stats.py
│
├── MPGCN/
│ ├── nets.py # MP-GCN model implementation
│ └── graphs.py # Graph adjacency definitions
│
├── train_mpgcn.py # Full training script
├── utils/ # Helper functions
└── README.md


---

## 🧩 Pipeline Summary

The system consists of **four major stages**:

### **1. Pose Extraction and Tracking**
- Frames extracted at **15 FPS**
- Human poses via **YOLO-Pose**
- Temporal identity via **DeepSort**
- Outputs:
  - 17-joint skeletons (COCO)
  - Consistent person IDs
  - Detected object centroids

### **2. Annotation in CVAT**
- Automatic upload of frames and detections
- Manual scene-level labeling:
  - Transit  
  - Play\_Object\_Normal  
  - Play\_Object\_Risk  
- Optional:
  - roles, safety flags, actions

### **3. Graph Tensor Construction**
Each clip becomes:
X ∈ R[ C=2 , T=30 , V'=21 , M=6 ]


Where:
- **C**: coordinates (x,y)
- **T**: frames per clip
- **V'**: 17 human joints + 4 object nodes
- **M**: max persons per clip

### **4. MP-GCN Model**
- Spatial graph convolutions  
- Temporal convolutions  
- Learnable adjacency refinement  
- Person-level attention pooling  
- 3-way softmax classification

---

## 🧠 Training the Model

Run:

```bash
python3 train_mpgcn.py \
    --data-dir data/npz \
    --epochs 21 \
    --batch-size 8 \
    --use-augmentation
```

### 📝 Training Logs Include
- Accuracy curves  
- Class-specific behavior  
- Confusion matrix  

---

### 🧪 Dataset Summary

**Final dataset class distribution:**

| Class               | Samples |
|--------------------|---------|
| Transit            | 74      |
| Play_Object_Normal | 25      |
| Play_Object_Risk   | 21      |

Dataset imbalance strongly influences validation metrics, especially between normal vs. risky behavior.

---

### 📊 Results

- **Training accuracy:** ~0.62  
- **Validation accuracy:** fluctuating due to:  
  - small dataset  
  - class imbalance  
  - subtle pose differences  

**MP-GCN successfully captures:**
- human–human interactions  
- human–object interactions  
- group-level motion patterns  

**Key plot examples:**
- `resultsTrainingVal.png`
- `confusionMatrix.png`

---

### 🔍 Limitations
- Dataset size and imbalance  
- Only 2D pose — no depth cues  
- Static object treatment (objects may move)  
- Fine-grained risk labeling is challenging  

---

### 🛠 Future Work
- Multi-view or 3D pose recovery  
- Dynamic object nodes  
- Self-supervised pretraining  
- Larger-scale annotated dataset  
- Temporal attention for micro-actions  

---

### 📚 References

**Li, Z., Chang, X., Li, Y., & Su, J. (2024).**  
*Skeleton-Based Group Activity Recognition via Spatial-Temporal Panoramic Graph.*

**Choi, W., Shahid, K., & Savarese, S. (2009).**  
*What are they doing? Collective activity classification using spatio-temporal relationships among people.*

---

### 👨‍💻 Authors
- **David Gómez** – Tecnológico de Monterrey  
- **Angela Aguilar** – Tecnológico de Monterrey  
- **Jorge Reyes** – Tecnológico de Monterrey  

---

### ⭐ Acknowledgements
This project was developed as part of a research-oriented course on computational vision and machine learning, applying MP-GCN methods to real-world safety monitoring scenarios.






# AFCG-Net
AFCG-Net: Frequency-aware cross-scale graph attention for robust breast ultrasound lesion detection using adaptive sparsity and deep learning.

# AFCG-Net: Adaptive Frequency-Aware Cross-Scale Graph Network

Official implementation of **AFCG-Net**, a deep learning framework for breast ultrasound lesion detection using frequency-aware graph attention, cross-scale feature reasoning, and adaptive sparsity.

## Framework Overview

AFCG-Net integrates a YOLOv3 detection framework with a ResNet-101 backbone and an Adaptive Frequency-aware Cross-scale Graph Attention Module (AFCG-AM).

The architecture includes three principal components:

- **Frequency-Aware Affinity (FAA):** Integrates spatial, semantic, and frequency-domain information using local 2D Discrete Cosine Transform (DCT) descriptors.
- **Cross-Scale Graph Edges (CSGE):** Establishes graph connections across multiple feature resolutions to improve lesion localization and preserve small-lesion information.
- **Adaptive Sparsity (AS):** Uses differentiable neighborhood selection to dynamically regulate graph connectivity and reduce unnecessary computation.

The framework also incorporates multi-head Graph Attention Networks (GATs) and a Global Channel Attention Module (GCAM).

## Datasets

The framework supports three publicly available breast ultrasound datasets:

| Dataset | Images | Categories |
|---|---:|---|
| BUSI | 647 | Benign, Malignant |
| UDIAT | 418 | Benign, Malignant |
| BrEaST | 252 | Benign, Malignant |

Images are resized to 416 × 416 pixels, with a 60:20:20 training, validation, and testing split.

## Training Configuration

- **Backbone:** ResNet-101
- **Detection framework:** YOLOv3
- **Input resolution:** 416 × 416
- **Optimizer:** SGD
- **Training epochs:** 500
- **Batch size:** 8
- **Initial learning rate:** 0.002
- **Momentum:** 0.9
- **Weight decay:** 0.0005
- **Graph attention layers:** 2
- **Attention heads:** 4
- **Random seed:** 0

## Evaluation

The evaluation pipeline supports:

- Precision, Recall, and F1-score
- mAP@0.5 and mAP@0.5:0.95
- Benign and malignant class-wise Average Precision
- Cross-dataset evaluation
- Lesion-size-based performance analysis
- Computational efficiency analysis
- Detection visualization and training curves

All evaluation metrics are calculated from model predictions and ground-truth annotations.

## Implementation

**Framework:** PyTorch

**Programming Language:** Python

**Application:** Medical Image Analysis, Breast Ultrasound Lesion Detection

**Core Techniques:** Graph Attention Networks, Frequency-Domain Feature Learning, Cross-Scale Feature Fusion, Adaptive Graph Sparsity

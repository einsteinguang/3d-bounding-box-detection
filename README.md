# 3d Bounding Box Detection from RGBD (image + pointcloud) and objects' 2d binary masks  

This repository contains code and documentation for a 3D object detection project using both a baseline model and deep learning–based models (early fusion and late fusion). Below are the key steps, important details, and instructions for setting up, training, and running the models.

---

## Table of Contents

1. [Overview](#overview)  
2. [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)  
3. [Baseline Model](#baseline-model)  
4. [Learning-Based Models](#learning-based-models)  
   - [Early Fusion](#early-fusion)  
   - [Late Fusion](#late-fusion)  
5. [Loss Function](#loss-function)  
6. [Evaluation](#evaluation)  
7. [How to Execute](#how-to-execute)  
   - [Environment Setup](#environment-setup)  
   - [Training](#training)  
   - [Inference](#inference)  
   - [Convert to ONNX](#convert-to-onnx)  
   - [Inference with ONNX Models](#inference-with-onnx-models)  

---

## Overview

This project estimates 3D bounding boxes (bbox) for objects based on:
- Point cloud data (e.g., from sensors like LiDAR).
- Corresponding RGB images and object masks.

Three main approaches are implemented:
1. **Baseline**: Estimates 3D bboxes by calculating geometric properties (center, dimensions, and orientation) directly from the point cloud’s principal components.
2. **Early Fusion**: Uses a cross-attention–based deep learning model that combines point cloud embeddings with image features before producing bbox predictions.
3. **Late Fusion**: Similar to Early Fusion but merges the point cloud and image features at a later stage.

---

## Data Preprocessing and Augmentation

1. **Shifting and Normalization**  
   - Shift all point clouds and ground-truth bbox coordinates by (0., 0., –1.).
   - Randomly scale and shift the coordinates and point clouds.

2. **Image Augmentations**  
   - Apply color jitter to the RGB image.
   - Add Gaussian noise to the image.

3. **Point Cloud Sampling**  
   - Sample 10,000 points for each object from the combined point cloud and mask.

4. **Dimension Adjustments**  
   - Pad the point cloud, RGB image, and mask to a fixed size (715 × 1003).
   - Downscale RGB images and masks by a fixed ratio.
   - Add Gaussian noise to the point cloud.

> Note: Some of these augmentations (especially color jitter and added noise) apply only during training.

---

## Baseline Model

1. **Computation**  
   - Calculates the center of the object.
   - Determines the 3 principal axes (via eigenvectors).
   - Computes the dimensions in each principal axis direction.

2. **Performance**  
   - Achieves around **0.36 average IoU** (intersection over union) on both training and test datasets.

---

## Learning-Based Models

### Early Fusion

1. **Input**  
   - Point cloud (shape: `[Batch, Nobj, Nsample, 3]`).
   - RGB image features from a ResNet-18 backbone.
   - Object masks.

2. **Architecture**  
   - A PointNet-like module generates point cloud embeddings.
   - ResNet-18 extracts image features; object masks are optionally used via pooling.
   - Cross-attention layers fuse the point cloud embeddings with the image features.
   - A final MLP (bbox head) outputs the 3D bounding box parameters.

### Late Fusion

1. **Input**  
   - Point cloud embeddings (via PointNet).
   - Mask embeddings (via a CNN).
   - RGB image features (via ResNet-18).

2. **Architecture**  
   - Embeddings are merged (concatenated + MLP) to form a query.
   - Cross-attention layers process these queries with the image feature maps.
   - Outputs the 3D bounding box parameters.

---

## Loss Function

- Each predicted bbox is compared to the ground truth with an L1-type loss, weighted by coefficients for:
  - Center coordinates
  - Dimensions
  - Orientation (using a continuous rotation representation, per [Yi Zhou et al., 2020])
  
Formally:

```
loss = mean(
    w1 * L1(diff_center - center_pred) +
    w2 * L1(diff_dims - dims_pred) +
    w3 * L1(diff_orient - orient_pred)
)
```

Where:
- `diff_center`, `diff_dims`, and `diff_orient` are the differences between the **baseline model estimate** and the ground truth.
- The model predicts corrections on top of the baseline’s estimates.

---

## Evaluation
 The following metrics are only trained on 180 samples and 20 evaluation samples. Of course each sample might contain multiple objects in one image.

| Method        | Avg. IoU (val) | Model Size     | Inference Time      |
|---------------|----------------|----------------|---------------------|
| **Baseline**  | 0.36           | 0 parameters   | ~17 ms*             |
| **Early Fusion** | 0.455      | ~20.6M         | ~37 ms (+17 ms)*    |
| **Late Fusion**  | 0.435      | ~20.7M         | ~34 ms (+17 ms)*    |

\* Inference time measured on an NVIDIA GeForce RTX 2080 Ti. The ONNX inference time was not fully tested due to cuDNN version issues.

---

## How to Execute

### 1. Environment Setup
- Use **Python 3.9** (recommended).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Training

1. **Rename Data Samples**  
   - Inside the `dl_challenge` folder, rename all data folders to `data_{i}`.

2. **Configure Parameters**  
   - Adjust `config/params_late_fusion.yaml` or `config/params_early_fusion.yaml` according to your requirements.

3. **Run Training**  
   - Example (using multiple GPUs):
     ```bash
     bash train.sh 0,1,2,3
     ```
   - Gathers data from `dl_challenge` folder and trains the model.

### 3. Inference

- Run:
  ```bash
  python inference.py
  ```
- This will:
  1. Perform inference with the **baseline**, **early_fusion**, and **late_fusion** models.
  2. Generate 3D bboxes, save them into each `data_{i}` folder (visualized on the images).
  3. Print IoU scores and inference times.

### 4. Convert to ONNX

- Example command:
  ```bash
  python convert_to_onnx.py \
    --config config/params_late_fusion.yaml \
    --checkpoint checkpoints/model_late_fusion.pth \
    --output checkpoints/model_late_fusion.onnx
  ```

### 5. Inference with ONNX Models

- Example command:
  ```bash
  python inference_onnx.py \
    --config config/params_early_fusion.yaml \
    --onnx checkpoints/model_early_fusion.onnx
  ```

---

## Additional Resources

- **Checkpoints** for all models are available for download at:
  ```
  https://drive.google.com/drive/folders/1VkFcz5G7i0p6Jw8tjyYIsJzOGzs3eWOf?usp=sharing
  ```

---

*Thank you for using this project! We hope this documentation helps you set up, train, and evaluate the models effectively.*
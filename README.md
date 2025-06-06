# Object Detection on BDD100K Dataset

[![Dataset Download](https://img.shields.io/badge/Download-BDD100K_Dataset-blue)](http://bdd-data.berkeley.edu/download.html)

This project focuses on object detection using the Berkeley DeepDrive (BDD100K) dataset, featuring analysis of class distributions, occlusion patterns, and model development.

## 🛠️ Setup
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q0-3MbEBTthbimI9UnCrwq5dI0m17Otf?usp=sharing)

```bash
# clone 
git clone https://github.com/basaanithanaveenkumar/object-detection-BBD.git
cd object-detection-BBD
```


### Docker Build and run
```bash
sudo docker build -t object-detection .
#run the docker
sudo docker run -it --rm --gpus all object-detection /bin/bash

```



### Dataset Preparation
```bash
# Download and extract dataset
mkdir -p data
python scripts/download_dataset.py
```

# Organize directory structure
```bash
mv data/100k/val data/100k/valid
```
# Project Overview

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```


## Class-wise Statistics

To get class-wise statistics, run the following script:

```bash
python scripts/get_stats_classwise.py
```

## scene-wise Statistics

To get scene-wise statistics, run the following script:

```bash
python scripts/get_stats_scenewise.py


```
## Class distribution Statistics

To get Class distribution, run the following script:

```bash
python scripts/get_stats_dist.py
```

# training distribution

![Train dataset distribution](images/train-distribution.png)

# validation distribution

![validation dataset distribution](images/validation-distribution.png)





## Training dataset  Analysis
![Train dataset Analusis](images/Screenshot%202025-04-12%20at%208.04.38%20PM.png)


### Use Case Specific Classes (10 Classes)

| Class         | Count    |
|---------------|----------|
| Car           | 714,121  |
| Traffic Sign  | 239,961  |
| Traffic Light | 186,301  |
| Person        | 91,435   |
| Truck         | 30,012   |
| Bus           | 11,688   |
| Bike          | 7,227    |
| Rider         | 4,522    |
| Motor         | 3,002    |
| Train         | 136      |


## All Classes Distribution

| Class                   | Count    |
|-------------------------|----------|
| Car                     | 714,121  |
| Lane/Single White       | 247,108  |
| Traffic Sign            | 239,961  |
| Traffic Light           | 186,301  |
| Lane/Road Curb          | 109,868  |
| Lane/Crosswalk          | 108,284  |
| Person                  | 91,435   |
| Area/Drivable           | 64,050   |
| Area/Alternative        | 61,799   |
| Lane/Double Yellow      | 37,519   |
| Truck                   | 30,012   |
| Lane/Single Yellow      | 20,220   |
| Bus                     | 11,688   |
| Bike                    | 7,227    |
| Lane/Double White       | 5,674    |
| Rider                   | 4,522    |
| Motor                   | 3,002    |
| Lane/Single Other       | 249      |
| Train                   | 136

## Occlusion Analysis

An analysis was performed on all classes with respect to the **`occluded`** attribute. Results show that nearly half of the objects are occluded.

### Occlusion Statistics

| Occlusion Status | Count    |
|------------------|----------|
| False            | 678,981  |
| True             | 609,424  |

⚠️ **Observation:** A significant portion of the dataset contains occluded objects, which may impact detection accuracy and should be considered during model training and evaluation.



# Additional training Dataset Analysis

## Statistics on BBOX size and aspect ratio

To get BBOX distribution statstics, run the following script:

```bash
python scripts/get_additional_analysis.py
```


## Bounding Box Size Distribution

| Size Category       | Percentage | Interpretation                                                                 |
|---------------------|------------|-------------------------------------------------------------------------------|
| **Small** (<2% area) | 92.2%      | Dominates dataset - consider higher resolution or small object detection techniques |
| **Medium** (2-10%)  | 6.3%       | Underrepresented - may need augmentation                                      |
| **Large** (>10%)    | 1.5%       | Very rare - ensure model can handle context                                   |

**Recommendations:**
- Implement multi-scale training
- Use Feature Pyramid Networks (FPN)

## Bounding Box Aspect Ratios

| Aspect Ratio        | Percentage | Interpretation                                                                 |
|---------------------|------------|-------------------------------------------------------------------------------|
| **Tall** (<0.5)     | 12.3%      | Vertical objects present - adjust anchor boxes                                |
| **Square & rectangle ** (0.5-2)  | 76.9%      | Majority class - standard anchors should work well                            |
| **Wide** (≥2)       | 10.8%      | Horizontal objects - may need custom anchors                                 |




## Scene-wise Statistics

To generate scene-wise statistics, run the following script:

```bash
python scripts/get_stats_sceneiwise.py
```


## convert the dataset into microsoft coco

To train the model using RF-DETR, it need to be converted into coco:

```bash
python scripts/convert_to_coco.py
```

## script to visualize the GT  based on ID (COCO based)

To visualize the bboxes, run the following script:

```bash
# please modify your script according to your needs
python scripts/visualize_image.py 
```

![sample](<images/Screenshot 2025-04-18 at 8.46.28 PM.png>)


## Training using RF-DETR

To train the model using RF-DETR, run the following script:

```bash
python scripts/train_rfdetr.py
```
## Visualizations & Key Concepts


### 1. Vision Transformer (ViT) Adaptations
#### Modified Architecture
![ViT](images/vit.png)  


![Modified ViT](images/modified_vit.png)  
**Description**: Custom ViT architecture with hierarchical features through global and windowed attention

#### Patch Projection
![ViT Projection](images/vit_projection.png)  
---


### 2. DINO (Self-Supervised Learning Framework)
![DINO](images/DINO.png)  
**Description**: Architecture of the DINO self-supervised learning framework, which uses knowledge distillation with a teacher-student network to learn robust visual representations without labeled data.

---

### 3. Deformable Convolution
#### Standard vs. Deformable Comparison
![Deformable Convolution](images/deformable_conv.png)  
**Description**: Comparison between standard convolution (fixed grid) and deformable convolution (adaptive sampling locations). Enhances CNNs for irregular object shapes.

#### Deformable Convolution in Action
![Deformable Conv Visualization](images/Deformable_conv2.png)  
**Description**: Visualization of deformable convolution offsets dynamically adjusting to object geometry.

---

### 5. Deformable DETR (Object Detection)
![Deformable-DETR](images/Defromable-DETE.png)  
**Description**: Combines deformable convolutions with Transformer attention for efficient object detection, reducing training complexity of vanilla DETR.

---

### 5. GIoU (Generalized Intersection over Union)
![GIoU](images/GIOU.png)  
**Description**: Improves bounding box regression by accounting for both overlap and enclosure, addressing limitations of standard IoU in non-overlapping cases.

---


# Model Evaluation Results

Below are the key evaluation metrics and visualizations from from RF-DETR model:

## 1. Confusion Matrix
![Confusion Matrix](images/confession_matirx_eval.png)  
**Description**: The confusion matrix shows the model's classification performance across different classes. High diagonal values indicate correct predictions, while off-diagonal values represent misclassifications.

## 2. Detection Grid
![Detection Grid](images/detections_grid.png)  

## 3. Predictions vs. Annotations
![Detections vs. Annotations](images/detections_vs_annotation.png)  
**Description**: Side-by-side overlay of predicted bounding boxes (red) and ground truth (green). 
## 4. mAP Scores
![mAP Scores](images/mAP_scores_Eval.png)  
**Description**: Mean Average Precision (mAP) across IoU thresholds. Higher mAP (close to 1.0) indicates better detection accuracy.



## Notes
    Model training
    sampling techniques to balance dataset
       --- as the dataset is imbalanced
    tweak the focal loss
    data aggumentation
       -- 
    Hyper parameter tunning
    change the backbone
    bapatite matching based architecture


#    Key metrics

    Mean Average Precision (mAP) - Standard for COCO/PASCAL VOC

    Precision-Recall Curves - Trade-off analysis

    F1 Score - Balanced precision/recall

    Inference Speed (FPS) - Critical for real-time applications (self-driving)
    

    

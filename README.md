# Object Detection on BDD100K Dataset

[![Dataset Download](https://img.shields.io/badge/Download-BDD100K_Dataset-blue)](http://bdd-data.berkeley.edu/download.html)

This project focuses on object detection using the Berkeley DeepDrive (BDD100K) dataset, featuring analysis of class distributions, occlusion patterns, and model development.

## 🛠️ Setup


```bash
# clone 
git cloen https://github.com/basaanithanaveenkumar/object-detection-BBD.git
cd object-detection-BBD
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
| **Square** (0.5-2)  | 76.9%      | Majority class - standard anchors should work well                            |
| **Wide** (≥2)       | 10.8%      | Horizontal objects - may need custom anchors                                 |




## Scene-wise Statistics

To generate scene-wise statistics, run the following script:

```bash
python scripts/get_stats_sceneiwise.py
```


## Training using RF-DETR

To train the model using RF-DETR, run the following script:

```bash
python scripts/train_rfdetr.py
```

## script to visualize the GT & predictions based on ID (COCO based)

To visualize the bboxes, run the following script:

```bash
# please modify your script according to your needs
python scripts/visualize_image.py 
```

![sample](<images/Screenshot 2025-04-18 at 8.46.28 PM.png>)
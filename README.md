# Object Detection on BDD100K Dataset

[![Dataset Download](https://img.shields.io/badge/Download-BDD100K_Dataset-blue)](http://bdd-data.berkeley.edu/download.html)

This project focuses on object detection using the Berkeley DeepDrive (BDD100K) dataset, featuring analysis of class distributions, occlusion patterns, and model development.

## 🛠️ Setup

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




## Scene-wise Statistics

To generate scene-wise statistics, run the following script:

```bash
python scripts/get_stats_sceneiwise.py
```


# need to understand the stats for this 
    Class imbalance

    Attribute distribution (occluded, truncated etc.)

    Bounding box characteristics

    Percentage distribution across classes

# TODO
1. find the stats for 
    different weather conditions 
    different types of scenes.
    stats of count of object for each class

2. find a way how to solve the imbalances in the dataset


Model

1. create a modular design in jupyter notebook
     -- Dataloader
     -- preprocessing
     -- backbone
     -- neck
     -- head
     -- Nms
     -- bapatite matching
     -- evalution
     --visulaization 
     -- add tensorboard to track the experiment logs







#TODO for analysis

analyzing the train and val split. (test set is not required for analysis)
 Based on the analysis, see if you can identify any anomalies or patterns in
each of the object detection classes.
 visualizing the stats of the dataset in a dashboard.
 Identifying and visualizing interesting/unique samples in different classes.
# object-detection-BBD

Link to Download the datset http://bdd-data.berkeley.edu/download.html (GUI)


run  the below command to download  and extract the dataet 

python scripts/download_dataset.py 


mkir data
mv data/100k/val data/100k/valid

Installation
what needs to be installed 

run 

pip install -r requirements.txt


# get the stats class wise

python scripts/get_stats_classwise.py 

# image resolution size in dataset
1280x720

# observation
  --- imballance in the classes distribution

# training analysis
# usecase specific calsses (10 classes)
car: 714121
traffic sign: 239961
traffic light: 186301
person: 91435
truck: 30012
bus: 11688
bike: 7227
rider: 4522
motor: 3002
train: 136



# all classes 

car: 714121
lane/single white: 247108
traffic sign: 239961
traffic light: 186301
lane/road curb: 109868
lane/crosswalk: 108284
person: 91435
area/drivable: 64050
area/alternative: 61799
lane/double yellow: 37519
truck: 30012
lane/single yellow: 20220
bus: 11688
bike: 7227
lane/double white: 5674
rider: 4522
motor: 3002
lane/single other: 249
train: 136
lane/double other: 26
area/unknown: 2

# occulusion analysis on all classes ( seems half of them are occluded) 
Attribute: occluded
  false: 678981
  true: 609424




# get the stats scene  wise

python scripts/get_stats_sceneiwise.py 


#TODO Need to save the plots instead of displaying it



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
import json
import os
from collections import defaultdict

# List of specific classes we're interested in
TARGET_CLASSES = [
    'car', 'traffic sign', 'traffic light', 'person', 
    'truck', 'bus', 'bike', 'rider', 'motor', 'train'
]

def analyze_json_files(folder_path):
    # Initialize counters
    class_distribution = defaultdict(int)
    class_attribute_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    global_attribute_counts = defaultdict(lambda: defaultdict(int))
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Process each frame in the JSON
                if 'frames' in data:
                    for frame in data['frames']:
                        if 'objects' in frame:
                            for obj in frame['objects']:
                                category = obj.get('category', 'unknown').lower()
                                
                                # Only count if it's one of our target classes
                                if category in TARGET_CLASSES:
                                    class_distribution[category] += 1
                                    
                                    # Count attributes for this specific class
                                    if 'attributes' in obj:
                                        for attr, value in obj['attributes'].items():
                                            # Convert boolean to string for consistent counting
                                            if isinstance(value, bool):
                                                value = str(value).lower()
                                            class_attribute_counts[category][attr][value] += 1
                                            global_attribute_counts[attr][value] += 1
                                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return class_distribution, class_attribute_counts, global_attribute_counts

def print_stats(class_dist, class_attr_counts, global_attr_counts):
    print("Class Distribution:")
    print("------------------")
    # Print in the order of TARGET_CLASSES
    for cls in TARGET_CLASSES:
        count = class_dist.get(cls, 0)
        print(f"{cls}: {count}")
    
    print("\nGlobal Attribute Counts:")
    print("----------------------")
    for attr, value_counts in global_attr_counts.items():
        print(f"\nAttribute: {attr}")
        for value, count in sorted(value_counts.items(), key=lambda x: -x[1]):
            print(f"  {value}: {count}")
    
    print("\nAttribute Counts by Class:")
    print("------------------------")
    for cls in TARGET_CLASSES:
        if cls in class_attr_counts:
            print(f"\nClass: {cls}")
            for attr, value_counts in class_attr_counts[cls].items():
                print(f"  Attribute: {attr}")
                for value, count in sorted(value_counts.items(), key=lambda x: -x[1]):
                    print(f"    {value}: {count}")

if __name__ == "__main__":
    print (" class wise stats for training ")
    folder_path = "data/100k/train"
    
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, class_attr_counts, global_attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, class_attr_counts, global_attr_counts)
    print (" class wise stats for testing ")
    folder_path = "data/100k/test"
    
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, class_attr_counts, global_attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, class_attr_counts, global_attr_counts)
    print (" class wise stats for validation ")
    folder_path = "data/100k/val"
    
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, class_attr_counts, global_attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, class_attr_counts, global_attr_counts)
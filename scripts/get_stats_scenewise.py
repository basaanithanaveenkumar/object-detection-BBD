import os
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def get_attribute_stats(folder_path):
    """
    Analyze attribute statistics from JSON files in a folder.
    Each JSON file should contain an "attributes" field with weather, scene, and timeofday.
    
    Args:
        folder_path (str): Path to folder containing JSON files
    
    Returns:
        dict: Nested dictionary containing counts for each attribute category
    """
    stats = {
        'weather': defaultdict(int),
        'scene': defaultdict(int),
        'timeofday': defaultdict(int)
    }
    
    total_files = 0
    
    # Iterate through all JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                # Extract attributes (with error handling)
                attributes = data.get('attributes', {})
                
                # Count each attribute type
                for attr_type in stats.keys():
                    attr_value = attributes.get(attr_type, 'undefined')
                    stats[attr_type][attr_value] += 1
                
                total_files += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Convert defaultdicts to regular dicts
    return {k: dict(v) for k, v in stats.items()}, total_files

def print_statistics(stats, total_files):
    """Print formatted statistics"""
    print(f"\nAnalyzed {total_files} files\n")
    
    for attr_type, counts in stats.items():
        print(f"=== {attr_type.upper()} ===")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for value, count in sorted_counts:
            percentage = (count / total_files) * 100
            print(f"{value.ljust(20)}: {count:6d} ({percentage:.1f}%)")
        print()

def plot_statistics(stats):
    """Create visualization plots"""
    plt.figure(figsize=(15, 5))
    
    for i, (attr_type, counts) in enumerate(stats.items(), 1):
        plt.subplot(1, 3, i)
        df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
        df.sort_values('count', ascending=False).plot(kind='bar', ax=plt.gca())
        plt.title(attr_type.capitalize() + ' Distribution')
        plt.xlabel(attr_type)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    print("printing the stats for trainset")
    # Path to your train folder containing JSON files
    train_folder = 'data/100k/train'
    
    # Get statistics
    attribute_stats, total_files = get_attribute_stats(train_folder)
    
    # Print results
    print_statistics(attribute_stats, total_files)
    
    # Visualize results
    plot_statistics(attribute_stats)

    print("printing the stats for test set")
    # Path to your train folder containing JSON files
    train_folder = 'data/100k/test'
    
    # Get statistics
    attribute_stats, total_files = get_attribute_stats(train_folder)
    
    # Print results
    print_statistics(attribute_stats, total_files)
    
    # Visualize results
    plot_statistics(attribute_stats)

    print("printing the stats for val set")
    # Path to your train folder containing JSON files
    train_folder = 'data/100k/val'
    
    # Get statistics
    attribute_stats, total_files = get_attribute_stats(train_folder)
    
    # Print results
    print_statistics(attribute_stats, total_files)
    
    # Visualize results
    plot_statistics(attribute_stats)
import json
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Target classes as a list
TARGET_CLASSES = [
    'car',
    'traffic sign', 
    'traffic light',
    'person',
    'truck',
    'bus',
    'bike',
    'rider',
    'motor',
    'train'
]

def analyze_dataset(folder_path):
    # Initialize statistics trackers
    stats = {
        'class_counts': defaultdict(int),
        'class_attributes': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        'global_attributes': defaultdict(lambda: defaultdict(int)),
        'height_width': defaultdict(list)
    }
    
    # Process each JSON file
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(folder_path, filename)) as f:
                    data = json.load(f)
                
                # Process each frame
                for frame in data.get('frames', []):
                    for obj in frame.get('objects', []):
                        category = obj.get('category', '').lower().strip()
                        
                        # Only process target classes
                        if category in TARGET_CLASSES:
                            stats['class_counts'][category] += 1
                            
                            # Process attributes
                            attributes = obj.get('attributes', {})
                            for attr, value in attributes.items():
                                # Standardize attribute values
                                val = str(value).lower() if isinstance(value, bool) else str(value)
                                stats['class_attributes'][category][attr][val] += 1
                                stats['global_attributes'][attr][val] += 1
                            
                            # Process bounding box dimensions
                            if 'box2d' in obj:
                                box = obj['box2d']
                                width = box['x2'] - box['x1']
                                height = box['y2'] - box['y1']
                                stats['height_width'][category].append((height, width))
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    return stats

def visualize_results(stats):
    # Create DataFrame for class distribution
    df_classes = pd.DataFrame.from_dict(
        stats['class_counts'], 
        orient='index',
        columns=['count']
    ).sort_values('count', ascending=False)
    
    # Plot settings
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = [15, 10]
    
    # 1. Class Distribution Plot
    plt.subplot(2, 2, 1)
    ax = sns.barplot(
        x=df_classes.index,
        y='count',
        data=df_classes,
        palette='viridis'
    )
    plt.title('Class Distribution', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Instances')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height()):,}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center', 
            xytext=(0, 10), 
            textcoords='offset points'
        )
    
    # 2. Percentage Distribution
    plt.subplot(2, 2, 2)
    df_classes['percentage'] = (df_classes['count'] / df_classes['count'].sum()) * 100
    df_classes['percentage'].plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        labels=df_classes.index,
        textprops={'fontsize': 8}
    )
    plt.title('Class Percentage Distribution', pad=20)
    plt.ylabel('')
    
    # 3. Common Attributes
    plt.subplot(2, 2, 3)
    if stats['global_attributes']:
        attr_name = next(iter(stats['global_attributes']))
        attr_data = pd.DataFrame.from_dict(
            stats['global_attributes'][attr_name],
            orient='index',
            columns=['count']
        ).sort_values('count', ascending=False)
        
        attr_data.plot.pie(
            y='count',
            autopct='%1.1f%%',
            legend=False,
            title=f'Global "{attr_name}" Distribution'
        )
    
    # 4. Bounding Box Analysis
    plt.subplot(2, 2, 4)
    if stats['height_width']:
        sample_class = df_classes.index[0]
        hw_data = pd.DataFrame(
            stats['height_width'][sample_class],
            columns=['height', 'width']
        )
        sns.scatterplot(
            data=hw_data,
            x='width',
            y='height',
            alpha=0.5
        )
        plt.title(f'BBox Dimensions for {sample_class}')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n=== CLASS DISTRIBUTION ===")
    print(df_classes)
    
    print("\n=== GLOBAL ATTRIBUTE COUNTS ===")
    for attr, values in stats['global_attributes'].items():
        print(f"\nAttribute: {attr}")
        for val, count in sorted(values.items(), key=lambda x: -x[1]):
            print(f"  {val}: {count:,}")
    
    print("\n=== CLASS-SPECIFIC ATTRIBUTES ===")
    for cls in df_classes.index:
        if cls in stats['class_attributes']:
            print(f"\nClass: {cls}")
            for attr, values in stats['class_attributes'][cls].items():
                print(f"  {attr}:")
                for val, count in sorted(values.items(), key=lambda x: -x[1]):
                    print(f"    {val}: {count:,}")

if __name__ == "__main__":
    print (" class wise stats for training ")
    folder_path = "data/100k/train"
    
    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path")
    else:
        print(f"Analyzing dataset for classes: {', '.join(TARGET_CLASSES)}")
        stats = analyze_dataset(folder_path)
        visualize_results(stats)
    print (" class wise stats for validation ")
    folder_path = "data/100k/val"
    
    if not os.path.isdir(folder_path):
        print("Error: Invalid folder path")
    else:
        print(f"Analyzing dataset for classes: {', '.join(TARGET_CLASSES)}")
        stats = analyze_dataset(folder_path)
        visualize_results(stats)
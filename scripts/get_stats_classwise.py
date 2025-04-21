import json
import os
from collections import defaultdict


def analyze_json_files(folder_path):
    # Initialize counters
    class_distribution = defaultdict(int)
    attribute_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Process each frame in the JSON
                if "frames" in data:
                    for frame in data["frames"]:
                        if "objects" in frame:
                            for obj in frame["objects"]:
                                # Count class distribution
                                category = obj.get("category", "unknown")
                                class_distribution[category] += 1

                                # Count attributes
                                if "attributes" in obj:
                                    for attr, value in obj["attributes"].items():
                                        # Convert boolean to string for consistent counting
                                        if isinstance(value, bool):
                                            value = str(value).lower()
                                        attribute_counts[attr][value] += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return class_distribution, attribute_counts


def print_stats(class_dist, attr_counts):
    print("Class Distribution:")
    print("------------------")
    for category, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"{category}: {count}")

    print("\nAttribute Counts:")
    print("----------------")
    for attr, value_counts in attr_counts.items():
        print(f"\nAttribute: {attr}")
        for value, count in sorted(value_counts.items(), key=lambda x: -x[1]):
            print(f"  {value}: {count}")


if __name__ == "__main__":
    print(" class wise stats for training ")
    folder_path = "data/100k/train"

    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, attr_counts)

    print(" class wise stats for validation ")
    folder_path = "data/100k/val"

    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, attr_counts)

    print(" class wise stats for testing ")
    folder_path = "data/100k/test"

    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        class_dist, attr_counts = analyze_json_files(folder_path)
        print_stats(class_dist, attr_counts)

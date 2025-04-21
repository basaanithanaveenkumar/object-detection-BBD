import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def analyze_coco_bbox_distribution(json_path):
    # Load COCO annotations
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create dictionary for image dimensions
    image_id_to_size = {
        img["id"]: (img["width"], img["height"]) for img in data["images"]
    }

    areas = []
    aspect_ratios = []

    # Process all annotations
    for ann in data["annotations"]:
        # Get image dimensions
        image_id = ann["image_id"]
        img_w, img_h = image_id_to_size[image_id]

        # COCO bbox format: [x, y, width, height] in pixels
        w_px = ann["bbox"][2]
        h_px = ann["bbox"][3]

        # Calculate normalized dimensions and area
        w_norm = w_px / img_w
        h_norm = h_px / img_h
        area = w_norm * h_norm  # Relative to image area

        # Calculate aspect ratio (using original pixel dimensions)
        aspect_ratio = w_px / h_px if h_px != 0 else 0

        areas.append(area)
        aspect_ratios.append(aspect_ratio)

    # Convert to numpy arrays
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)

    # Print statistics
    print("Size Statistics:")
    print(f" - Small boxes (<2% area): {(areas < 0.02).mean()*100:.1f}%")
    print(
        f" - Medium boxes (2-10% area): {((areas >= 0.02) & (areas < 0.1)).mean()*100:.1f}%"
    )
    print(f" - Large boxes (>10% area): {(areas >= 0.1).mean()*100:.1f}%")

    print("\nAspect Ratio Statistics:")
    print(f" - Tall boxes (ratio < 0.5): {(aspect_ratios < 0.5).mean()*100:.1f}%")
    print(
        f" - Square boxes (0.5 <= ratio < 2): {((aspect_ratios >= 0.5) & (aspect_ratios < 2)).mean()*100:.1f}%"
    )
    print(f" - Wide boxes (ratio >= 2): {(aspect_ratios >= 2).mean()*100:.1f}%")


# Usage
print("Training analysis - Size and AS Statistics")
analyze_coco_bbox_distribution("data/100k/train/_annotations.coco.json")

print()

print("Test set analysis - Size and AS Statistics")
analyze_coco_bbox_distribution("data/100k/test/_annotations.coco.json")

print()
print("Validation set analysis - Size and AS Statistics")
analyze_coco_bbox_distribution("data/100k/valid/_annotations.coco.json")
